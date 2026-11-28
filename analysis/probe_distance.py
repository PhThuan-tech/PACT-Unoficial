"""Probe distances between all task adapters, and alpha under different tau (why is alpha always ~0.9-0.97?).

    python analysis/probe_distance.py --run-dir outputs/pact_lr5e5_s42 \\
        --set data_root=/kaggle/working/data download=false model.pretrained=<clip .pt> num_workers=8
    python analysis/probe_distance.py --run-dir ... --no-real      # Gaussian probes only: no dataset needed, CPU ok

Needs the run's state.pt + config.yaml and the CLIP weights. With real probes (default) also the datasets; a
few minutes on one GPU.

Question: PACT gates its constraints with alpha_t = exp(-d_{j*} / tau) (Eq. 12-14), d = normalised Frobenius
distance between probe responses of two adapters. In our runs alpha stays at ~0.9-0.97 for every task ("frozen"
everywhere), so the phase-like transition never happens. Is tau (0.1, an assumption) too large for the scale of d,
or does d not separate tasks at all? And do Gaussian probes (paper) see the same structure as real inputs?

Measures, with the response of a model "m" = concatenation over the adapter layers of m(z) for the probes z:
    D[a][b]   d between every pair among {pretrained MLP, adapter 1..T}          (Eq. 12, symmetric)
    d_min[t]  min_{j<t} D[t][j] (task 1: D[1][MLP]), the value training used      (checked against log.jsonl)
    alpha[t]  exp(-d_min[t] / tau) for a grid of tau
    contrast  mean D between adapters / mean D to the MLP: ~1 means adapters are about as far from each
              other as from the MLP (no task structure), << 1 or >> 1 means structure
Two probe sets: "gaussian" = N(0, I_d) with the training seed and size (what PACT uses), and "real" = tokens
entering each adapter layer when frozen CLIP sees real test images of all datasets (pooled, sampled).
With real probes the script also gives D per source dataset (tokens of one dataset only), to see whether
adapters differ more on their own domain.

Results: <run-dir>/probe_distance.json and probe_distance.png (heatmaps, alpha vs tau).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAUS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def responses(model, probes):
    """List over models [MLP, adapter 1..T] of per-layer responses to per-layer probes."""
    import torch
    out = []
    with torch.no_grad():
        out.append([w.base(z).float() for w, z in zip(model.wrappers, probes)])
        for t in range(model.num_adapters):
            out.append([w.adapters[t](z).float() for w, z in zip(model.wrappers, probes)])
    return out


def distance_matrix(resp):
    n = len(resp)
    sq = [sum(h.pow(2).sum().item() for h in r) for r in resp]
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            diff = sum((x - y).pow(2).sum().item() for x, y in zip(resp[a], resp[b]))
            D[a, b] = D[b, a] = diff / (1e-8 + sq[a] + sq[b])
    return D


def summarise(D, names):
    T = len(names)
    d_min = [D[1, 0]] + [min(D[t, j] for j in range(1, t)) for t in range(2, T + 1)]
    nearest = ["MLP"] + [names[min(range(1, t), key=lambda j: D[t, j]) - 1] for t in range(2, T + 1)]
    between = D[1:, 1:][np.triu_indices(T, 1)]
    to_mlp = D[1:, 0]
    alpha = {str(tau): [math.exp(-d / tau) for d in d_min] for tau in TAUS}
    return {"D": D.tolist(), "d_min": d_min, "nearest": nearest, "d_mlp": to_mlp.tolist(),
            "mean_between": float(between.mean()), "mean_to_mlp": float(to_mlp.mean()),
            "contrast": float(between.mean() / max(to_mlp.mean(), 1e-12)), "alpha": alpha}


def cluster_order(D):
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        return leaves_list(linkage(squareform(D, checks=False), "average")).tolist()
    except Exception:  # noqa: BLE001
        return list(range(len(D)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=2000, help="accepted for the notebook; real probes use --images")
    ap.add_argument("--no-real", action="store_true", help="Gaussian probes only (no dataset needed)")
    ap.add_argument("--images", type=int, default=32, help="test images per dataset for real probes")
    ap.add_argument("--tokens", type=int, default=2048, help="real probe tokens per layer (pooled over datasets)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, same as pact.run_mtil")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from torch.utils.data import DataLoader

    from alignment_drift import load_run
    from pact.phase import ProbeBank, _layer_dim

    cfg, device, model, clip_model, preprocess, tokenizer, names = load_run(args.run_dir, args.set, 0)
    T, t0 = len(names), time.time()
    result = {"order": names, "taus": list(TAUS), "tau_used": cfg["pact"]["tau"]}

    # ------------------------------------------------------------------ Gaussian probes (as in training)
    bank = ProbeBank(_layer_dim(model.wrappers[0]), cfg["pact"]["num_probes"], cfg["pact"]["probe_seed"], device)
    gauss = summarise(distance_matrix(responses(model, [bank.z] * len(model.wrappers))), names)
    result["gaussian"] = gauss

    # check against what training logged (last record of each task that has d_min)
    logged = {}
    log_path = os.path.join(args.run_dir, "log.jsonl")
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            r = json.loads(line)
            if "d_min" in r and r.get("task") in names:
                logged[r["task"]] = {"d_min": r["d_min"], "d_mlp": r.get("d_mlp"), "alpha": r.get("alpha")}
    result["logged"] = logged

    # ------------------------------------------------------------------ real probes (tokens of frozen CLIP)
    if not args.no_real:
        from pact.data import build_mtil_dataset, eval_subset
        captured = {i: [] for i in range(len(model.wrappers))}
        hooks = [w.register_forward_pre_hook(lambda mod, inp, i=i: captured[i].append(inp[0].detach()))
                 for i, w in enumerate(model.wrappers)]
        per_ds_tokens = {}
        gen = torch.Generator().manual_seed(0)
        for name in names:
            _, test, _, _ = build_mtil_dataset(name, cfg["data_root"], preprocess, cfg["download"])
            test = eval_subset(test, args.images)
            for i in captured:
                captured[i].clear()
            with torch.no_grad(), model.snapshot(0):
                for x, _ in DataLoader(test, batch_size=args.images, shuffle=False, num_workers=cfg["num_workers"]):
                    model.encode_image(x.to(device), "base")
            per_layer = []
            for i in captured:
                tok = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured[i]]).float()
                idx = torch.randperm(len(tok), generator=gen)[: args.tokens]
                per_layer.append(tok[idx.to(tok.device)])
            per_ds_tokens[name] = per_layer
        for h in hooks:
            h.remove()
        share = max(1, args.tokens // T)
        pooled = [torch.cat([per_ds_tokens[n][i][:share] for n in names]) for i in range(len(model.wrappers))]
        real = summarise(distance_matrix(responses(model, pooled)), names)
        real["per_dataset"] = {}
        for name in names:
            Dn = distance_matrix(responses(model, per_ds_tokens[name]))
            real["per_dataset"][name] = {"contrast": float(Dn[1:, 1:][np.triu_indices(T, 1)].mean()
                                                           / max(Dn[1:, 0].mean(), 1e-12)),
                                         "own_adapter_to_mlp": float(Dn[names.index(name) + 1, 0]),
                                         "mean_to_mlp": float(Dn[1:, 0].mean())}
        result["real"] = real

    # ------------------------------------------------------------------ report
    def show(tag, s):
        print(f"\n== {tag} probes: mean d between adapters {s['mean_between']:.5f}, mean d to MLP "
              f"{s['mean_to_mlp']:.5f}, contrast {s['contrast']:.2f}")
        print(f"{'task':14s}{'d_min':>10s}{'nearest':>14s}{'d_mlp':>10s}" + "".join(f"{'a@' + str(t):>9s}" for t in TAUS))
        for t, name in enumerate(names):
            print(f"{name:14s}{s['d_min'][t]:10.5f}{s['nearest'][t]:>14s}{s['d_mlp'][t]:10.5f}"
                  + "".join(f"{s['alpha'][str(tau)][t]:9.3f}" for tau in TAUS))
    show("gaussian", gauss)
    if logged:
        print("\ncheck vs training log (gaussian d_min recomputed / logged at the end of the task):")
        for t, name in enumerate(names):
            if name in logged:
                print(f"  {name:14s}{gauss['d_min'][t]:.5f} / {logged[name]['d_min']:.5f}   alpha logged "
                      f"{logged[name]['alpha']:.3f}")
    if "real" in result:
        show("real", result["real"])
        print("\nper source dataset (real tokens of that dataset only): contrast, own adapter vs mean distance to MLP")
        for name, v in result["real"]["per_dataset"].items():
            print(f"  {name:14s} contrast {v['contrast']:.2f}   own {v['own_adapter_to_mlp']:.5f}   "
                  f"mean {v['mean_to_mlp']:.5f}")
    for tag in ("gaussian", "real"):
        if tag in result:
            spread = {tau: (min(a), max(a)) for tau, a in result[tag]["alpha"].items()}
            good = [tau for tau, (lo, hi) in spread.items() if lo < 0.3 and hi > 0.7]
            print(f"\n{tag}: tau values giving both melted (<0.3) and frozen (>0.7) tasks: {good or 'none'}")

    out = os.path.join(args.run_dir, "probe_distance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\nwritten: {out} ({(time.time() - t0) / 60:.1f} min)")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("no matplotlib, figure skipped:", e)
        return
    tags = [t for t in ("gaussian", "real") if t in result]
    fig, axes = plt.subplots(1, len(tags) + 1, figsize=(6 * len(tags) + 6, 5.5))
    labels = ["MLP"] + names
    for ax, tag in zip(axes, tags):
        D = np.asarray(result[tag]["D"])
        order = cluster_order(D)
        im = ax.imshow(D[np.ix_(order, order)], cmap="viridis")
        ax.set_xticks(range(len(order)), [labels[i] for i in order], rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)), [labels[i] for i in order], fontsize=7)
        ax.set_title(f"{tag} probes: d (clustered order)"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[-1]
    for tag, style in (("gaussian", "-"), ("real", "--")):
        if tag in result:
            for tau in TAUS:
                ax.plot(range(1, T + 1), result[tag]["alpha"][str(tau)], style, marker="o", ms=3,
                        label=f"{tag} tau={tau}" if tag == "gaussian" else None)
    ax.set_xticks(range(1, T + 1), names, rotation=90, fontsize=7)
    ax.set_ylabel("alpha_t"); ax.set_title("alpha per task (solid: gaussian, dashed: real)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(args.run_dir, "probe_distance.png"), dpi=110)
    print("figure: probe_distance.png")


if __name__ == "__main__":
    main()
