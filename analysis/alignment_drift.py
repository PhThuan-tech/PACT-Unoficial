"""Image-text alignment across task snapshots (does the image encoder drift away from the frozen text side?).

    python analysis/alignment_drift.py --run-dir outputs/pact_lr5e5_s42 --max-samples 2000 \\
        --set data_root=/kaggle/working/data download=false model.pretrained=<clip .pt> num_workers=8

Needs the run's state.pt + config.yaml, the datasets and the CLIP weights: a GPU job (about 15-25 min on one L4
for 11 datasets x 2000 test images; the test images of a dataset are decoded once and kept in memory).

Question: PACT only adapts the image encoder; the text side (class embeddings) stays frozen. Transfer on
not-yet-learned datasets is below zero-shot in every run so far. Is that because image features drift away from
the frozen text embeddings, and does the drift hit unseen data (Transfer) or only seen data?

For every snapshot k = 0 (frozen CLIP) .. T, branch (cond = mean of adapters 1..k, the evaluated model; marg =
adapter k alone) and dataset j, with f = normalised image features and t_c = frozen class text embeddings:
    acc_text   accuracy with the frozen text classifier (= what Transfer uses on unseen data)
    acc_head   accuracy with the task's learned head (seen datasets only): head vs text shows how much the
               learned head compensates for misalignment
    pos        mean cos(f, t_y)                  image to its own class text
    neg        mean cos(f, t_c), c != y          image to the other class texts
    margin     mean [cos(f, t_y) - max_{c!=y} cos(f, t_c)]   (> 0 means correctly ranked)
    gap        || mean f - mean t ||             modality gap (centroid distance, MG-CLIP / Liang et al.)
    self_sim   mean cos(f_k(x), f_0(x))          per-image similarity to frozen CLIP features
    cka        linear CKA(F_k, F_0)              representation similarity to frozen CLIP
Cells are "seen" when dataset j was learned at snapshot k (j <= k, 1-based), else "unseen".
The script also reports, over unseen cond cells, how well each change (vs k = 0) predicts the change of
acc_text (Pearson r): a high r for pos / margin / self_sim means alignment drift explains the Transfer loss.

Results: <run-dir>/alignment_drift.json, alignment_drift.png (curves), alignment_drift_unseen.png (heatmap).
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
METRICS = ("acc_text", "acc_head", "pos", "neg", "margin", "gap", "self_sim", "cka")


def load_run(run_dir, overrides, max_samples):
    import torch
    import yaml

    from pact.config import load_config
    from pact.engine import allow_tf32, build_clip, resolve_device
    from pact.model import PACTCLIP

    allow_tf32()  # same as the repo evaluator; several times faster fp32 matmuls on L4 / A100

    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    saved.pop("world_size", None)  # written by run_mtil next to the config, not a config key
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(saved, tmp)
    cfg = load_config(tmp.name, [*overrides, f"eval.max_samples={max_samples or 'null'}"])
    os.unlink(tmp.name)
    device = resolve_device(cfg["device"])
    clip_model, preprocess, tokenizer = build_clip(cfg, device)
    model = PACTCLIP(clip_model, cfg["adapter"], cfg["head"]).to(device)
    state = torch.load(os.path.join(run_dir, "state.pt"), map_location="cpu", weights_only=False)
    model.load_adapter_state(state["adapters"])
    model.eval()
    names = cfg["mtil"]["order"]
    if model.num_adapters != len(names):
        raise SystemExit(f"state.pt holds {model.num_adapters} tasks, the order has {len(names)}: run not finished")
    return cfg, device, model, clip_model, preprocess, tokenizer, names


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    num = (y.t() @ x).pow(2).sum()
    den = (x.t() @ x).pow(2).sum().sqrt() * (y.t() @ y).pow(2).sum().sqrt()
    return (num / den.clamp_min(1e-12)).item()


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=2000, help="test images per dataset (0 = all)")
    ap.add_argument("--branches", nargs="*", default=["cond", "marg"], choices=["cond", "marg"])
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of the run's datasets (default: all)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, same as pact.run_mtil")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    import torch
    from torch.utils.data import DataLoader

    from pact.data import build_mtil_dataset, eval_subset
    from pact.text_head import build_text_classifier

    cfg, device, model, clip_model, preprocess, tokenizer, names = load_run(args.run_dir, args.set, args.max_samples)
    T = len(names)
    use = args.datasets or names
    bs = cfg["eval"]["batch_size"]
    records, t0 = [], time.time()

    for name in use:
        j = names.index(name) + 1  # 1-based task index of this dataset
        _, test, classnames, templates = build_mtil_dataset(name, cfg["data_root"], preprocess, cfg["download"])
        test = eval_subset(test, cfg["eval"]["max_samples"])
        text = build_text_classifier(clip_model, tokenizer, classnames, templates, device,
                                     cfg["eval"]["template_mode"]).float().to(device)
        # decode the test images once (fp16 on the device when it fits, else on CPU)
        imgs, labels = [], []
        for x, y in DataLoader(test, batch_size=bs, shuffle=False, num_workers=cfg["num_workers"]):
            imgs.append(x.half()); labels.append(y)
        imgs, labels = torch.cat(imgs), torch.cat(labels).to(device)
        on_dev = device.type == "cuda" and imgs.numel() * 2 < 3 * 2**30
        if on_dev:
            imgs = imgs.to(device)

        @torch.no_grad()
        def features(k, branch):
            out = []
            with model.snapshot(k):
                for i in range(0, len(imgs), bs):
                    x = imgs[i:i + bs].to(device).float()
                    out.append(model.encode_image(x, branch).float())
            return torch.cat(out)

        f0 = features(0, "base")
        onehot = torch.nn.functional.one_hot(labels, len(classnames)).bool()
        t_mean = text.mean(0)
        for k in range(T + 1):
            for branch in (["base"] if k == 0 else args.branches):
                f = f0 if k == 0 else features(k, branch)
                sims = f @ text.t()                                   # (N, C) cosines
                pos = sims[onehot]
                other = sims.masked_fill(onehot, -2.0).max(1).values
                neg = sims.masked_fill(onehot, 0.0).sum(1) / max(len(classnames) - 1, 1)
                rec = {"k": k, "dataset": name, "j": j, "branch": branch, "seen": bool(1 <= j <= k),
                       "acc_text": 100.0 * (sims.argmax(1) == labels).float().mean().item(),
                       "acc_head": None,
                       "pos": pos.mean().item(), "neg": neg.mean().item(), "margin": (pos - other).mean().item(),
                       "gap": (f.mean(0) - t_mean).norm().item(),
                       "self_sim": (f * f0).sum(1).mean().item(), "cka": linear_cka(f, f0)}
                if rec["seen"] and model.use_heads:
                    w = model.head_weights(j, "marg" if branch == "marg" else "cond").float()
                    rec["acc_head"] = 100.0 * ((f @ w.t()).argmax(1) == labels).float().mean().item()
                records.append(rec)
        print(f"[{name}] {len(labels)} images, {T * len(args.branches) + 1} passes, "
              f"{(time.time() - t0) / 60:.1f} min so far", flush=True)
        del imgs

    # ------------------------------------------------------------------ summaries
    zero = {r["dataset"]: r for r in records if r["k"] == 0}
    for r in records:
        z = zero[r["dataset"]]
        for m in ("acc_text", "pos", "neg", "margin", "gap", "self_sim", "cka"):
            r[f"d_{m}"] = r[m] - z[m]

    def mean_of(rows, m):
        vals = [r[m] for r in rows if r[m] is not None]
        return float(np.mean(vals)) if vals else None

    by_k = []
    for k in range(T + 1):
        for branch in (["base"] if k == 0 else args.branches):
            for status in ("seen", "unseen"):
                rows = [r for r in records if r["k"] == k and r["branch"] == branch and r["seen"] == (status == "seen")]
                if rows:
                    by_k.append({"k": k, "branch": branch, "status": status, "n": len(rows),
                                 **{m: mean_of(rows, m) for m in METRICS}})

    unseen = [r for r in records if r["k"] >= 1 and r["branch"] == "cond" and not r["seen"]]
    predictors = {m: pearson([r[f"d_{m}"] for r in unseen], [r["d_acc_text"] for r in unseen])
                  for m in ("pos", "neg", "margin", "gap", "self_sim", "cka")}

    # ------------------------------------------------------------------ report
    print("\nMean per snapshot (cond branch; k = 0 is frozen CLIP)")
    print(f"{'k':>3s} {'status':7s}" + "".join(f"{m:>10s}" for m in METRICS))
    for row in by_k:
        if row["branch"] not in ("cond", "base"):
            continue
        cells = "".join(f"{row[m]:10.3f}" if row[m] is not None and m not in ("acc_text", "acc_head")
                        else (f"{row[m]:10.2f}" if row[m] is not None else f"{'-':>10s}") for m in METRICS)
        print(f"{row['k']:3d} {row['status']:7s}{cells}")
    if "marg" in args.branches:
        last = {(r["status"]): r for r in by_k if r["k"] == T and r["branch"] == "marg"}
        if "seen" in last:
            print(f"\nlast snapshot, marg branch, seen: acc_text {last['seen']['acc_text']:.2f}, "
                  f"acc_head {last['seen']['acc_head']:.2f}, pos {last['seen']['pos']:.3f}, cka {last['seen']['cka']:.3f}")

    print("\nUnseen datasets: change vs frozen CLIP, averaged over the snapshots where each is still unseen (cond)")
    print(f"{'dataset':14s}{'d_acc':>8s}{'d_pos':>9s}{'d_margin':>10s}{'d_gap':>8s}{'self_sim':>10s}{'cka':>7s}")
    for name in use:
        rows = [r for r in unseen if r["dataset"] == name]
        if rows:
            print(f"{name:14s}{mean_of(rows, 'd_acc_text'):+8.2f}{mean_of(rows, 'd_pos'):+9.4f}"
                  f"{mean_of(rows, 'd_margin'):+10.4f}{mean_of(rows, 'd_gap'):+8.4f}"
                  f"{mean_of(rows, 'self_sim'):10.4f}{mean_of(rows, 'cka'):7.3f}")
    print("\nPearson r with the change of unseen accuracy (cond, all unseen cells):")
    for m, r in predictors.items():
        print(f"  d_{m:9s} r = {r:+.3f}")

    out = os.path.join(args.run_dir, "alignment_drift.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"order": names, "datasets": use, "branches": args.branches, "max_samples": args.max_samples,
                   "records": records, "by_snapshot": by_k, "unseen_predictors": predictors}, f, indent=1)
    print(f"\nwritten: {out}")

    # ------------------------------------------------------------------ figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("no matplotlib, figures skipped:", e)
        return
    panels = ("acc_text", "pos", "margin", "gap", "self_sim", "cka")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.flat, panels):
        for branch, style in (("cond", "-"), ("marg", "--")):
            if branch not in args.branches:
                continue
            for status, color in (("seen", "tab:blue"), ("unseen", "tab:red")):
                pts = [(row["k"], row[m]) for row in by_k
                       if row["status"] == status and (row["branch"] == branch or row["k"] == 0) and row[m] is not None]
                if pts:
                    ax.plot(*zip(*pts), style, color=color, marker="o", ms=3, label=f"{status} ({branch})")
        ax.set_title(m); ax.set_xlabel("snapshot k (0 = CLIP)"); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Image-text alignment across snapshots ({os.path.basename(os.path.normpath(args.run_dir))})")
    fig.tight_layout(); fig.savefig(os.path.join(args.run_dir, "alignment_drift.png"), dpi=110)

    grid = np.full((len(use), T), np.nan)
    for r in unseen:
        grid[use.index(r["dataset"]), r["k"] - 1] = r["d_acc_text"]
    fig, ax = plt.subplots(figsize=(9, 5))
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(use)), use); ax.set_xticks(range(T), [f"{k + 1}" for k in range(T)])
    ax.set_xlabel("snapshot k"); ax.set_title("unseen accuracy - zero-shot (cond branch, frozen text head)")
    for (a, b), v in np.ndenumerate(grid):
        if math.isfinite(v):
            ax.text(b, a, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(args.run_dir, "alignment_drift_unseen.png"), dpi=110)
    print("figures: alignment_drift.png, alignment_drift_unseen.png")


if __name__ == "__main__":
    main()
