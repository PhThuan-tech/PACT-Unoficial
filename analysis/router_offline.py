"""Offline analysis of the test-time router (no GPU, no retraining).

    python analysis/router_offline.py outputs/pact_paper/evaluator.pt
    python analysis/router_offline.py outputs/pact_paper/evaluator.pt --deltas -0.1 -0.05 0 0.05 0.1 --out analysis_out

evaluator.pt (a few MB, always in the notebook's result zip) or state.pt (also holds the adapters)
both work.

What it reads (all saved by pact/run_mtil.py in state.pt, key "evaluator"):
  - batch_feats[name]: mean frozen-CLIP feature of every test batch (B, D)
  - prototypes[k-1]:   (normalised mean training feature of task k, acceptance threshold)
  - counts[(name, k, head)]: per-batch correct counts of snapshot k with head `head` (cached)

What it does: replays the router of pact/evaluation.py for every row t (tasks 1..t learned)
with all thresholds shifted by delta (delta > 0 = stricter, more batches fall back to CLIP):
    score_k = batch_mean . prototype_k,  k* = argmax (score_k - thr_k - delta),
    accept k* if score_k* - thr_k* - delta >= 0, else frozen CLIP (k = 0).

Outputs:
  1. routing quality per delta: learned batches sent to their own task / to CLIP / to another
     task; unseen batches sent to CLIP / to a learned snapshot;
  2. Transfer / Avg / Last of the replayed router per delta. A decision needs the counts of
     (dataset, snapshot, head); only combinations evaluated during the run are cached, so some
     deltas cannot be scored: they are reported with the share of batches that is missing;
  3. confusion matrices (last row and all rows) at delta = 0;
  4. a consistency check: at delta = 0 the replay must reproduce the run's own router protocol.
Results are also written to <out>/router_offline.json (+ router_offline.png if matplotlib).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pact.metrics import mtil_metrics  # noqa: E402


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    ev = state.get("evaluator")
    if not ev or not ev.get("prototypes"):
        raise SystemExit(f"{path}: no router data (needs a run with 'router' in eval.protocols, "
                         "written by the snapshot-aware code)")
    counts = {}
    for key, v in ev["counts"].items():
        name, k, head = key.rsplit("|", 2)
        counts[(name, int(k), int(head))] = np.asarray(v)
    order = state["order"]
    return {"order": order, "counts": counts, "sizes": ev["sizes"],
            "feats": {n: torch.as_tensor(f).float() for n, f in ev["batch_feats"].items()},
            "protos": torch.stack([torch.as_tensor(p).float() for p, _ in ev["prototypes"]]),
            "thr": torch.tensor([float(t) for _, t in ev["prototypes"]]),
            "acc": state.get("acc", {}), "router_log": ev.get("router_log", []),
            "learned_heads": (bool(state["adapters"].get("heads")) if "adapters" in state
                              else bool(state.get("learned_heads")))}


def route(d, name, t, delta):
    """Decision per batch (0 = CLIP, k = snapshot k) with tasks 1..t learned."""
    scores = d["feats"][name] @ d["protos"][:t].t()           # (B, t) mean cosines
    margin = scores - d["thr"][:t] - delta
    best = margin.argmax(dim=1)
    ok = margin.gather(1, best[:, None]).squeeze(1) >= 0
    return torch.where(ok, best + 1, torch.zeros_like(best)).numpy()


def head_for(d, j, k):
    return j if (d["learned_heads"] and 1 <= j <= k) else 0


def replay(d, delta):
    names = d["order"]
    T = d["protos"].shape[0]  # one prototype per learned task = number of rows
    acc, missing_batches, total_batches = [], 0, 0
    stats = {"learned_own": 0, "learned_clip": 0, "learned_other": 0, "learned_n": 0,
             "unseen_clip": 0, "unseen_snap": 0, "unseen_n": 0}
    conf_all = np.zeros((len(names), T + 1), dtype=np.int64)
    conf_last = np.zeros_like(conf_all)
    for t in range(1, T + 1):
        row = []
        for j, name in enumerate(names, start=1):
            dec = route(d, name, t, delta)
            sizes = np.asarray(d["sizes"][name])
            for k in dec:
                conf_all[j - 1, k] += 1
                if t == T:
                    conf_last[j - 1, k] += 1
            if j <= t:
                stats["learned_n"] += len(dec)
                stats["learned_own"] += int((dec == j).sum())
                stats["learned_clip"] += int((dec == 0).sum())
                stats["learned_other"] += int(((dec != j) & (dec != 0)).sum())
            else:
                stats["unseen_n"] += len(dec)
                stats["unseen_clip"] += int((dec == 0).sum())
                stats["unseen_snap"] += int((dec != 0).sum())
            correct, complete = 0, True
            for k in np.unique(dec):
                sel = dec == k
                key = (name, int(k), head_for(d, j, int(k)))
                total_batches += int(sel.sum())
                if key not in d["counts"]:
                    complete = False
                    missing_batches += int(sel.sum())
                    continue
                correct += d["counts"][key][sel].sum()
            row.append(100.0 * correct / max(sizes.sum(), 1) if complete else None)
        acc.append(row)
    rate = lambda a, n: round(100.0 * stats[a] / max(stats[n], 1), 2)
    out = {"delta": delta,
           "learned_to_own_%": rate("learned_own", "learned_n"),
           "learned_to_clip_%": rate("learned_clip", "learned_n"),
           "learned_to_other_%": rate("learned_other", "learned_n"),
           "unseen_to_clip_%": rate("unseen_clip", "unseen_n"),
           "unseen_to_snapshot_%": rate("unseen_snap", "unseen_n"),
           "missing_batches_%": round(100.0 * missing_batches / max(total_batches, 1), 2)}
    if all(a is not None for r in acc for a in r):
        out["summary"] = mtil_metrics(acc, names)[1]
        out["acc_matrix"] = acc
    else:
        out["summary"] = None
    return out, conf_all, conf_last


def print_confusion(title, conf, names):
    cols = ["CLIP"] + [n[:8] for n in names[:conf.shape[1] - 1]]
    print(f"\n{title} (rows: test dataset, cols: where its batches were sent, % of batches)")
    print(f"{'':12s}" + "".join(f"{c:>9s}" for c in cols))
    for i, n in enumerate(names):
        tot = max(conf[i].sum(), 1)
        print(f"{n[:12]:12s}" + "".join(f"{100.0 * v / tot:9.1f}" for v in conf[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
    ap.add_argument("--out", default=None, help="output folder (default: next to state.pt)")
    args = ap.parse_args()
    d = load(args.state)
    names = d["order"]
    deltas = sorted(set(args.deltas) | {0.0})

    results, conf0 = [], None
    print(f"{'delta':>7s}{'own':>8s}{'->CLIP':>8s}{'->other':>9s}{'unseen->CLIP':>14s}"
          f"{'unseen->snap':>14s}{'missing':>9s}{'Transfer':>10s}{'Avg':>8s}{'Last':>8s}")
    for delta in deltas:
        r, conf_all, conf_last = replay(d, delta)
        if delta == 0.0:
            conf0 = (conf_all, conf_last)
        results.append(r)
        s = r["summary"]
        tail = (f"{s['Transfer']:10.2f}{s['Avg']:8.2f}{s['Last']:8.2f}" if s
                else f"{'(thiếu counts)':>26s}")
        print(f"{delta:7.2f}{r['learned_to_own_%']:8.1f}{r['learned_to_clip_%']:8.1f}"
              f"{r['learned_to_other_%']:9.1f}{r['unseen_to_clip_%']:14.1f}"
              f"{r['unseen_to_snapshot_%']:14.1f}{r['missing_batches_%']:9.1f}{tail}")
    print("\nown / ->CLIP / ->other: batches of LEARNED datasets sent to their own task / to frozen CLIP"
          " / to another task (%). unseen->...: batches of not-yet-learned datasets."
          "\nmissing: batches whose (dataset, snapshot, head) was never evaluated in the run, so the"
          " accuracy of that delta cannot be computed offline.")

    print_confusion("Confusion, last row (all tasks learned), delta=0", conf0[1], names)
    print_confusion("Confusion, all rows, delta=0", conf0[0], names)

    check = None
    ref = d["acc"].get("router") if isinstance(d["acc"], dict) else None
    zero = next(r for r in results if r["delta"] == 0.0)
    if ref and zero.get("acc_matrix"):
        diff = max(abs(a - b) for ra, rb in zip(zero["acc_matrix"], ref) for a, b in zip(ra, rb))
        check = bool(diff < 1e-6)
        print(f"\nconsistency: replay at delta=0 vs the run's 'router' protocol: max |diff| = {diff:.2e}"
              f" -> {'OK' if check else 'KHÁC (kiểm tra lại code router)'}")

    out = args.out or os.path.dirname(os.path.abspath(args.state))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "router_offline.json"), "w", encoding="utf-8") as f:
        json.dump({"order": names, "sweep": results, "consistency_ok": check,
                   "confusion_delta0_last": conf0[1].tolist(),
                   "confusion_delta0_all": conf0[0].tolist()}, f, indent=2, default=float)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        xs = [r["delta"] for r in results]
        for key, lab in [("learned_to_own_%", "learned -> own task"),
                         ("unseen_to_clip_%", "unseen -> CLIP")]:
            axes[0].plot(xs, [r[key] for r in results], marker="o", label=lab)
        axes[0].set_xlabel("threshold shift delta"); axes[0].set_ylabel("% batches")
        axes[0].set_title("routing"); axes[0].legend()
        pts = [(r["delta"], r["summary"]) for r in results if r["summary"]]
        for key in ("Transfer", "Avg", "Last"):
            if pts:
                axes[1].plot([p[0] for p in pts], [p[1][key] for p in pts], marker="o", label=key)
        axes[1].set_xlabel("threshold shift delta"); axes[1].set_title("accuracy (scored deltas)")
        axes[1].legend()
        plt.tight_layout(); fig.savefig(os.path.join(out, "router_offline.png"), dpi=110)
    except Exception as e:  # plotting is optional
        print("(no plot:", e, ")")
    print(f"\nwritten: {os.path.join(out, 'router_offline.json')}")


if __name__ == "__main__":
    main()
