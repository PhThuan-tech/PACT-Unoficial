"""Oracle best snapshot on not-yet-learned datasets (step 4 of the research plan).

    python analysis/oracle_snapshot.py --run-dir outputs/pact_s42 --max-samples 2000 \\
        --set data_root=/kaggle/working/data download=false model.pretrained=<clip .pt> num_workers=8

Needs the run's state.pt (adapters + heads) and config.yaml, plus the datasets and CLIP weights: a GPU job.

Question: is there ANY snapshot that helps a dataset the model has not learned yet, compared with frozen CLIP
(zero-shot)? Dataset j (0-based) is unseen at snapshots k = 1..j. With the text head (head 0, the one used for
unseen data) the script measures acc[j][k] for k = 0 (frozen CLIP) .. j and reports, using the Transfer definition
of pact.metrics (Transfer_j = mean over k = 1..j):
    zero_shot   : acc[j][0]                                       (what the router's CLIP fallback gets)
    latest      : mean_k acc[j][k]                                (the `oracle_latest` protocol)
    oracle_best : mean_k max(acc[j][0], max_{k'<=k} acc[j][k'])   (a router with hindsight picks the best of CLIP and
                                                                  the snapshots learned so far, for every row)
    best_snap   : which snapshot is best for dataset j and by how much it beats zero-shot
If oracle_best is not above zero_shot (beyond sampling noise, about 1.4 points for 1000 test images), no routing
scheme can lift Transfer above zero-shot and the "send unseen data to a helpful snapshot" idea is dead.
Results: <run-dir>/oracle_snapshot.json plus a printed table.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def snapshot_table(evaluator, names):
    """acc[j][k] (%), k = 0..j: dataset j under snapshot k with the frozen text head, while still unseen."""
    return [[float(evaluator.accuracy(name, k, 0)) for k in range(j + 1)] for j, name in enumerate(names)]


def summarise(table, names):
    rows = {}
    for j, name in enumerate(names):
        if j == 0:
            continue
        a = np.asarray(table[j], dtype=float)           # a[0] = zero-shot, a[1..j] = snapshots 1..j
        zs = a[0]
        running = np.maximum.accumulate(a[1:])           # best snapshot among 1..k
        best_k = int(a[1:].argmax()) + 1
        rows[name] = {"zero_shot": float(zs), "latest": float(a[1:].mean()),
                      "oracle_best": float(np.maximum(zs, running).mean()),
                      "best_snapshot": names[best_k - 1], "best_snapshot_acc": float(a[best_k]),
                      "gain_best_vs_zero_shot": float(a[best_k] - zs),
                      "snapshots_above_zero_shot": int((a[1:] > zs + 1e-9).sum()), "snapshots": j}
    keys = ("zero_shot", "latest", "oracle_best")
    summary = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys} if rows else {}
    return rows, summary


def report(rows, summary):
    print(f"{'dataset':14s}{'zero-shot':>10s}{'latest':>9s}{'oracle':>9s}  best snapshot (acc, vs zero-shot)   #snap > zs")
    for name, r in rows.items():
        print(f"{name:14s}{r['zero_shot']:10.2f}{r['latest']:9.2f}{r['oracle_best']:9.2f}  "
              f"{r['best_snapshot']:>12s} ({r['best_snapshot_acc']:.2f}, {r['gain_best_vs_zero_shot']:+.2f})"
              f"      {r['snapshots_above_zero_shot']}/{r['snapshots']}")
    if summary:
        print(f"{'Transfer':14s}{summary['zero_shot']:10.2f}{summary['latest']:9.2f}{summary['oracle_best']:9.2f}")
        gain = summary["oracle_best"] - summary["zero_shot"]
        print(f"\noracle_best - zero_shot = {gain:+.2f} points of Transfer. "
              + ("Within sampling noise: no routing scheme lifts Transfer above zero-shot." if gain < 0.5 else
                 "Some snapshots do help unseen data: worth a training-free proxy (entropy / margin) to pick them."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=2000, help="test images per dataset (0 = all)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, same as pact.run_mtil")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    import torch
    import yaml

    from pact import distributed as dist
    from pact.config import load_config
    from pact.data import build_mtil_dataset, eval_subset
    from pact.engine import build_clip, resolve_device
    from pact.evaluation import MTILEvaluator
    from pact.model import PACTCLIP
    from pact.text_head import build_text_classifier

    with open(os.path.join(args.run_dir, "config.yaml"), encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    saved.pop("world_size", None)  # written by run_mtil next to the config, not a config key
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(saved, tmp)
    cfg = load_config(tmp.name, [*args.set, f"eval.max_samples={args.max_samples or 'null'}"])
    os.unlink(tmp.name)
    device = resolve_device(cfg["device"])
    clip_model, preprocess, tokenizer = build_clip(cfg, device)
    model = PACTCLIP(clip_model, cfg["adapter"], cfg["head"]).to(device)
    state = torch.load(os.path.join(args.run_dir, "state.pt"), map_location="cpu", weights_only=False)
    model.load_adapter_state(state["adapters"])
    names = cfg["mtil"]["order"]
    data = {}
    for name in names:
        train, test, classnames, templates = build_mtil_dataset(name, cfg["data_root"], preprocess, cfg["download"])
        text = build_text_classifier(clip_model, tokenizer, classnames, templates, device,
                                     cfg["eval"]["template_mode"]).cpu()
        data[name] = {"train": train, "test": eval_subset(test, cfg["eval"]["max_samples"]), "text": text}
    evaluator = MTILEvaluator(model, data, names, cfg, device)
    table = snapshot_table(evaluator, names)
    rows, summary = summarise(table, names)
    report(rows, summary)
    out = os.path.join(args.run_dir, "oracle_snapshot.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"order": names, "acc": table, "per_dataset": rows, "summary": summary,
                   "max_samples": args.max_samples}, f, indent=1)
    print(f"\nwritten: {out}")
    dist.cleanup()


if __name__ == "__main__":
    main()
