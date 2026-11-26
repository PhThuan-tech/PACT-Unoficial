"""Re-evaluate a finished MTIL run with another test-time branch (phase A of the gap analysis).

    python analysis/reeval_branch.py --run-dir outputs/pact_s42 --max-samples 2000 \\
        --set data_root=/kaggle/working/data download=false model.pretrained=<clip .pt> num_workers=8

Needs the run's state.pt (adapters + heads) and config.yaml, the datasets and the CLIP weights: a GPU job
(about 10 minutes on one L4 for 11 datasets at 2000 test images each).

Question: is part of the gap to the paper caused by the conditional branch "diluting" the adapter of the task
being evaluated? Snapshot k evaluated with
    cond/cond   mean of adapters 1..k, cond head k        (what the run reported; the repo default)
    marg/marg   adapter k only, marg head k               (the plasticity branch, trained by L_marg)
    marg/cond   adapter k only, cond head k               (for isolation runs: their marg head is never trained)
Unseen datasets always use the frozen text head; only the feature branch changes there.

Protocols: oracle_zeroshot (Last / Avg per learned task, task id known) and oracle_latest (adds Transfer of the
latest model on unseen data). Zero-shot passes (snapshot 0) are shared between variants.
Results: <run-dir>/reeval_branch.json plus printed tables (per variant Transfer / Avg / Last, and the accuracy
right after learning each task, with the paper's supp. Table 8 when the order is the full Order-I).
Use --max-samples 0 for full test sets: cond/cond then reproduces the run's own results.json exactly.
"""
import argparse
import copy
import json
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from pact.metrics import mtil_metrics  # noqa: E402

VARIANTS = {"cond/cond": ("cond", "cond"), "marg/marg": ("marg", "marg"), "marg/cond": ("marg", "cond")}
PAPER_LAST = {"Aircraft": 65.58, "Caltech101": 96.60, "CIFAR100": 91.21, "DTD": 80.53, "EuroSAT": 97.98,
              "Flowers": 98.59, "Food": 93.81, "MNIST": 99.53, "OxfordPet": 95.66, "StanfordCars": 89.90,
              "SUN397": 86.60}


def evaluate_variants(model, data, names, cfg, device, variants=tuple(VARIANTS),
                      protocols=("oracle_zeroshot", "oracle_latest")):
    """{variant: {"rows": {protocol: acc_matrix}, "summary": {protocol: metrics}}} for snapshot rows t = 1..T."""
    from pact.evaluation import MTILEvaluator

    zero_counts, sizes = {}, {}  # snapshot 0 = frozen CLIP + text head: identical for every variant
    out = {}
    for v in variants:
        branch, head_branch = VARIANTS[v]
        c = copy.deepcopy(cfg)
        c["eval"]["branch"], c["eval"]["head_branch"] = branch, head_branch
        c["eval"]["protocols"] = list(protocols)
        ev = MTILEvaluator(model, data, names, c, device)
        ev.counts.update(zero_counts)
        ev.sizes.update(sizes)
        rows = {p: [] for p in protocols}
        for t in range(1, len(names) + 1):
            r, _ = ev.row(t)
            for p in protocols:
                rows[p].append(r[p])
        zero_counts.update({k: c_ for k, c_ in ev.counts.items() if k[1] == 0})
        sizes.update(ev.sizes)
        out[v] = {"rows": rows, "summary": {p: mtil_metrics(rows[p], names)[1] for p in protocols}}
    return out


def report(results, names):
    protos = list(next(iter(results.values()))["summary"])
    print(f"{'variant':11s}" + "".join(f"{p + ' T/A/L':>32s}" for p in protos))
    for v, r in results.items():
        cells = "".join(f"{s['Transfer']:>12.2f}{s['Avg']:>10.2f}{s['Last']:>10.2f}" for s in
                        (r["summary"][p] for p in protos))
        print(f"{v:11s}{cells}")
    print(f"\naccuracy right after learning each task (oracle_zeroshot diagonal)")
    paper = all(n in PAPER_LAST for n in names) and len(names) == 11
    print(f"{'dataset':14s}" + "".join(f"{v:>11s}" for v in results) + ("     paper" if paper else ""))
    for j, n in enumerate(names):
        line = "".join(f"{r['rows']['oracle_zeroshot'][j][j]:11.2f}" for r in results.values())
        print(f"{n:14s}{line}" + (f"{PAPER_LAST[n]:10.2f}" if paper else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=2000, help="test images per dataset (0 = all)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--protocols", nargs="*", default=["oracle_zeroshot", "oracle_latest"],
                    choices=["oracle_zeroshot", "oracle_latest"])
    ap.add_argument("--set", nargs="*", default=[], help="overrides, same as pact.run_mtil")
    args = ap.parse_args()
    import torch
    import yaml

    from pact import distributed as dist
    from pact.config import load_config
    from pact.data import build_mtil_dataset, eval_subset
    from pact.engine import build_clip, resolve_device
    from pact.model import PACTCLIP
    from pact.text_head import build_text_classifier

    with open(os.path.join(args.run_dir, "config.yaml"), encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    saved.pop("world_size", None)
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
    if model.num_adapters != len(names):
        raise SystemExit(f"state.pt holds {model.num_adapters} tasks, the order has {len(names)}: run not finished")
    data = {}
    for name in names:
        _, test, classnames, templates = build_mtil_dataset(name, cfg["data_root"], preprocess, cfg["download"])
        text = build_text_classifier(clip_model, tokenizer, classnames, templates, device,
                                     cfg["eval"]["template_mode"]).cpu()
        data[name] = {"test": eval_subset(test, cfg["eval"]["max_samples"]), "text": text}
    results = evaluate_variants(model, data, names, cfg, device, args.variants, args.protocols)
    report(results, names)
    out = os.path.join(args.run_dir, "reeval_branch.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"order": names, "max_samples": args.max_samples, "results": results,
                   "pact": {"use_marg": cfg["pact"]["use_marg"], "use_pa": cfg["pact"]["use_pa"]}},
                  f, indent=1, default=float)
    print(f"\nwritten: {out}")
    dist.cleanup()


if __name__ == "__main__":
    main()
