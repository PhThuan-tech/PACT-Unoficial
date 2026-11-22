"""Multi-domain Task-Incremental Learning (MTIL), full-shot or few-shot.

    python -m pact.run_mtil --config configs/mtil_order1.yaml
    python -m pact.run_mtil --config configs/mtil_order1.yaml --set train.batch_size=16 train.grad_accum=4
    torchrun --standalone --nproc_per_node=2 -m pact.run_mtil --config configs/mtil_order1.yaml
"""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from . import distributed as dist
from .config import load_config
from .data import build_mtil_dataset, eval_subset, few_shot_subset
from .engine import Logger, allow_tf32, amp_context, build_clip, resolve_device, set_seed, train_task
from .evaluation import MTILEvaluator
from .metrics import format_mtil_table, mtil_metrics
from .model import PACTCLIP
from .text_head import build_text_classifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=5e-5")
    parser.add_argument("--resume", action="store_true", help="continue from output_dir/state.pt")
    return parser.parse_args()


@torch.no_grad()
def zero_shot_accuracies(model, dataset, classifiers, cfg, device):
    """Frozen-CLIP accuracy of one image pass against several text classifiers."""
    if dist.world_size() > 1:
        dataset = Subset(dataset, dist.shard_indices(len(dataset)))
    loader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    autocast, _ = amp_context(device, cfg["eval"]["amp"])
    allow_tf32()
    weights = {k: w.to(device) for k, w in classifiers.items()}
    correct = {k: 0 for k in weights}
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast:
            f = model.encode_image(images, "base")
        for k, w in weights.items():
            correct[k] += (f @ w.to(f.dtype).t()).argmax(-1).eq(labels).sum().item()
        total += labels.numel()
    out = {}
    for k in weights:
        c, n = dist.sum_counts(correct[k], total, device)
        out[k] = 100.0 * c / max(n, 1)
    return out


def summarise(acc, names):
    rows, summary = mtil_metrics(acc, names)
    return {"acc_matrix": acc, "per_dataset": rows, "summary": summary}


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    device = dist.init(cfg["device"]) or resolve_device(cfg["device"])
    set_seed(cfg["seed"])
    out = cfg["output_dir"]
    if dist.is_main():
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({**cfg, "world_size": dist.world_size()}, f, sort_keys=False,
                           allow_unicode=True)
    dist.barrier()
    log = Logger(out)
    log({"msg": "start", "world_size": dist.world_size(), "device": str(device)})

    with dist.main_first():  # with download=true, rank 0 downloads before the others read
        clip_model, preprocess, tokenizer = build_clip(cfg, device)
    model = PACTCLIP(clip_model, cfg["adapter"], cfg["head"]).to(device)
    names = cfg["mtil"]["order"]

    data = {}
    for name in names:
        with dist.main_first():
            train, test, classnames, templates = build_mtil_dataset(
                name, cfg["data_root"], preprocess, cfg["download"])
        if cfg["mtil"]["shots"]:
            train = few_shot_subset(train, cfg["mtil"]["shots"], cfg["mtil"]["fewshot_seed"])
        test = eval_subset(test, cfg["eval"]["max_samples"])
        classifiers = {mode: build_text_classifier(clip_model, tokenizer, classnames, templates,
                                                   device, mode).cpu()
                       for mode in ("first", "ensemble")}
        data[name] = {"train": train, "test": test, "classifiers": classifiers,
                      "text": classifiers[cfg["eval"]["template_mode"]],          # evaluation
                      "train_text": classifiers[cfg["train"]["template_mode"]]}  # training
        log({"dataset": name, "train": len(train), "test": len(test), "classes": len(classnames)})

    if cfg["adapter"]["type"] == "none":  # zero-shot only: one evaluation, no training loop
        by_mode = {"first": [], "ensemble": []}
        for n in names:
            accs = zero_shot_accuracies(model, data[n]["test"], data[n]["classifiers"], cfg, device)
            for mode, a in accs.items():
                by_mode[mode].append(a)
        zero_shot = by_mode[cfg["eval"]["template_mode"]]
        log({"zero_shot": dict(zip(names, [round(a, 2) for a in zero_shot])),
             "template_mode": cfg["eval"]["template_mode"]})
        log({"zero_shot_by_template": {m: dict(zip(names, [round(a, 2) for a in v]))
                                       for m, v in by_mode.items()}})
        if dist.is_main():
            with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
                json.dump({"order": names, "zero_shot": zero_shot,
                           "zero_shot_by_template": by_mode,
                           "summary": {"zero_shot_mean": sum(zero_shot) / len(zero_shot)}},
                          f, indent=2)
        dist.cleanup()
        return

    evaluator = MTILEvaluator(model, data, names, cfg, device, log)
    protocols = evaluator.protocols
    head_key = "text" if cfg["head"]["init"] == "eval" else "train_text"
    state_path = os.path.join(out, "state.pt")
    acc, zero_shot = {p: [] for p in protocols}, None
    if args.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(state.get("acc"), dict):
            raise SystemExit(f"{state_path} was written by an older version of the code "
                             "(no task snapshots / heads): use a new output_dir")
        model.load_adapter_state(state["adapters"])
        evaluator.load_state_dict(state.get("evaluator"))
        acc = {p: state["acc"].get(p, []) for p in protocols}
        zero_shot = state.get("zero_shot")
        done = len(acc[protocols[0]])
        if any(len(v) != done for v in acc.values()):
            raise SystemExit("eval.protocols changed since the checkpoint: keep the same list")
        log({"msg": f"resumed after {done} tasks"})
    elif cfg["eval"]["zero_shot_row"]:
        zero_shot = evaluator.zero_shot_row()
        log({"zero_shot": dict(zip(names, [round(a, 2) for a in zero_shot]))})

    for t in range(len(acc[protocols[0]]), len(names)):
        name = names[t]
        train_task(model, data[name]["train"], data[name]["train_text"], cfg, device, log, name,
                   head_init=data[name][head_key])
        if "router" in protocols:
            threshold = evaluator.add_prototype(data[name]["train"], seed=cfg["seed"] + t)
            log({"task": name, "router_threshold": round(threshold, 4)})
        rows, routing = evaluator.row(t + 1)
        for p in protocols:
            acc[p].append(rows[p])
        log({"after_task": name, "protocol": protocols[0],
             "acc": dict(zip(names, [round(a, 2) for a in rows[protocols[0]]])),
             "mean": {p: round(sum(r) / len(r), 2) for p, r in rows.items()}})
        if routing:
            log({"after_task": name, "routing": routing}, echo=False)
        if dist.is_main():
            torch.save({"adapters": model.adapter_state(), "acc": acc, "zero_shot": zero_shot,
                        "order": names, "evaluator": evaluator.state_dict()}, state_path)

    if dist.is_main():
        results = {"order": names, "main_protocol": protocols[0], "zero_shot": zero_shot,
                   **summarise(acc[protocols[0]], names),
                   "protocols": {p: summarise(acc[p], names) for p in protocols},
                   "router": evaluator.router_log}
        for p in protocols:
            print(f"--- protocol: {p}")
            print(format_mtil_table(results["protocols"][p]["per_dataset"],
                                    results["protocols"][p]["summary"], names))
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
    dist.cleanup()


if __name__ == "__main__":
    main()
