"""Class-Incremental Learning on CIFAR-100 / TinyImageNet.

    python -m pact.run_cil --config configs/cil_cifar100_10steps.yaml
"""

import argparse
import json
import os

import torch
import yaml

from . import distributed as dist
from .config import load_config
from .data import build_cil_tasks, eval_subset
from .engine import Logger, build_clip, evaluate, resolve_device, set_seed, train_task
from .metrics import cil_metrics
from .model import PACTCLIP
from .text_head import build_text_classifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=5e-5")
    parser.add_argument("--resume", action="store_true", help="continue from output_dir/state.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    cfg["setting"] = "cil"
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

    with dist.main_first():
        clip_model, preprocess, tokenizer = build_clip(cfg, device)
    if cfg["cil"]["head"] == "text":
        cfg["head"]["type"] = "text"  # no task id at test: frozen text classifier everywhere
    elif cfg["cil"]["train_label_space"] != "current":
        raise ValueError("cil.head=concat needs cil.train_label_space=current")
    model = PACTCLIP(clip_model, cfg["adapter"], cfg["head"]).to(device)
    with dist.main_first():
        tasks, classnames = build_cil_tasks(cfg["cil"], cfg["data_root"], preprocess,
                                            cfg["download"])
    weights_all = build_text_classifier(clip_model, tokenizer, classnames, [cfg["cil"]["template"]],
                                        device).cpu()
    log({"dataset": cfg["cil"]["dataset"], "tasks": len(tasks),
         "classes_per_task": [len(t["classes"]) for t in tasks]})

    state_path = os.path.join(out, "state.pt")
    step_acc = []
    if args.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        model.load_adapter_state(state["adapters"])
        step_acc = state["step_acc"]
        log({"msg": f"resumed after {len(step_acc)} tasks"})

    for t in range(len(step_acc), len(tasks)):
        task = tasks[t]
        num_seen = len(task["seen"])
        if cfg["cil"]["train_label_space"] == "current":
            train_weights, offset = weights_all[task["classes"]], task["classes"][0]
        else:
            train_weights, offset = weights_all[:num_seen], 0
        train_task(model, task["train"], train_weights, cfg, device, log, f"task{t + 1}", offset)
        test = eval_subset(task["test_seen"], cfg["eval"]["max_samples"])
        if model.use_heads:  # cil.head=concat: learned cond heads of tasks 1..t, in class order
            classifier = torch.cat([model.head_weights(k + 1, "cond").detach().cpu()
                                    for k in range(t + 1)])
        else:
            classifier = weights_all[:num_seen]
        accuracy = evaluate(model, test, classifier, cfg, device)
        step_acc.append(accuracy)
        log({"after_task": t + 1, "seen_classes": num_seen, "acc": round(accuracy, 2)})
        if dist.is_main():
            torch.save({"adapters": model.adapter_state(), "step_acc": step_acc}, state_path)

    if dist.is_main():
        summary = cil_metrics(step_acc)
        print(f"CIL {cfg['cil']['dataset']}: Average={summary['Average']:.2f} "
              f"Last={summary['Last']:.2f}")
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"step_acc": step_acc, "summary": summary}, f, indent=2)
    dist.cleanup()


if __name__ == "__main__":
    main()
