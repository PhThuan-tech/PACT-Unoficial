"""Training of one task with the PACT objective (Eq. 15) and evaluation."""

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from . import distributed as dist_utils
from .losses import prior_anchoring_loss
from .phase import PostConvergenceSchedule, ProbeBank, compute_alpha


# ----------------------------------------------------------------------------- setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def build_clip(cfg, device):
    import open_clip

    arch, pretrained = cfg["model"]["arch"], cfg["model"]["pretrained"]
    openai_like = pretrained == "openai" or os.path.isfile(str(pretrained))
    if openai_like and "quickgelu" not in arch.lower():
        # OpenAI CLIP weights were trained with QuickGELU. open_clip >= 2.2x builds "ViT-B-16"
        # with plain GELU even for pretrained="openai" (it only warns): silent accuracy drop.
        print(f"[warn] arch={arch} uses GELU; OpenAI CLIP weights need {arch}-quickgelu",
              flush=True)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(cfg["model"]["arch"])
    return clip_model.to(device).eval(), preprocess, tokenizer


def allow_tf32():
    """fp32 evaluation as upstream, with TF32 matmuls (Ampere+, negligible effect on argmax)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def amp_context(device, enabled):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False), None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype), dtype


class Logger:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "log.jsonl")

    def __call__(self, record, echo=True):
        if not dist_utils.is_main():  # multi-GPU: only rank 0 writes
            return
        record = {"time": round(time.time(), 1), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if echo:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in record.items() if k != "time"), flush=True)


# ----------------------------------------------------------------------------- training
def _infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)  # new shuffle every epoch, identical across ranks
        for batch in loader:
            yield batch
        epoch += 1


def _train_loader(train_set, tr, cfg, device):
    """Per-rank loader; train.batch_size is the global batch (split across GPUs)."""
    world = dist_utils.world_size()
    common = dict(num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                  persistent_workers=cfg["num_workers"] > 0, drop_last=True)
    if world == 1:
        batch_size = min(tr["batch_size"], len(train_set))
        return DataLoader(train_set, batch_size=batch_size, shuffle=True, **common), None
    if tr["batch_size"] % world:
        raise ValueError(f"train.batch_size={tr['batch_size']} must be divisible by the "
                         f"number of GPUs ({world})")
    sampler = DistributedSampler(train_set, num_replicas=world, rank=dist_utils.rank(),
                                 shuffle=True, seed=cfg["seed"], drop_last=True)
    if len(sampler) == 0:
        raise ValueError(f"training set of {len(train_set)} samples is smaller than the "
                         f"number of GPUs ({world})")
    batch_size = min(tr["batch_size"] // world, len(sampler))
    return DataLoader(train_set, batch_size=batch_size, sampler=sampler, **common), sampler


def _cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_task(model, train_set, text_weights, cfg, device, log, task_name, label_offset=0,
               head_init=None):
    """Adds the task's adapter (+ heads) and trains with L_cond + alpha_tilde (L_marg + L_pa).

    text_weights: frozen text classifier used when there are no learned heads;
    head_init:    initial weights of the task's two learned heads (default: text_weights).
    """
    tr, pact = cfg["train"], cfg["pact"]
    model.add_task(head_init if head_init is not None else text_weights)
    adapter_params, gate_params = model.adapter_parameters(), model.gate_parameters()
    head_params = model.head_parameters()
    if not adapter_params:
        log({"task": task_name, "msg": "no trainable adapter (adapter.type=none), skipping"})
        return

    trainable = adapter_params + gate_params + head_params
    dist_utils.broadcast_params(trainable)  # identical start on every GPU

    groups = [{"params": adapter_params, "lr": tr["lr"]}]
    if gate_params:
        groups.append({"params": gate_params, "lr": tr["lr"] * tr["gate_lr_mult"]})
    if head_params:
        groups.append({"params": head_params,
                       "lr": tr["lr"] * cfg.get("head", {}).get("lr_mult", 1.0)})
    optimizer = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    base_lrs = [g["lr"] for g in groups]

    loader, sampler = _train_loader(train_set, tr, cfg, device)
    batches = _infinite(loader, sampler)

    dim = next(m for m in model.wrappers[0].base.modules()
               if isinstance(m, torch.nn.Linear)).in_features
    probes = ProbeBank(dim, pact["num_probes"], pact["probe_seed"], device)
    schedule_cfg = dict(pact["schedule"])
    if not pact["post_convergence"]:
        schedule_cfg["mode"] = "none"
    schedule = PostConvergenceSchedule(schedule_cfg, tr["iterations"])
    autocast, amp_dtype = amp_context(device, tr["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    text_weights = text_weights.to(device)
    regularise = pact["use_marg"] or pact["use_pa"]

    def classifier(branch):
        return model.head_weights(-1, branch) if model.use_heads else text_weights

    alpha, phase_info = 1.0, {}
    model.train()
    for it in range(tr["iterations"]):
        factor = _cosine_lr(it, tr["iterations"], tr["warmup"])
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * factor

        stats = {}
        for micro in range(tr["grad_accum"]):
            images, labels = next(batches)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) - label_offset
            with autocast:
                loss_cond = F.cross_entropy(model.logits(images, classifier("cond"), "cond"), labels,
                                            label_smoothing=tr["label_smoothing"])
            if micro == 0:
                # averaged over GPUs so that every rank takes the same schedule decision
                s = schedule.update(it, dist_utils.mean_scalar(loss_cond.item(), device))
                if regularise and schedule.converged_at == it and schedule.reason:
                    log({"task": task_name, "pc_start": it, "reason": schedule.reason})
                # alpha_t depends only on the adapter weights, which change once per
                # optimizer step: compute it at most once per iteration.
                if regularise and s > 0:
                    if not pact["use_phase"]:
                        alpha = 1.0
                    elif it % pact["alpha_every"] == 0 or not phase_info:
                        alpha, phase_info = compute_alpha(model, probes, pact["tau"],
                                                          pact["first_task_reference"])
            loss = loss_cond
            if regularise and s > 0:
                alpha_tilde = s * alpha
                reg = torch.zeros((), device=device)
                if pact["use_marg"]:
                    with autocast:
                        loss_marg = F.cross_entropy(model.logits(images, classifier("marg"), "marg"),
                                                    labels, label_smoothing=tr["label_smoothing"])
                    reg = reg + loss_marg
                    stats["L_marg"] = loss_marg.item()
                if pact["use_pa"]:
                    loss_pa = prior_anchoring_loss(model, pact["pa_sigma_rel"])
                    reg = reg + loss_pa
                    stats["L_pa"] = loss_pa.item()
                loss = loss_cond + alpha_tilde * reg
                stats["alpha_tilde"] = alpha_tilde
            stats["L_cond"] = loss_cond.item()
            scaler.scale(loss / tr["grad_accum"]).backward()
        dist_utils.all_reduce_grads(trainable)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % tr["log_every"] == 0 or it == tr["iterations"] - 1:
            record = {"task": task_name, "it": it, "s": s, "alpha": alpha,
                      "lr": optimizer.param_groups[0]["lr"], **stats}
            if schedule.phi is not None:
                record["phi"] = schedule.phi
            if phase_info.get("nearest") is not None:
                record["nearest"] = phase_info["nearest"]
                record["d_min"] = min(phase_info["distances"])
            if phase_info.get("d_mlp") is not None:
                record["d_mlp"] = phase_info["d_mlp"]
            if gate_params:
                record["gate"] = float(torch.sigmoid(torch.stack(gate_params).detach()).mean())
            log(record)
    model.eval()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, dataset, text_weights, cfg, device, branch=None):
    """Top-1 accuracy (%) of the chosen branch ("base" = untouched CLIP).

    Multi-GPU: each rank evaluates a disjoint shard, counts are summed (exact accuracy).
    """
    branch = branch or cfg["eval"]["branch"]
    if dist_utils.world_size() > 1:
        dataset = Subset(dataset, dist_utils.shard_indices(len(dataset)))
    loader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    autocast, _ = amp_context(device, cfg["eval"].get("amp", cfg["train"]["amp"]))
    allow_tf32()
    text_weights = text_weights.to(device)
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast:
            predictions = model.logits(images, text_weights, branch).argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    correct, total = dist_utils.sum_counts(correct, total, device)
    return 100.0 * correct / max(total, 1)
