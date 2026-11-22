"""CLIP with PACT adapters in the image encoder (two-branch ViT, paper Fig. 2)."""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import BranchContext, PACTMLP


def find_resblocks(clip_model):
    """Visual transformer blocks for open_clip (and OpenAI clip) models."""
    visual = clip_model.visual
    transformer = getattr(visual, "transformer", None)
    if transformer is None or not hasattr(transformer, "resblocks"):
        raise ValueError("PACT expects a ViT image encoder with `visual.transformer.resblocks`")
    return transformer.resblocks


class TaskHead(nn.Module):
    """The two lightweight heads of one task (supp.: "only the current adapter and two
    lightweight heads are updated"), one per branch.

    ASSUMPTION (form not given): cosine classifiers over the task's classes, initialised
    from the CLIP text embeddings, so that a new task starts from the zero-shot classifier.
    """

    def __init__(self, init_weights):
        super().__init__()
        init = F.normalize(init_weights.detach().float(), dim=-1)
        self.cond = nn.Parameter(init.clone())
        self.marg = nn.Parameter(init.clone())

    def weights(self, branch):
        return F.normalize(self.marg if branch == "marg" else self.cond, dim=-1)


class PACTCLIP(nn.Module):
    def __init__(self, clip_model, adapter_cfg, head_cfg=None):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.adapter_cfg = adapter_cfg
        self.ctx = BranchContext()
        self.ctx.noise_std = float(adapter_cfg["noise_std"])
        self.num_tasks = 0
        self.use_heads = (head_cfg or {}).get("type", "text") == "learned"
        self.heads = nn.ModuleList()

        self.wrappers = []  # plain list: the wrappers are already registered inside self.clip
        if adapter_cfg["type"] != "none":
            blocks = find_resblocks(clip_model)
            last = max(adapter_cfg["layers"])
            for layer in adapter_cfg["layers"]:  # 1-indexed, as in the paper
                block = blocks[layer - 1]
                wrapper = PACTMLP(block.mlp, self.ctx, adapter_cfg, depth=layer / last)
                block.mlp = wrapper
                self.wrappers.append(wrapper)

    # ----- tasks -----------------------------------------------------------
    def add_task(self, head_init=None):
        """Insert (or, for a shared adapter, keep) the adapter and the heads of the next task."""
        self.num_tasks += 1
        if not self.wrappers:
            return
        if self.adapter_cfg["per_task"] or not self.wrappers[0].adapters:
            for w in self.wrappers:
                w.add_adapter()
        if self.use_heads:
            if head_init is None:
                raise ValueError("head.type=learned needs the text embeddings of the task")
            device = next(self.clip.parameters()).device
            self.heads.append(TaskHead(head_init).to(device))
        self.freeze_old()

    def freeze_old(self):
        for w in self.wrappers:
            w.freeze_all_but_last()
        for i, head in enumerate(self.heads):
            for p in head.parameters():
                p.requires_grad_(i == len(self.heads) - 1)

    @property
    def num_adapters(self):
        return len(self.wrappers[0].adapters) if self.wrappers else 0

    def adapter_parameters(self):
        return [p for w in self.wrappers for a in w.adapters
                for p in w.adapter_params(a) if p.requires_grad]

    def gate_parameters(self):
        return [g for w in self.wrappers for g in w.gates if g.requires_grad]

    def head_parameters(self):
        return [p for h in self.heads for p in h.parameters() if p.requires_grad]

    def head_weights(self, task, branch="cond"):
        """Classifier of task `task` (1-based; -1 = current) for a branch."""
        return self.heads[task - 1 if task > 0 else task].weights(branch)

    @contextmanager
    def snapshot(self, k):
        """Model as it was right after task k: adapters 1..k, gate k (k = 0: frozen CLIP).

        Old adapters / gates / heads are frozen after their task, so snapshot k is exactly
        the model that was evaluated at the end of task k (supp. Table 8 invariance).
        """
        previous = self.ctx.active
        self.ctx.active = k
        try:
            yield
        finally:
            self.ctx.active = previous

    # ----- forward ---------------------------------------------------------
    def encode_image(self, images, branch="cond"):
        if self.ctx.active == 0:
            branch = "base"
        self.ctx.branch = branch
        try:
            feats = self.clip.encode_image(images)
        finally:
            self.ctx.branch = "base"
        return F.normalize(feats, dim=-1)

    def logits(self, images, text_weights, branch="cond"):
        feats = self.encode_image(images, branch)
        scale = self.clip.logit_scale.exp()
        return scale * feats @ text_weights.to(feats.dtype).t()

    # ----- checkpoints -----------------------------------------------------
    def adapter_state(self):
        """Only what PACT learns (adapters, gates, heads, anchors); CLIP itself is re-loaded."""
        return {
            "num_tasks": self.num_tasks,
            "heads": [{k: v.detach().cpu() for k, v in h.state_dict().items()} for h in self.heads],
            "layers": [
                {
                    "adapters": [{k: v.detach().cpu() for k, v in a.state_dict().items()}
                                 for a in w.adapters],
                    "gates": [g.detach().cpu() for g in w.gates],
                    "anchors": [[t.cpu() for t in anchor] for anchor in w.anchors],
                }
                for w in self.wrappers
            ],
        }

    def load_adapter_state(self, state):
        n_adapters = len(state["layers"][0]["adapters"]) if state["layers"] else 0
        heads = state.get("heads", [])
        if self.use_heads and len(heads) < n_adapters:
            raise ValueError("checkpoint has no learned heads but head.type=learned")
        while self.num_adapters < n_adapters:
            k = self.num_adapters
            self.add_task(heads[k]["cond"] if self.use_heads else None)
        self.num_tasks = state["num_tasks"]
        for head, sd in zip(self.heads, heads):
            head.load_state_dict(sd)
        for w, layer_state in zip(self.wrappers, state["layers"]):
            device = next(w.base.parameters()).device
            for adapter, sd in zip(w.adapters, layer_state["adapters"]):
                adapter.load_state_dict(sd)
            for gate, value in zip(w.gates, layer_state["gates"]):
                gate.data.copy_(value.to(device))
            w.anchors = [[t.to(device) for t in anchor] for anchor in layer_state["anchors"]]
        self.freeze_old()
