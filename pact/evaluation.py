"""Task-aware MTIL evaluation: task snapshots, a prototype router, several test protocols.

Why (supp. Table 8): after task k the accuracy on dataset k never changes again, and the
proof mentions an otherwise undefined "routing algorithm". So the data of a learned task is
evaluated with that task's own frozen snapshot -- adapters 1..k, gate k, head k -- instead of
the latest model (which forgets, because every new adapter changes the conditional mean).
How the authors pick the snapshot at test time is not described, so several protocols are
computed from the same run (config `eval.protocols`):

    router          per test batch, nearest task prototype on frozen-CLIP features; below the
                    task's acceptance threshold -> frozen CLIP (fallback, as MoE-Adapters' DDAS)
    oracle_zeroshot learned -> own snapshot (task id known); unseen -> frozen CLIP
    oracle_latest   learned -> own snapshot; unseen -> latest model
    latest          everything through the latest model (former behaviour, no task id)

Everything reduces to "snapshot k on dataset j with head h" (k = 0: frozen CLIP). Snapshots
are frozen once their task is over, so their per-batch correct counts are cached and reused
in every later row: one pass per new (dataset, snapshot, head) combination.

Multi-GPU: the test set is cut into fixed global batches, each rank evaluates every
world-th batch, per-batch counts / mean features are summed over ranks (exact, identical on
every rank, also for the router decisions).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import distributed as dist_utils
from .engine import allow_tf32, amp_context

PROTOCOLS = ("router", "oracle_zeroshot", "oracle_latest", "latest")


class MTILEvaluator:
    def __init__(self, model, data, names, cfg, device, log=None):
        """data[name] = {"test": dataset, "text": eval classifier (C, D) on CPU, "train": ...}"""
        self.model, self.data, self.names = model, data, names
        self.cfg, self.device = cfg, device
        self.log = log or (lambda record, echo=True: None)
        self.ecfg = cfg["eval"]
        self.protocols = list(self.ecfg.get("protocols") or ["latest"])
        unknown = set(self.protocols) - set(PROTOCOLS)
        if unknown:
            raise ValueError(f"unknown eval.protocols {sorted(unknown)}; choose from {PROTOCOLS}")
        self.branch = self.ecfg.get("branch", "cond")
        # head used with the learned heads; defaults to the feature branch (cond -> cond head).
        # analysis/reeval_branch.py sets it to evaluate e.g. marg features with the cond head.
        self.head_branch = self.ecfg.get("head_branch") or self.branch
        self.counts = {}       # (name, k, head) -> per-batch correct counts (np.int64)
        self.sizes = {}        # name -> per-batch sample counts
        self.batch_feats = {}  # name -> per-batch mean of frozen-CLIP image features (B, D)
        self.prototypes = []   # task k-1 -> (normalised mean feature (D,), threshold)
        self.router_log = []   # one entry per row

    # ------------------------------------------------------------------ batch passes
    def _batches(self, n):
        bs = self.ecfg["batch_size"]
        return [(b * bs, min((b + 1) * bs, n)) for b in range(math.ceil(n / bs))]

    @torch.no_grad()
    def _pass(self, name, k, head):
        """Per-batch correct counts of snapshot k with head `head` (0 = text) on `name`."""
        ds = self.data[name]["test"]
        batches = self._batches(len(ds))
        mine = list(range(dist_utils.rank(), len(batches), dist_utils.world_size()))
        indices = [i for b in mine for i in range(*batches[b])]
        loader = DataLoader(Subset(ds, indices), batch_size=self.ecfg["batch_size"],
                            shuffle=False, num_workers=self.cfg["num_workers"],
                            pin_memory=self.device.type == "cuda")
        want_feats = k == 0 and name not in self.batch_feats
        counts = torch.zeros(len(batches), dtype=torch.float64)
        sizes = torch.zeros(len(batches), dtype=torch.float64)
        feats = None
        autocast, _ = amp_context(self.device, self.ecfg.get("amp", False))
        allow_tf32()
        weights = (self.model.head_weights(head, self.head_branch) if head > 0
                   else self.data[name]["text"].to(self.device))
        model = self.model
        model.eval()
        with model.snapshot(k):
            for b, (images, labels) in zip(mine, loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast:
                    f = model.encode_image(images, self.branch)
                    logits = f @ weights.to(f.dtype).t()
                counts[b] = (logits.argmax(-1) == labels).sum().item()
                sizes[b] = labels.numel()
                if want_feats:
                    if feats is None:
                        feats = torch.zeros(len(batches), f.shape[-1], dtype=torch.float64)
                    feats[b] = f.float().mean(0).double().cpu()
        counts, sizes = dist_utils.sum_tensor(counts), dist_utils.sum_tensor(sizes)
        self.sizes[name] = sizes.numpy().astype(np.int64)
        if want_feats:
            if feats is None:  # this rank had no batch: still join the all-reduce
                dim = self.data[name]["text"].shape[-1]
                feats = torch.zeros(len(batches), dim, dtype=torch.float64)
            self.batch_feats[name] = dist_utils.sum_tensor(feats).float()
        return counts.numpy().astype(np.int64)

    def counts_for(self, name, k, head=0):
        key = (name, int(k), int(head))
        if key not in self.counts:
            self.counts[key] = self._pass(name, k, head)
        return self.counts[key]

    def accuracy(self, name, k, head=0):
        return 100.0 * self.counts_for(name, k, head).sum() / max(self.sizes[name].sum(), 1)

    def _head(self, j, k):
        """Head for dataset index j (1-based) under snapshot k: its own learned head if the
        snapshot contains task j and heads are learned, else the frozen text classifier."""
        return j if (self.model.use_heads and 1 <= j <= k) else 0

    # ------------------------------------------------------------------ router
    @torch.no_grad()
    def add_prototype(self, train_set, seed=0):
        """Prototype of the task just learned, from frozen-CLIP features of its training data.

        Threshold = `eval.router.quantile` of the cosines of those training features to the
        prototype (self-calibrating per task; ASSUMPTION, the paper describes no router).
        """
        rcfg = self.ecfg.get("router", {})
        n = min(len(train_set), int(rcfg.get("samples", 1000)))
        idx = np.random.RandomState(seed).choice(len(train_set), n, replace=False).tolist()
        idx = [idx[i] for i in dist_utils.shard_indices(len(idx))]
        loader = DataLoader(Subset(train_set, idx), batch_size=self.ecfg["batch_size"],
                            shuffle=False, num_workers=self.cfg["num_workers"])
        autocast, _ = amp_context(self.device, self.ecfg.get("amp", False))
        feats = []
        with self.model.snapshot(0):
            for images, _ in loader:
                with autocast:
                    feats.append(self.model.encode_image(images.to(self.device)).float().cpu())
        dim = self.data[self.names[0]]["text"].shape[-1]
        feats = torch.cat(feats) if feats else torch.zeros(0, dim)
        total = dist_utils.sum_tensor(feats.sum(0).double()).float()
        proto = F.normalize(total, dim=-1)
        # gather all cosines (small: <= samples values) through a padded all-reduce
        cos = feats @ proto
        buf = torch.full((dist_utils.world_size(), n), float("nan"), dtype=torch.float64)
        buf[dist_utils.rank(), :len(cos)] = cos.double()
        buf = torch.nan_to_num(buf, nan=0.0)
        mask = torch.zeros_like(buf)
        mask[dist_utils.rank(), :len(cos)] = 1
        buf, mask = dist_utils.sum_tensor(buf), dist_utils.sum_tensor(mask)
        values = buf[mask > 0].float()
        threshold = float(torch.quantile(values, float(rcfg.get("quantile", 0.05))))
        self.prototypes.append((proto, threshold))
        return threshold

    def route(self, name, t):
        """Router decision per batch of `name` with tasks 1..t learned: k in 0..t (0 = CLIP)."""
        if name not in self.batch_feats:
            self.counts_for(name, 0, 0)
        means = self.batch_feats[name]  # batch mean of unit features: mean cosine = mean . proto
        protos = torch.stack([p for p, _ in self.prototypes[:t]])
        thresholds = torch.tensor([th for _, th in self.prototypes[:t]])
        scores = means @ protos.t()                          # (B, t)
        margin = scores - thresholds
        best = margin.argmax(dim=1)
        accept = margin.gather(1, best[:, None]).squeeze(1) >= 0
        return torch.where(accept, best + 1, torch.zeros_like(best)).numpy()

    def _routed_accuracy(self, name, j, t):
        decisions = self.route(name, t)
        correct = 0
        for k in np.unique(decisions):
            sel = decisions == k
            correct += self.counts_for(name, int(k), self._head(j, int(k)))[sel].sum()
        return 100.0 * correct / max(self.sizes[name].sum(), 1), decisions

    # ------------------------------------------------------------------ rows
    def zero_shot_row(self):
        return [self.accuracy(n, 0, 0) for n in self.names]

    def row(self, t):
        """Accuracies of every protocol after training tasks 1..t (t >= 1)."""
        rows = {p: [] for p in self.protocols}
        routing = {}
        for j, name in enumerate(self.names, start=1):
            learned = j <= t
            own = self.accuracy(name, j, self._head(j, j)) if learned else None
            for p in self.protocols:
                if p == "oracle_zeroshot":
                    acc = own if learned else self.accuracy(name, 0, 0)
                elif p == "oracle_latest":
                    acc = own if learned else self.accuracy(name, t, 0)
                elif p == "latest":
                    acc = self.accuracy(name, t, self._head(j, t))
                else:  # router
                    acc, decisions = self._routed_accuracy(name, j, t)
                    values, freq = np.unique(decisions, return_counts=True)
                    routing[name] = {"to": {(self.names[v - 1] if v else "CLIP"): round(c / len(decisions), 3)
                                            for v, c in zip(values.tolist(), freq.tolist())}}
                    if learned:
                        routing[name]["correct"] = round(float((decisions == j).mean()), 3)
                rows[p].append(float(acc))
        if routing:
            self.router_log.append(routing)
        return rows, routing

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {"counts": {"|".join(map(str, k)): v for k, v in self.counts.items()},
                "sizes": self.sizes, "batch_feats": self.batch_feats,
                "prototypes": self.prototypes, "router_log": self.router_log,
                "batch_size": self.ecfg["batch_size"]}

    def load_state_dict(self, state):
        if not state:
            return
        # prototypes depend only on the training data: always kept
        self.prototypes, self.router_log = state["prototypes"], state["router_log"]
        if state.get("batch_size") != self.ecfg["batch_size"]:
            return  # counts built with other batches: recompute them
        for name, sizes in state["sizes"].items():
            if name not in self.data or int(sizes.sum()) != len(self.data[name]["test"]):
                return  # other test subset (eval.max_samples changed): recompute
        for key, v in state["counts"].items():
            name, k, head = key.rsplit("|", 2)
            self.counts[(name, int(k), int(head))] = v
        self.sizes, self.batch_feats = state["sizes"], state["batch_feats"]
