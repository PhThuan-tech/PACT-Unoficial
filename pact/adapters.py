"""Task adapters placed in parallel with the frozen ViT MLP blocks (paper Sec. 3.3-3.4).

`PACTMLP` replaces `resblock.mlp`. Because the residual block calls `mlp(ln_2(x))`, the
wrapper sees exactly the MLP input, so the adapters are connected in parallel with the
MLP (SAI (1)) regardless of the CLIP implementation.
"""

import copy
import math

import torch
import torch.nn as nn


class BranchContext:
    """Shared switch read by every PACTMLP during a forward pass."""

    def __init__(self):
        self.branch = "base"  # "base" | "cond" | "marg"
        self.noise_std = 0.0
        self.active = None    # snapshot k: use adapters 1..k and gate k; None = all (latest)


class MLPAdapter(nn.Module):
    """Adapter with the same architecture as the parallel MLP (SAI (2)).

    clone=True copies the MLP parameters (SAI (3)); otherwise the linear layers are
    re-initialised randomly (ablation "SAI (1)(2)").
    """

    def __init__(self, base_mlp, clone=True):
        super().__init__()
        self.net = copy.deepcopy(base_mlp)
        if not clone:
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        for p in self.net.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        return self.net(x)


class LoRAAdapter(nn.Module):
    """Low-rank residual on top of the frozen MLP: h = mlp(x) + B A x (ablation baselines)."""

    def __init__(self, base_mlp, rank):
        super().__init__()
        dim = next(m for m in base_mlp.modules() if isinstance(m, nn.Linear)).in_features
        object.__setattr__(self, "base", base_mlp)  # frozen MLP, referenced but not registered
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))

    def own_parameters(self):
        return list(self.down.parameters()) + list(self.up.parameters())


class PACTMLP(nn.Module):
    """Frozen MLP + one adapter per task + one gate per task.

    Branches (Eq. 4, 5, 8), with t = number of active adapters (all, or snapshot k):
        base: h = mlp(x)
        cond: h_ad = mean_{i<=t} adapter_i(x) + 0.1 eps   (old adapters frozen)
        marg: h_ad = adapter_t(x) + 0.1 eps               (current adapter only)
        out  = (1 - g_t) h_base + g_t h_ad              (with gate)
        out  = h_base + h_ad                            (MLP adapter without gate: plain parallel
                                                        sum, the "uncontrolled branch" ablation)
        out  = h_ad                                     (LoRA: h_ad already contains the MLP)
    """

    def __init__(self, base_mlp, ctx, cfg, depth=1.0):
        super().__init__()
        self.base = base_mlp
        self.ctx = ctx
        self.cfg = cfg
        self.depth = depth  # layer / last adapter layer, for the layer-wise gate schedule
        self.adapters = nn.ModuleList()
        self.gates = nn.ParameterList()
        self.anchors = []  # initial adapter parameters per task, for prior anchoring

    # ----- task management -------------------------------------------------
    def add_adapter(self):
        if self.cfg["type"] == "lora":
            adapter = LoRAAdapter(self.base, self.cfg["lora_rank"])
        else:
            adapter = MLPAdapter(self.base, clone=self.cfg["clone_mlp"])
        device = next(self.base.parameters()).device
        adapter.to(device)
        gamma = self._initial_gamma(num_after=len(self.adapters) + 1)
        self.adapters.append(adapter)
        self.gates.append(nn.Parameter(torch.tensor(gamma, device=device)))
        self.anchors.append([p.detach().clone() for p in self.adapter_params(adapter)])

    def _initial_gamma(self, num_after):
        """Initial gate logit gamma_t of the adapter being inserted (SAI (4)).

        Task 1, or adapters that are not MLP clones: gamma = `gate_init`. Task 1 is then
        exactly prediction-preserving because the clone equals the MLP for any g.

        Task t >= 2 with a cloned adapter (`gate_init_mode: preserve`): before insertion the
        conditional branch outputs (1 - g) h + g m, with m the mean of the t-1 old adapters
        and g = sigmoid(gamma_{t-1}). After insertion the new adapter equals h, so with gate
        g' the output is (1 - g' (t-1)/t) h + g' (t-1)/t m. It is unchanged iff
        g' = g t / (t - 1). If that exceeds `gate_max_init`, g' is clipped (sigmoid would
        saturate) and the insertion is only approximately prediction-preserving.
        """
        cfg = self.cfg
        mode = cfg.get("gate_mode", "learned")
        if mode in ("layerwise", "constant"):
            if mode == "layerwise":
                g = float(cfg["gate_last"]) * self.depth ** float(cfg.get("gate_power", 1.0))
            else:
                g = float(cfg["gate_value"])
            g = min(max(g, 1e-6), 1 - 1e-6)
            return math.log(g / (1.0 - g))
        fixed = float(cfg["gate_init"])
        if (len(self.gates) == 0 or not cfg["use_gate"] or not cfg["clone_mlp"]
                or cfg["type"] != "mlp" or cfg.get("gate_init_mode", "preserve") != "preserve"):
            return fixed
        g_prev = torch.sigmoid(self.gates[-1].detach().float()).item()
        g_new = min(g_prev * num_after / (num_after - 1), float(cfg["gate_max_init"]))
        g_new = max(g_new, 1e-4)
        return math.log(g_new / (1.0 - g_new))

    @staticmethod
    def adapter_params(adapter):
        if isinstance(adapter, LoRAAdapter):
            return adapter.own_parameters()
        return list(adapter.parameters())

    def freeze_all_but_last(self):
        for i, (adapter, gate) in enumerate(zip(self.adapters, self.gates)):
            trainable = i == len(self.adapters) - 1
            for p in self.adapter_params(adapter):
                p.requires_grad_(trainable)
            gate.requires_grad_(trainable and self.cfg["use_gate"] and self.gate_learned)

    @property
    def gate_learned(self):
        return self.cfg.get("gate_mode", "learned") == "learned"

    def num_active(self):
        n = len(self.adapters)
        if self.ctx.active is not None:
            n = min(self.ctx.active, n)  # shared adapter (per_task=False): only one exists
        return n

    # ----- forward ---------------------------------------------------------
    def forward(self, x):
        h_base = self.base(x)
        branch = self.ctx.branch
        n = self.num_active()
        if branch == "base" or n == 0:
            return h_base
        if branch == "marg":
            h_ad = self.adapters[n - 1](x)
        elif branch == "cond":
            h_ad = torch.stack([a(x) for a in self.adapters[:n]], dim=0).mean(dim=0)
        else:
            raise ValueError(f"unknown branch {branch}")
        if self.training and self.ctx.noise_std > 0:
            h_ad = h_ad + self.ctx.noise_std * torch.randn_like(h_ad)
        if self.cfg["type"] == "lora":
            return h_ad
        if not self.cfg["use_gate"]:
            return h_base + h_ad
        g = torch.sigmoid(self.gates[n - 1]).to(h_ad.dtype)
        return (1 - g) * h_base + g * h_ad
