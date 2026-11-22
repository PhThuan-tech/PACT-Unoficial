"""Phase-like transition weight (paper Sec. 3.5, Eq. 11-15) and the post-convergence schedule."""

import math
from collections import deque

import torch


class ProbeBank:
    """Fixed Gaussian probes z_m ~ N(0, I_d), shared across steps and layers (Eq. 11)."""

    def __init__(self, dim, num_probes, seed, device):
        gen = torch.Generator().manual_seed(seed)
        self.z = torch.randn(num_probes, dim, generator=gen).to(device)


def _layer_dim(wrapper):
    linear = next(m for m in wrapper.base.modules() if isinstance(m, torch.nn.Linear))
    return linear.in_features


@torch.no_grad()
def compute_alpha(model, probes, tau, first_task_reference="pretrained"):
    """alpha_t = exp(-min_j d_j / tau), d_j = ||H_t - H_j||_F^2 / (eps + ||H_t||^2 + ||H_j||^2).

    Responses of all adapter layers are concatenated, so d_j compares the whole task-t
    adapter with the whole task-j adapter. Task 1 has no historical adapter; by default it
    is compared with the frozen pretrained MLP (ASSUMPTION), otherwise alpha = 0 (free).

    Returns (alpha, info) where info holds every d_j, the nearest index j* and d_mlp, the
    same distance between the current adapter and the frozen pretrained MLP (its drift,
    logged to calibrate tau and pa_sigma_rel).
    """
    wrappers = model.wrappers
    num = len(wrappers[0].adapters)
    if num == 0:
        return 0.0, {"distances": [], "nearest": None}
    z = probes.z
    current = [w.adapters[-1](z) for w in wrappers]
    pretrained = [w.base(z) for w in wrappers]

    eps = 1e-8
    cur_sq = sum(h.float().pow(2).sum() for h in current)

    def distance(ref):
        diff = sum((h - r).float().pow(2).sum() for h, r in zip(current, ref))
        ref_sq = sum(r.float().pow(2).sum() for r in ref)
        return (diff / (eps + cur_sq + ref_sq)).item()

    d_mlp = distance(pretrained)
    if num > 1:
        distances = [distance([w.adapters[j](z) for w in wrappers]) for j in range(num - 1)]
    elif first_task_reference == "pretrained":
        distances = [d_mlp]
    else:
        return 0.0, {"distances": [], "nearest": None, "d_mlp": d_mlp}

    nearest = min(range(len(distances)), key=distances.__getitem__)
    alpha = math.exp(-distances[nearest] / tau)
    return alpha, {"distances": distances, "nearest": nearest, "d_mlp": d_mlp}


class PostConvergenceSchedule:
    """s_k in [0, 1] such that alpha_tilde = s_k * alpha_t (Eq. 15).

    mode "window" (supp. Sec. 6.3, Eq. 40-44): over the last W losses
        mean  l = 1/W sum l_k,  var v = 1/(W-1) sum (l_k - l)^2,  slope m = 1/W sum (l_k - l_{k-1})
        phi   = w1 sqrt(v) / (l + eps) + w2 |m|,       switch = 1[phi <= gamma]
      ASSUMPTIONS (no value / rule is given): W, w1, w2, gamma; the loss sequence is L_cond
      (identical to L_total before the switch); the switch latches once on; it is never on in
      the first `warmup_frac` of training and forced on at `latest_frac` (safety net);
      alpha_tilde = switch * alpha_t (hard, `ramp_iters` = 0) as Fig. 6 shows a jump.
    mode "stability": former assumption, relative change of an EMA of L_cond < `rel_tol`.
    mode "linear":    s ramps 0 -> 1 between fractions `linear_start` and `linear_end`.
    mode "none":      s = 1 from the first iteration (ablation without "PC").
    `window` / `ramp_iters` are capped at 10% / 20% of the run so short runs behave like
    long ones. `started_at` / `reason` record when and why the constraint switched on, `phi`
    the last stability indicator (logged to calibrate gamma).
    """

    def __init__(self, cfg, total_iters):
        self.cfg = cfg
        self.total = total_iters
        self.window = max(2, min(int(cfg["window"]), int(0.1 * total_iters)))
        self.ramp = max(1, min(int(cfg.get("ramp_iters", 0)), int(0.2 * total_iters)))
        self.ema = None
        self.history = deque(maxlen=self.window + 1)
        self.converged_at = None
        self.reason = None
        self.phi = None
        self.on = False

    @property
    def started_at(self):
        return self.converged_at

    def _phi(self):
        losses = list(self.history)  # W + 1 values: l_{i-W} .. l_i
        window = losses[1:]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
        slope = (losses[-1] - losses[0]) / len(window)  # telescoped (1/W) sum (l_k - l_{k-1})
        w1, w2 = float(self.cfg.get("w1", 0.5)), float(self.cfg.get("w2", 0.5))
        return w1 * math.sqrt(var) / (mean + 1e-8) + w2 * abs(slope)

    def _ramp(self, it):
        if self.converged_at is None or not self.on:
            return 0.0
        return float(min((it - self.converged_at + 1) / self.ramp, 1.0))

    def update(self, it, loss_value):
        mode = self.cfg["mode"]
        if mode == "none":
            return 1.0
        if mode == "linear":
            start = self.cfg["linear_start"] * self.total
            end = max(self.cfg["linear_end"] * self.total, start + 1)
            return float(min(max((it - start) / (end - start), 0.0), 1.0))
        past_warmup = it >= self.cfg["warmup_frac"] * self.total
        deadline = it >= self.cfg.get("latest_frac", 1.0) * self.total

        if mode == "window":
            self.history.append(float(loss_value))
            if len(self.history) > self.window:
                self.phi = self._phi()
            stable = self.phi is not None and self.phi <= float(self.cfg["gamma"])
            latch = self.cfg.get("latch", True)
            if self.converged_at is None:
                if past_warmup and stable:
                    self.converged_at, self.reason = it, "converged"
                elif deadline:
                    self.converged_at, self.reason = it, "deadline"
                self.on = self.converged_at is not None
            elif not latch and self.reason == "converged":
                self.on = stable or deadline
            return self._ramp(it)

        if mode != "stability":
            raise ValueError(f"unknown schedule mode {mode}")
        beta = self.cfg["ema"]
        self.ema = loss_value if self.ema is None else beta * self.ema + (1 - beta) * loss_value
        self.history.append(self.ema)
        if self.converged_at is None and past_warmup:
            if len(self.history) > self.window:
                past = self.history[0]
                if abs(self.ema - past) / (abs(past) + 1e-8) < self.cfg["rel_tol"]:
                    self.converged_at, self.reason = it, "converged"
            if self.converged_at is None and deadline:
                self.converged_at, self.reason = it, "deadline"
        self.on = self.converged_at is not None
        return self._ramp(it)
