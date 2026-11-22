"""MTIL (Transfer / Avg / Last) and CIL (Average / Last) metrics."""

import numpy as np


def mtil_metrics(acc, names):
    """acc[i][j]: accuracy (%) on dataset j after training task i (both in training order).

    Per dataset j, following ZSCL / MoE-Adapters tables:
      Transfer_j = mean_{i<j} acc[i][j]   (zero-shot on a not-yet-learned dataset; j >= 1)
      Avg_j      = mean_i     acc[i][j]   (over all T steps)
      Last_j     = acc[T-1][j]
    Overall Transfer averages datasets 2..T; Avg and Last average all datasets.
    """
    acc = np.asarray(acc, dtype=float)
    steps, num = acc.shape
    rows = {"Transfer": {}, "Avg": {}, "Last": {}}
    for j, name in enumerate(names):
        if j > 0 and j <= steps:
            rows["Transfer"][name] = acc[:j, j].mean()
        rows["Avg"][name] = acc[:, j].mean()
        rows["Last"][name] = acc[-1, j]
    summary = {key: float(np.mean(list(vals.values()))) if vals else float("nan")
               for key, vals in rows.items()}
    return rows, summary


def cil_metrics(step_accuracies):
    accs = np.asarray(step_accuracies, dtype=float)
    return {"Average": float(accs.mean()), "Last": float(accs[-1])}


def format_mtil_table(rows, summary, names):
    short = [n[:9] for n in names]
    header = f"{'':10s}" + "".join(f"{n:>10s}" for n in short) + f"{'Mean':>10s}"
    lines = [header]
    for key in ("Transfer", "Avg", "Last"):
        cells = "".join(f"{rows[key][n]:10.2f}" if n in rows[key] else f"{'-':>10s}"
                        for n in names)
        lines.append(f"{key:10s}{cells}{summary[key]:10.2f}")
    return "\n".join(lines)
