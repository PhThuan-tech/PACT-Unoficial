"""Test of the core of analysis/oracle_snapshot.py with the tiny fake CLIP (CPU, no downloads).

    python tests/oracle_snapshot_test.py

Trains 3 fake tasks with the real train_task / MTILEvaluator, then checks that
  - snapshot_table has the right shape (dataset j: snapshots 0..j) and column 0 is the frozen-CLIP accuracy,
  - it agrees with the run's own `oracle_latest` protocol on every unseen entry (same numbers, same definition),
  - Transfer of "latest" equals pact.metrics' Transfer of the oracle_latest matrix,
  - oracle_best >= max(zero_shot, ...) logic: never below zero-shot, never above the best snapshot per row.
"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from pact.config import load_config
from pact.engine import train_task
from pact.evaluation import MTILEvaluator
from pact.metrics import mtil_metrics
from pact.model import PACTCLIP
import oracle_snapshot as osnap

D = 16
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln_2 = nn.LayerNorm(D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x): return x + s.mlp(s.ln_2(x))
class Visual(nn.Module):
    def __init__(s):
        super().__init__(); s.proj = nn.Linear(12, D)
        s.transformer = nn.Module(); s.transformer.resblocks = nn.ModuleList([Block() for _ in range(12)])
    def forward(s, x):
        h = s.proj(x)
        for b in s.transformer.resblocks: h = b(h)
        return h
class FakeCLIP(nn.Module):
    def __init__(s):
        super().__init__(); s.visual = Visual(); s.logit_scale = nn.Parameter(torch.tensor(2.0))
    def encode_image(s, x): return s.visual(x)

def make_task(t, n=96, ncls=4):
    g = torch.Generator().manual_seed(t)
    x = 0.3 * torch.randn(n, 12, generator=g)
    x[:, t] += 4.0
    y = torch.randint(0, ncls, (n,), generator=g)
    x[torch.arange(n), 4 + y] += 1.0
    return TensorDataset(x, y), nn.functional.normalize(torch.randn(ncls, D, generator=g), dim=-1)

torch.manual_seed(0)
cfg = load_config(None, ["device=cpu", "num_workers=0", "train.iterations=20", "train.batch_size=16",
                         "train.warmup=2", "train.log_every=100", "train.lr=1e-3",
                         "eval.batch_size=16", "eval.router.samples=64"])
model = PACTCLIP(FakeCLIP(), cfg["adapter"], cfg["head"])
names = ["A", "B", "C"]
tasks = [make_task(t) for t in range(3)]
data = {n: {"train": ds, "test": ds, "text": w} for n, (ds, w) in zip(names, tasks)}
ev = MTILEvaluator(model, data, names, cfg, torch.device("cpu"))
acc = {p: [] for p in ev.protocols}
zero_shot = ev.zero_shot_row()
for t, n in enumerate(names):
    train_task(model, data[n]["train"], data[n]["text"], cfg, torch.device("cpu"), lambda r, echo=True: None, n)
    ev.add_prototype(data[n]["train"], seed=t)
    rows, _ = ev.row(t + 1)
    for p in ev.protocols:
        acc[p].append(rows[p])

table = osnap.snapshot_table(ev, names)
assert [len(r) for r in table] == [1, 2, 3], table
for j in range(3):
    assert abs(table[j][0] - zero_shot[j]) < 1e-9, (j, table[j][0], zero_shot[j])
# same numbers as the run's own oracle_latest rows: acc[k-1][j] for k <= j (dataset j unseen after k tasks)
for j in range(3):
    for k in range(1, j + 1):
        assert abs(table[j][k] - acc["oracle_latest"][k - 1][j]) < 1e-9, (j, k)
rows, summary = osnap.summarise(table, names)
assert set(rows) == {"B", "C"}
_, ref = mtil_metrics(np.array(acc["oracle_latest"]), names)
assert abs(summary["latest"] - ref["Transfer"]) < 1e-9, (summary, ref)
assert abs(summary["zero_shot"] - (zero_shot[1] + zero_shot[2]) / 2) < 1e-9
for name, r in rows.items():
    a = np.array(table[names.index(name)])
    assert r["oracle_best"] >= r["zero_shot"] - 1e-9
    assert r["oracle_best"] <= a.max() + 1e-9
    assert r["zero_shot"] == a[0] and r["best_snapshot_acc"] == a[1:].max()
osnap.report(rows, summary)
print("oracle_snapshot_test: ok")
