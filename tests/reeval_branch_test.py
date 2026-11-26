"""Test of analysis/reeval_branch.py (core function) with a tiny fake CLIP on CPU.

    python tests/reeval_branch_test.py

Checks: cond/cond reproduces the evaluator used during training; marg/marg and marg/cond differ from it once
there are several adapters; marg/cond uses the cond head (equal to marg/marg when both heads are identical);
learned columns stay invariant (supp. Table 8) in every variant; zero-shot passes are shared.
"""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from pact.config import load_config
from pact.engine import train_task
from pact.evaluation import MTILEvaluator
from pact.model import PACTCLIP
from reeval_branch import evaluate_variants

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


def make_task(t, n=80, ncls=4):
    g = torch.Generator().manual_seed(t)
    x = 0.5 * torch.randn(n, 12, generator=g); x[:, t] += 2.0
    y = torch.randint(0, ncls, (n,), generator=g); x[torch.arange(n), 4 + y] += 1.5
    return TensorDataset(x, y), nn.functional.normalize(torch.randn(ncls, D, generator=g), dim=-1)


torch.manual_seed(0)
cfg = load_config(None, ["device=cpu", "num_workers=0", "train.iterations=30", "train.batch_size=16",
                         "train.warmup=2", "train.log_every=100", "train.lr=3e-3", "head.lr_mult=5",
                         "eval.batch_size=16", "eval.protocols=[oracle_zeroshot,oracle_latest]",
                         "pact.schedule.warmup_frac=0.1"])
model = PACTCLIP(FakeCLIP(), cfg["adapter"], cfg["head"])
names = ["A", "B", "C"]
tasks = [make_task(t) for t in range(3)]
data = {n: {"train": ds, "test": ds, "text": w} for n, (ds, w) in zip(names, tasks)}
ev = MTILEvaluator(model, data, names, cfg, torch.device("cpu"))
ref = {p: [] for p in ev.protocols}
for t, n in enumerate(names):
    train_task(model, data[n]["train"], data[n]["text"], cfg, torch.device("cpu"), lambda r, echo=True: None, n)
    rows, _ = ev.row(t + 1)
    for p in ev.protocols:
        ref[p].append(rows[p])

res = evaluate_variants(model, data, names, cfg, torch.device("cpu"))
assert res["cond/cond"]["rows"] == ref, "cond/cond must reproduce the training-time evaluation"
for v, r in res.items():
    m = r["rows"]["oracle_zeroshot"]
    assert all(m[i][j] == m[j][j] for i in range(3) for j in range(i + 1)), f"{v}: learned columns changed"
    # unseen data uses frozen CLIP in oracle_zeroshot: identical for every variant
    assert all(m[i][j] == ref["oracle_zeroshot"][i][j] for i in range(3) for j in range(i + 1, 3)), v
# task 1 has a single adapter: cond and marg features are equal, so cond/cond == marg/cond there
assert res["marg/cond"]["rows"]["oracle_zeroshot"][0][0] == ref["oracle_zeroshot"][0][0]
diag = {v: [r["rows"]["oracle_zeroshot"][j][j] for j in range(3)] for v, r in res.items()}
print("diagonal per variant:", {v: [round(a, 1) for a in d] for v, d in diag.items()})
assert diag["marg/marg"] != diag["cond/cond"] or diag["marg/cond"] != diag["cond/cond"], \
    "marg variants should differ from cond once several adapters exist"

# head override: with identical cond / marg heads, marg/cond == marg/marg
with torch.no_grad():
    for h in model.heads:
        h.marg.copy_(h.cond)
res2 = evaluate_variants(model, data, names, cfg, torch.device("cpu"), variants=("marg/marg", "marg/cond"))
assert res2["marg/marg"]["rows"] == res2["marg/cond"]["rows"]
print({v: {p: {k: round(x, 2) for k, x in s.items()} for p, s in r["summary"].items()} for v, r in res.items()})
print("ALL OK")
