"""Test of analysis/router_offline.py on a synthetic run (tiny fake CLIP, CPU, no downloads).

    python tests/router_offline_test.py

Builds 3 tasks whose inputs are clearly separated, trains them with the real train_task /
MTILEvaluator, saves a state.pt in the same format as pact/run_mtil.py, then checks that the
offline replay at delta = 0 reproduces the run's router protocol, that learned batches are
routed to their own task, and that a very strict threshold sends everything to CLIP.
"""
import json, os, subprocess, sys, tempfile
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from pact.config import load_config
from pact.engine import train_task
from pact.evaluation import MTILEvaluator
from pact.model import PACTCLIP

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
    x[:, t] += 4.0                                   # each task lives in its own direction
    y = torch.randint(0, ncls, (n,), generator=g)
    x[torch.arange(n), 4 + y] += 1.0                 # learnable class signal
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
    train_task(model, data[n]["train"], data[n]["text"], cfg, torch.device("cpu"),
               lambda r, echo=True: None, n)
    ev.add_prototype(data[n]["train"], seed=t)
    rows, _ = ev.row(t + 1)
    for p in ev.protocols:
        acc[p].append(rows[p])

out = tempfile.mkdtemp()
state_path = os.path.join(out, "state.pt")
torch.save({"adapters": model.adapter_state(), "acc": acc, "zero_shot": zero_shot,
            "order": names, "evaluator": ev.state_dict()}, state_path)

small_path = os.path.join(out, "evaluator.pt")   # the small file run_mtil.py writes next to state.pt
torch.save({"order": names, "acc": acc, "zero_shot": zero_shot, "learned_heads": model.use_heads,
            "evaluator": ev.state_dict()}, small_path)
for path in (state_path, small_path):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "router_offline.py"), path, "--deltas", "0"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0 and "-> OK" in p.stdout, (path, p.stdout[-1500:], p.stderr[-1500:])
print("state.pt and evaluator.pt both readable: OK")
proc = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "router_offline.py"), state_path,
                       "--deltas", "-0.05", "0", "0.05", "2.0"],
                      capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print(proc.stdout[-3500:]); print(proc.stderr[-2000:])
assert proc.returncode == 0
res = json.load(open(os.path.join(out, "router_offline.json"), encoding="utf-8"))
assert res["consistency_ok"] is True, "replay at delta=0 must equal the run's router protocol"
zero = next(r for r in res["sweep"] if r["delta"] == 0.0)
assert zero["learned_to_own_%"] == 100.0, zero
strict = next(r for r in res["sweep"] if r["delta"] == 2.0)
assert strict["learned_to_clip_%"] == 100.0 and strict["unseen_to_clip_%"] == 100.0, strict
assert strict["summary"] is not None                      # all-CLIP decisions are always cached
assert abs(strict["summary"]["Last"] - sum(zero_shot) / 3) < 1e-9
print("ALL OK")
