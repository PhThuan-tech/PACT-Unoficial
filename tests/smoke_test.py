"""Smoke test of PACT logic with a tiny fake CLIP (no open_clip, no downloads, CPU).

    python tests/smoke_test.py
"""
import copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from pact.config import load_config
from pact.model import PACTCLIP
from pact.engine import train_task, evaluate, Logger
from pact.phase import ProbeBank, compute_alpha
from pact.metrics import mtil_metrics, format_mtil_table
from pact.data.cil import task_splits, class_order

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

torch.manual_seed(0)
cfg = load_config(None, ["device=cpu", "num_workers=0", "train.iterations=60", "train.batch_size=8",
                         "train.warmup=5", "train.log_every=20", "train.lr=1e-3",
                         "pact.schedule.warmup_frac=0.1", "pact.schedule.window=5",
                         "pact.schedule.rel_tol=0.5", "pact.schedule.ramp_iters=10", "pact.alpha_every=5"])
assert cfg["mtil"]["order"][0] == "Aircraft"
clip = FakeCLIP(); ref = copy.deepcopy(clip)
model = PACTCLIP(clip, cfg["adapter"])
x = torch.randn(32, 12)
model.eval()
base = model.encode_image(x, "base")

# task 1: SAI must be prediction-preserving at insertion (both branches)
model.add_task()
for br in ("marg", "cond"):
    assert torch.allclose(model.encode_image(x, br), base, atol=1e-5), br
assert torch.allclose(base, torch.nn.functional.normalize(ref.encode_image(x), dim=-1), atol=1e-5)
print("SAI prediction-preserving: OK")
probes = ProbeBank(D, 64, 0, "cpu")
a, info = compute_alpha(model, probes, 0.1)
assert abs(a - 1.0) < 1e-6, a  # clone of the pretrained MLP -> frozen regime
print("alpha at insertion (vs pretrained MLP) =", a)
model.num_tasks -= 1
for w in model.wrappers:  # undo, train_task adds it again
    w.adapters = nn.ModuleList(); w.gates = nn.ParameterList(); w.anchors = []

def make_task(seed, ncls=4, n=64):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 12, generator=g); Y = torch.randint(0, ncls, (n,), generator=g)
    return TensorDataset(X, Y), nn.functional.normalize(torch.randn(ncls, D, generator=g), dim=-1)

log = Logger(tempfile.mkdtemp())
tasks = [make_task(s) for s in range(3)]
acc = []
snap = None
for t, (ds, W) in enumerate(tasks):
    train_task(model, ds, W, cfg, torch.device("cpu"), log, f"t{t}")
    assert model.num_adapters == t + 1
    if snap is not None:  # previous adapters / gates must stay frozen
        for w, (ps, gs) in zip(model.wrappers, snap):
            for a_old, p_old in zip(w.adapters[:-1], ps):
                for p, q in zip(a_old.parameters(), p_old): assert torch.equal(p, q)
            for g, g0 in zip(w.gates[:-1], gs): assert torch.equal(g, g0)
    snap = [([[p.detach().clone() for p in a.parameters()] for a in w.adapters],
             [g.detach().clone() for g in w.gates]) for w in model.wrappers]
    acc.append([evaluate(model, d, Wd, cfg, torch.device("cpu")) for d, Wd in tasks])
print("old adapters frozen: OK")
assert all(not p.requires_grad for n_, p in clip.named_parameters() if "adapters" not in n_ and "gates" not in n_)
a, info = compute_alpha(model, probes, 0.1)
print("alpha after 3 tasks =", round(a, 4), info)
rows, summary = mtil_metrics(acc, ["A", "B", "C"])
print(format_mtil_table(rows, summary, ["A", "B", "C"]))

# checkpoint round-trip
state = model.adapter_state()
clip2 = copy.deepcopy(ref); m2 = PACTCLIP(clip2, cfg["adapter"]); m2.load_adapter_state(state); m2.eval()
assert torch.allclose(m2.encode_image(x, "cond"), model.encode_image(x, "cond"), atol=1e-6)
print("checkpoint round-trip: OK")

# ablation variants run
for over in (["adapter.type=lora", "adapter.use_gate=false", "pact.use_marg=false", "pact.use_pa=false"],
             ["adapter.clone_mlp=false", "adapter.use_gate=false"],
             ["adapter.per_task=false"], ["pact.post_convergence=false", "pact.use_phase=false"]):
    c2 = load_config(None, ["device=cpu", "num_workers=0", "train.iterations=15", "train.batch_size=8",
                            "train.warmup=2", "train.log_every=100"] + over)
    m = PACTCLIP(copy.deepcopy(ref), c2["adapter"])
    for ds, W in tasks[:2]: train_task(m, ds, W, c2, torch.device("cpu"), lambda r, echo=True: None, "x")
    evaluate(m, tasks[0][0], tasks[0][1], c2, torch.device("cpu"))
    st = m.adapter_state(); m3 = PACTCLIP(copy.deepcopy(ref), c2["adapter"]); m3.load_adapter_state(st)
    print("variant OK:", over, "adapters:", m.num_adapters)

# SAI at task t >= 2: conditional branch unchanged at insertion (gate g_t = g_{t-1} t/(t-1))
from pact.losses import prior_anchoring_loss
def drift_last(m, scale, gamma):
    with torch.no_grad():
        for w in m.wrappers:
            for p in w.adapters[-1].parameters(): p.add_(scale * torch.randn_like(p))
            w.gates[-1].fill_(gamma)
cfg_learned = load_config(None, ["adapter.gate_mode=learned"])
m = PACTCLIP(copy.deepcopy(ref), cfg_learned["adapter"]); m.eval()
m.add_task(); drift_last(m, 0.05, -1.0)            # g_1 = 0.269 -> g_2 = 0.538 (exact)
before = m.encode_image(x, "cond"); m.add_task()
assert torch.allclose(m.encode_image(x, "cond"), before, atol=1e-5), "SAI task 2"
drift_last(m, 0.05, -0.5)                           # g_2 = 0.378 -> g_3 = 0.567 (exact)
before = m.encode_image(x, "cond"); m.add_task()
assert torch.allclose(m.encode_image(x, "cond"), before, atol=1e-5), "SAI task 3"
print("SAI prediction-preserving at tasks 2, 3: OK")
drift_last(m, 0.05, 3.0)                            # g_3 = 0.953 -> 1.27 > cap: clipped
m.add_task()
g4 = torch.sigmoid(m.wrappers[0].gates[-1]).item()
assert abs(g4 - cfg_learned["adapter"]["gate_max_init"]) < 1e-5, g4
print("gate clipped at gate_max_init:", round(g4, 4))

# L_pa: zero at insertion, 0.5 at a relative drift of exactly sigma_rel
assert prior_anchoring_loss(m, 0.2).item() == 0.0
with torch.no_grad():
    for w in m.wrappers:
        for p in w.adapters[-1].parameters(): p.mul_(1.2)   # relative drift 0.2
assert abs(prior_anchoring_loss(m, 0.2).item() - 0.5) < 1e-4
print("L_pa relative scale: OK")

# post-convergence schedule: always starts (deadline) even if the loss keeps decreasing
from pact.phase import PostConvergenceSchedule
sched_cfg = dict(load_config(None)["pact"]["schedule"], mode="stability", ramp_iters=200)
sch = PostConvergenceSchedule(sched_cfg, 300)
s_vals = [sch.update(i, 3.0 * 0.99 ** i) for i in range(300)]   # still decreasing ~1%/iter
assert sch.started_at == 150 and sch.reason == "deadline", (sch.started_at, sch.reason)
assert s_vals[149] == 0 and s_vals[-1] == 1.0
sch = PostConvergenceSchedule(sched_cfg, 300)
for i in range(300): sch.update(i, 1.0)                         # flat loss: converged after warm-up
assert sch.reason == "converged" and sch.started_at == 60, (sch.started_at, sch.reason)
print("post-convergence schedule (deadline / converged): OK")

# windowed hard switch (supp. Eq. 40-44)
import random
win_cfg = load_config(None)["pact"]["schedule"]
assert win_cfg["mode"] == "window"
rng = random.Random(0)
sch = PostConvergenceSchedule(win_cfg, 1000)                    # W = 100
s_vals = [sch.update(i, 1.0 + 0.05 * rng.gauss(0, 1)) for i in range(1000)]   # CV 5% -> phi ~ 0.025
assert sch.reason == "converged" and sch.started_at == 200, (sch.started_at, sch.reason)
assert s_vals[199] == 0.0 and s_vals[200] == 1.0                 # hard switch, no ramp
sch = PostConvergenceSchedule(win_cfg, 1000)
for i in range(1000): sch.update(i, 1.0 + 0.6 * rng.gauss(0, 1) ** 2)  # very noisy: never stable
assert sch.reason == "deadline" and sch.started_at == 500, (sch.started_at, sch.reason)
print("windowed switch (converged / deadline): OK")

# layer-wise gate schedule (supp.): g = 0.5 * l / 12 at layers 3/6/9/12, frozen, same for all tasks
m = PACTCLIP(copy.deepcopy(ref), cfg["adapter"]); m.add_task(); m.add_task()
gs = [[round(torch.sigmoid(g).item(), 4) for g in w.gates] for w in m.wrappers]
assert gs == [[0.125] * 2, [0.25] * 2, [0.375] * 2, [0.5] * 2], gs
assert not m.gate_parameters()
print("layer-wise frozen gates:", [g[0] for g in gs])

# learned heads + task snapshots + evaluator protocols
from pact.evaluation import MTILEvaluator
hcfg = load_config(None, ["device=cpu", "num_workers=0", "train.iterations=30", "train.batch_size=8",
                          "train.warmup=2", "train.log_every=100", "train.lr=1e-3", "head.lr_mult=10",
                          "eval.batch_size=16", "eval.router.samples=64"])
m = PACTCLIP(copy.deepcopy(ref), hcfg["adapter"], hcfg["head"]); quiet = lambda r, echo=True: None
names = ["A", "B", "C"]
data = {n: {"train": ds, "test": ds, "text": Wd} for n, (ds, Wd) in zip(names, tasks)}
ev = MTILEvaluator(m, data, names, hcfg, torch.device("cpu"))
zs = ev.zero_shot_row()
assert abs(zs[0] - evaluate(m, tasks[0][0], tasks[0][1], hcfg, torch.device("cpu"), "base")) < 1e-9
rows_all = []
for t, n in enumerate(names):
    train_task(m, data[n]["train"], data[n]["text"], hcfg, torch.device("cpu"), quiet, n)
    assert len(m.heads) == t + 1 and all(not p.requires_grad for h in m.heads[:-1] for p in h.parameters())
    ev.add_prototype(data[n]["train"], seed=t)
    rows, routing = ev.row(t + 1)
    rows_all.append(rows)
assert not torch.allclose(m.heads[0].cond, tasks[0][1]), "head 0 did not train"
# Table-8 invariance: learned columns identical in later rows (recomputed from scratch here)
fresh = MTILEvaluator(m, data, names, hcfg, torch.device("cpu"))
for p in ("oracle_zeroshot", "oracle_latest"):
    for t in range(3):
        for j in range(t + 1):
            assert rows_all[t][p][j] == rows_all[j][p][j]
    assert abs(fresh.accuracy("A", 1, 1) - rows_all[0][p][0]) < 1e-9
# snapshot 0 = frozen CLIP, latest of last row = snapshot 3
assert rows_all[0]["oracle_zeroshot"][2] == zs[2]
assert rows_all[2]["latest"][2] == rows_all[2]["oracle_latest"][2]
print("heads trained, snapshots invariant, protocols:",
      {p: [round(a, 1) for a in rows_all[2][p]] for p in ev.protocols}, "routing:", routing)
st = ev.state_dict(); ev2 = MTILEvaluator(m, data, names, hcfg, torch.device("cpu")); ev2.load_state_dict(st)
assert ev2.row(3)[0] == rows_all[2]
# checkpoint round-trip with heads
m2 = PACTCLIP(copy.deepcopy(ref), hcfg["adapter"], hcfg["head"]); m2.load_adapter_state(m.adapter_state())
m2.eval(); m.eval()
for k in (1, 2, 3):
    with m.snapshot(k), m2.snapshot(k):
        assert torch.allclose(m.logits(x, m.head_weights(k)), m2.logits(x, m2.head_weights(k)), atol=1e-5)
print("evaluator cache + checkpoint with heads: OK")

assert task_splits(100, 10, 10)[-1] == list(range(90, 100)) and len(task_splits(200, 100, 20)) == 6
print("class order[:5] =", class_order(100, 1993)[:5])
print("ALL OK")
