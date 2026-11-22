"""Multi-process test of pact.distributed on CPU (gloo backend, 2 ranks, tiny fake CLIP).

    python tests/ddp_test.py

Checks: trainable parameters stay identical on every rank, alpha_tilde decisions agree,
sharded evaluation equals single-process evaluation, only rank 0 writes the log.
"""
import copy
import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import TensorDataset

D = 16


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        return x + self.mlp(self.ln_2(x))


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(12, D)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block() for _ in range(12)])

    def forward(self, x):
        h = self.proj(x)
        for b in self.transformer.resblocks:
            h = b(h)
        return h


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def encode_image(self, x):
        return self.visual(x)


def make_tasks():
    tasks = []
    for seed in range(2):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(96, 12, generator=g)
        y = torch.randint(0, 4, (96,), generator=g)
        w = nn.functional.normalize(torch.randn(4, D, generator=g), dim=-1)
        tasks.append((TensorDataset(x, y), w))
    return tasks


def overrides(out):
    return ["device=cpu", "num_workers=0", "train.iterations=40", "train.batch_size=16",
            "train.warmup=4", "train.log_every=10", "train.lr=1e-3", "pact.alpha_every=3",
            "pact.schedule.warmup_frac=0.1", "pact.schedule.window=4",
            "pact.schedule.rel_tol=0.5", "pact.schedule.ramp_iters=5", f"output_dir={out}",
            "eval.batch_size=16", "eval.router.samples=50"]


def evaluator_rows(model, tasks, cfg, device):
    """Task-aware evaluator (snapshots + router) on 2 tasks, the odd-sized set as 3rd 'dataset'."""
    from pact.evaluation import MTILEvaluator
    odd = torch.utils.data.Subset(tasks[0][0], list(range(37)))
    names = ["t0", "t1", "odd"]
    data = {"t0": {"test": tasks[0][0], "text": tasks[0][1]},
            "t1": {"test": tasks[1][0], "text": tasks[1][1]},
            "odd": {"test": odd, "text": tasks[0][1]}}
    ev = MTILEvaluator(model, data, names, cfg, device)
    out = {"zs": ev.zero_shot_row()}
    for t in range(2):
        ev.add_prototype(tasks[t][0], seed=t)
        out[t] = ev.row(t + 1)
    return out


def worker(rank, world, port, out, ref_state):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world),
                      RANK=str(rank), LOCAL_RANK=str(rank), USE_LIBUV="0")
    from pact import distributed as dist
    from pact.config import load_config
    from pact.engine import Logger, evaluate, set_seed, train_task
    from pact.model import PACTCLIP

    cfg = load_config(None, overrides(out))
    device = dist.init(cfg["device"])
    assert dist.world_size() == world and dist.rank() == rank
    set_seed(cfg["seed"] + rank)  # different RNG per rank on purpose: broadcast must fix it
    clip = FakeCLIP()
    clip.load_state_dict(ref_state)
    model = PACTCLIP(clip, cfg["adapter"], cfg["head"])
    log = Logger(out)
    tasks = make_tasks()
    for t, (ds, w) in enumerate(tasks):
        train_task(model, ds, w, cfg, device, log, f"t{t}")

    # 1) identical trainable state on every rank
    flat = torch.cat([p.detach().reshape(-1) for wr in model.wrappers
                      for a in wr.adapters for p in a.parameters()] +
                     [g.detach().reshape(-1) for wr in model.wrappers for g in wr.gates] +
                     [p.detach().reshape(-1) for p in model.heads.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    tdist.all_gather(gathered, flat)
    for other in gathered[1:]:
        assert torch.equal(gathered[0], other), "parameters diverged across ranks"

    # 2) sharded evaluation (odd test size, so shards are unequal)
    accs = [evaluate(model, tasks[i][0], tasks[i][1], cfg, device) for i in range(2)]
    odd = torch.utils.data.Subset(tasks[0][0], list(range(37)))
    accs.append(evaluate(model, odd, tasks[0][1], cfg, device))
    rows = evaluator_rows(model, tasks, cfg, device)
    if rank == 0:
        torch.save({"state": model.adapter_state(), "accs": accs, "rows": rows},
                   os.path.join(out, "r0.pt"))
    dist.cleanup()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    from pact.config import load_config
    from pact.engine import evaluate
    from pact.model import PACTCLIP

    torch.manual_seed(0)
    ref = FakeCLIP()
    out = tempfile.mkdtemp()
    world = 2
    mp.spawn(worker, args=(world, free_port(), out, ref.state_dict()), nprocs=world, join=True)

    saved = torch.load(os.path.join(out, "r0.pt"), weights_only=False)
    cfg = load_config(None, overrides(out))
    model = PACTCLIP(copy.deepcopy(ref), cfg["adapter"], cfg["head"])
    model.load_adapter_state(saved["state"])
    tasks = make_tasks()
    single = [evaluate(model, tasks[i][0], tasks[i][1], cfg, torch.device("cpu")) for i in range(2)]
    odd = torch.utils.data.Subset(tasks[0][0], list(range(37)))
    single.append(evaluate(model, odd, tasks[0][1], cfg, torch.device("cpu")))
    assert all(abs(a - b) < 1e-9 for a, b in zip(saved["accs"], single)), (saved["accs"], single)
    print("parameters identical across ranks: OK")
    print("sharded eval == single-process eval:", [round(a, 2) for a in single], "OK")
    rows = evaluator_rows(model, tasks, cfg, torch.device("cpu"))
    assert rows == saved["rows"], (rows, saved["rows"])
    print("sharded task-aware evaluator (snapshots, router) == single-process:",
          {p: [round(a, 1) for a in v] for p, v in rows[1][0].items()}, "OK")

    with open(os.path.join(out, "log.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    starts = [r for r in records if r.get("task") == "t0" and r.get("it") == 0]
    assert len(starts) == 1, "log must be written by rank 0 only"
    assert any("alpha_tilde" in r for r in records), "constraint never switched on"
    print("log written once (rank 0 only):", len(records), "records OK")
    print("ALL OK")


if __name__ == "__main__":
    main()
