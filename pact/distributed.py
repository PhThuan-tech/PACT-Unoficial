"""Optional multi-GPU (data-parallel) support, launched with torchrun.

    torchrun --standalone --nproc_per_node=2 -m pact.run_mtil --config configs/mtil_order1.yaml

Design:
- Every process holds a full model; parameters start identical (broadcast from rank 0).
- `train.batch_size` stays the GLOBAL batch: each rank loads batch_size / world_size samples.
  The ViT has no BatchNorm, so this is the same objective as one GPU with the full batch.
- Gradients of the trainable adapter / gate are averaged with all_reduce after backward
  (no DistributedDataParallel wrapper: each iteration runs two forward passes, cond and
  marg, before a single backward, which a manual all-reduce handles without caveats).
- The loss fed to the post-convergence schedule is averaged over ranks, so every rank
  takes the same alpha_tilde decision. alpha_t itself is computed from the (identical)
  weights and a fixed probe bank, so it is identical on every rank.
- Evaluation shards the test set without padding and sums correct / total counts, so the
  accuracy is exactly the single-GPU one.
- Only rank 0 writes logs, checkpoints and results.

With a single process (plain `python -m ...`) every helper is a no-op.
"""

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def init(device_name):
    """Initialise the process group if launched by torchrun; return the device to use."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None  # single process: caller resolves the device as before
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_name.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    kwargs = dict(backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=60))
    if use_cuda:
        kwargs["device_id"] = device  # binds rank -> GPU (silences NCCL "guessing device ID")
    try:
        dist.init_process_group(**kwargs)
    except TypeError:  # torch < 2.3 has no device_id
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)
    return device


def cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def barrier():
    if is_dist():
        dist.barrier()


@contextmanager
def main_first():
    """Rank 0 runs the block first (e.g. dataset download), the others wait, then run it."""
    if is_dist() and not is_main():
        dist.barrier()
    yield
    if is_dist() and is_main():
        dist.barrier()


def broadcast_params(params):
    if is_dist():
        for p in params:
            dist.broadcast(p.data, src=0)


def all_reduce_grads(params):
    """Average .grad over ranks (missing grads count as zero so every rank calls the same ops)."""
    if not is_dist():
        return
    n = world_size()
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(n)


def mean_scalar(value, device):
    if not is_dist():
        return value
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size()


def sum_counts(correct, total, device):
    if not is_dist():
        return correct, total
    t = torch.tensor([correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t[0].item()), int(t[1].item())


def shard_indices(n):
    """Indices of this rank's share of n items (strided, no padding, no duplicates)."""
    return list(range(rank(), n, world_size()))


def sum_tensor(t):
    """Element-wise sum over ranks (CPU tensors are moved to the process device and back)."""
    if not is_dist():
        return t
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() \
        and dist.get_backend() == "nccl" else torch.device("cpu")
    buf = t.to(device=device, dtype=torch.float64).contiguous()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return buf.to(device=t.device, dtype=t.dtype)
