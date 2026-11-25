"""Per-sample router analysis on a mixed stream (Direction A1 of the research plan).

Two stages. `extract` needs a GPU (once); `analyze` runs on CPU in seconds.

  # 1. frozen-CLIP features of train/test images + class text embeddings -> one .npz
  python analysis/router_persample.py extract --config configs/mtil_order1.yaml \\
      --out outputs/router_features.npz --set data_root=data mtil.order=[Aircraft,DTD,EuroSAT,OxfordPet]

  # 2. routing quality for several scores, per sample and per batch
  python analysis/router_persample.py analyze outputs/router_features.npz [--alpha 0.05]

Why: the router in pact/evaluation.py decides per test BATCH from the batch-mean feature, which
only works because the loader serves one dataset per batch, and it was only ever measured on 4-11
very different datasets (100% own-task accuracy). Here every test image is routed on its own, as if
the 11 test sets were shuffled together, with row t = "tasks 1..t learned, tasks t+1.. unseen":
  seen images  : routed to their own task / to frozen CLIP / to another task
  unseen images: wrongly routed to a learned snapshot (false route); AUROC and FPR95 of "seen vs
                 unseen" using max_k margin_k.
Scores (per task k; decision = argmax_k (score_k - thr_k), accept if >= 0, else frozen CLIP):
  proto    cosine to the normalised mean train feature (the repo's router)
  knn      k-th largest cosine to the task's train features
  maha     negative Mahalanobis distance, per-task Gaussian with shrunk covariance (DIKI-like)
  maxcos   max cosine to the task's class text embeddings (max-logit)
  mcm      max softmax prob of the task's classes over the union of learned classes (MCM-like)
Thresholds thr_k = alpha-quantile of the task's score on a held-out half of its train features
(conformal-style: about 1 - alpha of seen data is accepted). `proto_repo` calibrates on the
same features that build the prototype, as pact/evaluation.py does (optimistic).
Mode "batch" averages per-sample scores over consecutive test batches of eval.batch_size and gives
every image of the batch the batch decision (mimics the repo router, which needs single-dataset
batches); mode "sample" routes each image alone.

Not covered: Transfer/Avg/Last of the routed model (needs per-sample correctness of every snapshot;
see analysis/router_offline.py for the batch version).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
METHODS = {  # name -> (scorer, threshold calibration data)
    "proto": ("proto", "held"), "proto_repo": ("proto", "fit"), "knn": ("knn", "held"),
    "maha": ("maha", "held"), "maxcos": ("maxcos", "held"), "mcm": ("mcm", "held")}
MODES = ("sample", "batch")


# ============================================================================ stage 1: extract
def extract(args):
    sys.path.insert(0, ROOT)
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from pact.config import load_config
    from pact.data import build_mtil_dataset, eval_subset
    from pact.engine import build_clip, resolve_device
    from pact.text_head import build_text_classifier

    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg["device"])
    clip_model, preprocess, tokenizer = build_clip(cfg, device)
    bs, workers = cfg["eval"]["batch_size"], cfg["num_workers"]

    @torch.no_grad()
    def encode(dataset):
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=workers)
        out = []
        for images, _ in loader:  # frozen CLIP, same as snapshot 0 / the router's features
            out.append(F.normalize(clip_model.encode_image(images.to(device)).float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    arrays, names = {}, cfg["mtil"]["order"]
    for t, name in enumerate(names):
        train, test, classnames, templates = build_mtil_dataset(name, cfg["data_root"], preprocess,
                                                                cfg["download"])
        test = eval_subset(test, cfg["eval"]["max_samples"])  # same images as the training run
        n = min(len(train), args.train_samples)
        idx = np.random.RandomState(cfg["seed"] + t).choice(len(train), n, replace=False).tolist()
        arrays[f"{name}/train"] = encode(Subset(train, idx)).astype(np.float16)
        arrays[f"{name}/test"] = encode(test).astype(np.float16)
        text = build_text_classifier(clip_model, tokenizer, classnames, templates, device,
                                     cfg["eval"]["template_mode"])
        arrays[f"{name}/text"] = text.float().cpu().numpy().astype(np.float16)
        print(f"{name}: train {n}, test {len(test)}, classes {len(classnames)}", flush=True)
    meta = {"order": names, "batch_size": bs, "seed": cfg["seed"], "train_samples": args.train_samples,
            "max_samples": cfg["eval"]["max_samples"], "logit_scale": float(clip_model.logit_scale.detach().exp())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **arrays)
    print(f"written: {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MB; float16, renormalised on load)")


# ============================================================================ stage 2: analyze
def load(path):
    import torch
    import torch.nn.functional as F
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))

    def get(key):
        return F.normalize(torch.from_numpy(z[key].astype(np.float32)), dim=-1)

    d = {"meta": meta, "order": meta["order"], "bs": int(meta["batch_size"]),
         "scale": float(meta.get("logit_scale", 100.0)), "fit": {}, "calib": {}, "test": {}, "text": {}}
    for n in d["order"]:
        train = get(f"{n}/train")
        h = len(train) // 2  # train subset is already a random draw: first half fits, second calibrates
        d["fit"][n], d["calib"][n] = train[:h], train[h:]
        d["test"][n], d["text"][n] = get(f"{n}/test"), get(f"{n}/text")
    return d


class Scorers:
    """score(method, X, t) -> (N, t): score of every image for tasks 1..t (higher = more 'seen')."""

    def __init__(self, d, knn_k=5, shrink=0.1):
        import torch
        self.d, self.knn_k = d, knn_k
        self.names = d["order"]
        self.protos = torch.stack([torch.nn.functional.normalize(d["fit"][n].mean(0), dim=0) for n in self.names])
        self.mu, self.prec = [], []
        for n in self.names:
            x = d["fit"][n].double()
            mu = x.mean(0)
            cov = torch.cov((x - mu).t())
            cov = (1 - shrink) * cov + shrink * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype)
            self.mu.append(mu.float())
            self.prec.append(torch.linalg.inv(cov).float())

    def score(self, method, X, t):
        import torch
        if method == "proto":
            return X @ self.protos[:t].t()
        if method == "knn":
            cols = [(X @ self.d["fit"][n].t()).topk(min(self.knn_k, len(self.d["fit"][n])), dim=1).values[:, -1]
                    for n in self.names[:t]]
            return torch.stack(cols, 1)
        if method == "maha":
            cols = []
            for k in range(t):
                z = X - self.mu[k]
                cols.append(-((z @ self.prec[k]) * z).sum(1))
            return torch.stack(cols, 1)
        if method == "maxcos":
            return torch.stack([(X @ self.d["text"][n].t()).max(1).values for n in self.names[:t]], 1)
        if method == "mcm":
            texts = [self.d["text"][n] for n in self.names[:t]]
            probs = torch.softmax(self.d["scale"] * X @ torch.cat(texts).t(), dim=1)
            cols, start = [], 0
            for tx in texts:
                cols.append(probs[:, start:start + len(tx)].max(1).values)
                start += len(tx)
            return torch.stack(cols, 1)
        raise ValueError(method)


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg), average ranks for ties."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    ranks = ((cum - cnt + 1 + cum) / 2)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fpr_at_tpr95(seen, unseen):
    if len(seen) == 0 or len(unseen) == 0:
        return float("nan")
    thr = np.quantile(seen, 0.05)  # 95% of seen data scores >= thr
    return float((unseen >= thr).mean())


def decide(margin):
    """margin (N, t) -> decision per row (0 = CLIP, k = task k) and detection score max margin."""
    best, top = margin.argmax(1), margin.max(1)
    dec = np.where(top >= 0, best + 1, 0)
    return dec, top


def batch_mean(scores, bs):
    """Mean score over consecutive batches, repeated for every image of the batch."""
    n = len(scores)
    out = np.empty_like(scores)
    for s in range(0, n, bs):
        out[s:s + bs] = scores[s:s + bs].mean(0)
    return out


def analyze(args):
    import torch
    d = load(args.features)
    names, T = d["order"], len(d["order"])
    sc = Scorers(d, args.knn_k, args.shrink)
    methods = args.methods or list(METHODS)
    res = {m: {mode: {"rows": [], "per_dataset": {n: {"unseen_false": [], "seen_own": []} for n in names}}
               for mode in MODES} for m in methods}
    for m in methods:
        scorer, calib = METHODS[m]
        for t in range(1, T + 1):
            thr = []
            for k in range(t):
                src = d["calib" if calib == "held" else "fit"][names[k]]
                thr.append(float(np.quantile(sc.score(scorer, src, t)[:, k].numpy(), args.alpha)))
            thr = np.array(thr)
            raw = {n: sc.score(scorer, d["test"][n], t).numpy() for n in names}
            for mode in MODES:
                dec, det = {}, {}
                for n in names:
                    s = raw[n] if mode == "sample" else batch_mean(raw[n], d["bs"])
                    dec[n], det[n] = decide(s - thr)
                own, clip, other, false = [], [], [], []
                for j, n in enumerate(names, start=1):
                    if j <= t:
                        r = (dec[n] == j).mean()
                        own.append(r)
                        clip.append((dec[n] == 0).mean())
                        other.append(((dec[n] != j) & (dec[n] != 0)).mean())
                        res[m][mode]["per_dataset"][n]["seen_own"].append(float(r))
                    else:
                        f = (dec[n] != 0).mean()
                        false.append(f)
                        res[m][mode]["per_dataset"][n]["unseen_false"].append(float(f))
                seen = np.concatenate([det[n] for n in names[:t]])
                unseen = np.concatenate([det[n] for n in names[t:]]) if t < T else np.array([])
                res[m][mode]["rows"].append({
                    "t": t, "seen_own": float(np.mean(own)), "seen_to_clip": float(np.mean(clip)),
                    "seen_to_other": float(np.mean(other)),
                    "unseen_false": float(np.mean(false)) if false else float("nan"),
                    "auroc": auroc(seen, unseen), "fpr95": fpr_at_tpr95(seen, unseen)})
    report(res, names, args)
    out = args.out or os.path.splitext(args.features)[0] + "_persample.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": d["meta"], "results": res}, f, indent=1)
    print(f"\nwritten: {out}")


def report(res, names, args):
    def mean(rows, key, only_unseen=False):
        v = [r[key] for r in rows if not (only_unseen and r["t"] == len(names))]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    print(f"\nalpha={args.alpha} (share of a task's own held-out data rejected), averaged over rows t=1..{len(names) - 1}"
          f" (tasks 1..t learned)\n{'method':11s}{'mode':8s}{'seen->own':>10s}{'seen->CLIP':>11s}{'seen->other':>12s}"
          f"{'unseen->snap':>13s}{'AUROC':>8s}{'FPR95':>8s}")
    for m, modes in res.items():
        for mode, r in modes.items():
            rows = r["rows"]
            print(f"{m:11s}{mode:8s}{100 * mean(rows, 'seen_own', True):10.1f}{100 * mean(rows, 'seen_to_clip', True):11.1f}"
                  f"{100 * mean(rows, 'seen_to_other', True):12.1f}{100 * mean(rows, 'unseen_false', True):13.1f}"
                  f"{mean(rows, 'auroc', True):8.3f}{mean(rows, 'fpr95', True):8.3f}")
    for mode in MODES:
        print(f"\nunseen -> wrongly sent to a learned snapshot, % per dataset (rows where it is unseen), mode={mode}")
        print(f"{'method':11s}" + "".join(f"{n[:9]:>10s}" for n in names[1:]))
        for m, modes in res.items():
            cells = []
            for n in names[1:]:
                v = modes[mode]["per_dataset"][n]["unseen_false"]
                cells.append(f"{100 * np.mean(v):10.1f}" if v else f"{'-':>10s}")
            print(f"{m:11s}" + "".join(cells))
    print("\nbatch vs sample gap (pp of seen->own and unseen->snap) shows how much of the batch router's"
          " quality comes from the one-dataset-per-batch assumption.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="GPU: frozen-CLIP features -> .npz")
    e.add_argument("--config", default=None)
    e.add_argument("--set", nargs="*", default=[], help="config overrides, same as pact.run_mtil")
    e.add_argument("--out", required=True)
    e.add_argument("--train-samples", type=int, default=2000, help="train images per dataset (half fit, half calibrate)")
    a = sub.add_parser("analyze", help="CPU: routing quality per score, per sample and per batch")
    a.add_argument("features")
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--knn-k", type=int, default=5)
    a.add_argument("--shrink", type=float, default=0.1, help="covariance shrinkage for maha")
    a.add_argument("--methods", nargs="*", choices=list(METHODS))
    a.add_argument("--out", default=None)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else analyze(args)


if __name__ == "__main__":
    main()
