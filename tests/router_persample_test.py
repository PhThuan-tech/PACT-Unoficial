"""Test of the `analyze` stage of analysis/router_persample.py on synthetic features (CPU, no CLIP).

    python tests/router_persample_test.py

Four fake datasets = Gaussian clusters on the unit sphere (D=64). Checks that
  - well separated datasets are routed to their own task, unseen ones go to CLIP (per sample and per batch),
  - AUROC is high and bounded, FPR95 in [0, 1],
  - a dataset that overlaps another one (same centre) shows a clearly higher unseen false-route rate,
  - the JSON output is written and holds every method x mode.
"""
import json, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D, N_TRAIN, N_TEST, BS, C = 64, 600, 400, 50, 5
rng = np.random.RandomState(0)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make(path, overlap):
    names = ["A", "B", "C", "D"]
    centres = unit(rng.randn(4, D))
    if overlap:  # dataset D lives where dataset B lives
        centres[3] = unit((centres[1] + 0.05 * rng.randn(D))[None])[0]
    arrays = {}
    for i, n in enumerate(names):
        cls = unit(centres[i] + 0.05 * rng.randn(C, D))
        for split, count in (("train", N_TRAIN), ("test", N_TEST)):
            y = rng.randint(0, C, count)
            arrays[f"{n}/{split}"] = unit(cls[y] + 0.08 * rng.randn(count, D)).astype(np.float16)
        arrays[f"{n}/text"] = cls.astype(np.float16)
    meta = {"order": names, "batch_size": BS, "logit_scale": 100.0}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def run(path, out):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "router_persample.py"), "analyze",
                        path, "--out", out], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    with open(out, encoding="utf-8") as f:
        return json.load(f)["results"], r.stdout


def mean_rows(res, m, mode, key):
    v = [r[key] for r in res[m][mode]["rows"] if r["t"] < 4 and r[key] == r[key]]
    return float(np.mean(v))


with tempfile.TemporaryDirectory() as tmp:
    sep, ov = os.path.join(tmp, "sep.npz"), os.path.join(tmp, "ov.npz")
    make(sep, False)
    make(ov, True)
    res, stdout = run(sep, os.path.join(tmp, "sep.json"))
    assert set(res) == {"proto", "proto_repo", "knn", "maha", "maxcos", "mcm"}, res.keys()
    for m in res:
        assert set(res[m]) == {"sample", "batch"}
        for mode in res[m]:
            assert len(res[m][mode]["rows"]) == 4
    # separated clusters: batch routing is near perfect on the learned side
    assert mean_rows(res, "proto", "batch", "seen_own") > 0.9, mean_rows(res, "proto", "batch", "seen_own")
    # per-sample AUROC high and valid for every score that sees the image features
    for m in ("proto", "knn", "maha"):
        auc = mean_rows(res, m, "sample", "auroc")
        assert 0.9 < auc <= 1.0, (m, auc)
        assert 0.0 <= mean_rows(res, m, "sample", "fpr95") <= 1.0
    # conformal calibration: about 1 - alpha of held-out seen data is accepted (seen_to_clip ~ alpha)
    clip = mean_rows(res, "proto", "sample", "seen_to_clip")
    assert clip < 0.15, clip
    # overlapping dataset D (same place as B): when B is learned and D is unseen the false-route rate is high
    res_ov, _ = run(ov, os.path.join(tmp, "ov.json"))
    d_false = np.mean(res_ov["proto"]["sample"]["per_dataset"]["D"]["unseen_false"])
    d_false_sep = np.mean(res["proto"]["sample"]["per_dataset"]["D"]["unseen_false"])
    assert d_false > d_false_sep + 0.2, (d_false, d_false_sep)
    assert "unseen->snap" in stdout and "AUROC" in stdout
print("router_persample_test: ok")
