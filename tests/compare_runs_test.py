"""Test tools/compare_runs.py trên run giả (không cần GPU).

    python tests/compare_runs_test.py

Kiểm tra: ghép cặp theo preset + seed, chênh Last, cả ba nhánh của quy tắc quyết định (không đóng góp,
có giá trị, chưa kết luận), cảnh báo khác bản code, bỏ qua run thiếu cặp, lấy run mới nhất khi trùng.
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ORDER = ["Aircraft", "DTD", "EuroSAT"]


def run(kind, seed, last, transfer=61.0):
    cfg = "configs/mtil_order1.yaml" if kind == "pact" else "configs/analysis/isolation_baseline.yaml"
    p = {"oracle_zeroshot": {"Transfer": 62.0, "Avg": 70.0, "Last": last},
         "oracle_latest": {"Transfer": transfer, "Avg": 70.0, "Last": last},
         "router": {"Transfer": transfer, "Avg": 70.0, "Last": last}}
    return {"config_file": cfg, "key_config": {"seed": seed}, "protocols": p, "zero_shot": [24.3, 44.6, 54.9]}


def write(dirname, rid, created, runs, code_hash="aaaa"):
    d = {"id": rid, "created": created, "preset": "full", "order": ORDER, "runs": runs,
         "code_build": {"id": "x-" + code_hash, "code_hash": code_hash}}
    with open(os.path.join(dirname, rid + ".json"), "w", encoding="utf-8") as f:
        json.dump(d, f)


def compare(tmp, *extra):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "compare_runs.py"), "--runs-dir", tmp, *extra],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def case(pairs, **kw):
    """pairs: {seed: (pact_last, iso_last)}"""
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for s, (lp, li) in pairs.items():
            runs[f"pact_s{s}"], runs[f"iso_s{s}"] = run("pact", s, lp), run("iso", s, li)
        write(tmp, "r1", "2026-11-26 10:00:00", runs, kw.get("hash_p", "aaaa"))
        return compare(tmp)


# 1. isolation >= PACT at both seeds -> không đóng góp
out = case({42: (77.0, 79.5), 2026: (77.4, 79.1)})
assert "không đóng góp vào Last" in out, out
# 2. PACT clearly better at both seeds -> có giá trị
out = case({42: (82.0, 79.5), 2026: (81.5, 79.0)})
assert "ràng buộc PACT có giá trị" in out.replace("Ràng buộc", "ràng buộc"), out
# 3. opposite signs -> chưa kết luận
out = case({42: (80.5, 79.5), 2026: (78.0, 79.0)})
assert "CHƯA KẾT LUẬN" in out and "trái dấu" in out, out
# 4. one seed only -> chưa kết luận
out = case({42: (77.0, 79.5)})
assert "CHƯA KẾT LUẬN: mới có 1 seed" in out and "-2.50" in out, out
# 5. a PACT run without isolation partner is skipped and reported
with tempfile.TemporaryDirectory() as tmp:
    write(tmp, "r1", "2026-11-26 10:00:00", {"pact_s42": run("pact", 42, 80.0), "iso_s42": run("iso", 42, 79.0),
                                             "pact_s7": run("pact", 7, 80.0)})
    out = compare(tmp)
    assert "seed 7: thiếu run isolation" in out, out
# 6. the newest run wins for the same (preset, kind, seed)
with tempfile.TemporaryDirectory() as tmp:
    write(tmp, "old", "2026-11-25 10:00:00", {"pact_s42": run("pact", 42, 50.0), "iso_s42": run("iso", 42, 79.0)})
    write(tmp, "new", "2026-11-26 10:00:00", {"pact_s42": run("pact", 42, 80.0)})
    out = compare(tmp)
    assert "+1.00" in out and "new" in out, out
# 7. different code builds are flagged
with tempfile.TemporaryDirectory() as tmp:
    write(tmp, "a", "2026-11-26 10:00:00", {"pact_s42": run("pact", 42, 80.0)}, "aaaa")
    write(tmp, "b", "2026-11-26 11:00:00", {"iso_s42": run("iso", 42, 79.0)}, "bbbb")
    out = compare(tmp)
    assert "CẢNH BÁO" in out and "aaaa" in out and "bbbb" in out, out
# 8. runs of other configs (ablations) are ignored; the real repo log loads without error
with tempfile.TemporaryDirectory() as tmp:
    ab = run("pact", 42, 70.0); ab["config_file"] = "configs/ablation/learned_gate.yaml"
    write(tmp, "r1", "2026-11-26 10:00:00", {"abl": ab})
    assert "Không tìm thấy run" in compare(tmp)
out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "compare_runs.py")], capture_output=True,
                     text=True, encoding="utf-8")
assert out.returncode == 0, out.stderr
print("compare_runs_test: ok")
