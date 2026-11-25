"""Ghi kết quả một lần chạy notebook (file zip) vào experiments/LOG.md rồi (tuỳ chọn) xoá zip.

    python tools/log_run.py output/pact_results_quick_20261125_0140.zip --note "isolation vs pact_paper, seed 42"
    python tools/log_run.py output/*.zip                  # ghi nhiều zip (zip đã ghi sẽ bỏ qua)
    python tools/log_run.py output/x.zip --delete         # ghi xong, kiểm tra lại rồi mới xoá zip

Mỗi zip tạo ra:
  experiments/LOG.md                  một mục: thời gian, phương pháp, bảng kết quả, ghi chú
  experiments/runs/<id>.json          số liệu gọn (ma trận độ chính xác, router, config chính), dùng lại được
  experiments/artifacts/<id>/<run>/evaluator.pt   cho analysis/router_offline.py (đã gitignore)
Zip chỉ bị xoá khi có --delete và cả ba thứ trên đã được ghi và đọc lại thành công.
Đọc thẳng từ zip, không giải nén ra đĩa.
"""
import argparse
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone

try:
    import yaml
except ImportError:  # chỉ mất phần trích config
    yaml = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
LOG = os.path.join(EXP, "LOG.md")
HEADER = ("# Nhật ký thí nghiệm\n\n"
          "Mỗi mục do `tools/log_run.py` ghi từ một file zip kết quả. Số liệu đầy đủ nằm ở `experiments/runs/<id>.json`.\n"
          "Các số của preset khác nhau (quick / mid / full) không so trực tiếp với nhau hay với paper.\n")
KEY_CFG = [("pact", "use_marg"), ("pact", "use_pa"), ("adapter", "gate_mode"), ("head", "type"),
           ("train", "lr"), ("train", "iterations"), ("train", "batch_size"), ("train", "label_smoothing"),
           ("pact", "tau"), ("pact", "pa_sigma_rel")]


def marker(rid):
    return f"<!-- run:{rid} -->"


def read_json(z, name):
    try:
        return json.loads(z.read(name))
    except KeyError:
        return None


def duration(z, name):
    """Số giây giữa dòng log đầu và cuối của một run (bỏ qua thời gian chờ giữa các lần resume lớn)."""
    try:
        times = [json.loads(l).get("time") for l in z.read(name).decode("utf-8").splitlines() if l.strip()]
    except KeyError:
        return None
    times = [t for t in times if t]
    if len(times) < 2:
        return None
    gaps = [b - a for a, b in zip(times, times[1:])]
    return sum(g for g in gaps if g < 3600)  # khoảng nghỉ > 1 giờ coi như dừng/resume


def key_config(z, run):
    if yaml is None:
        return {}
    try:
        cfg = yaml.safe_load(z.read(f"outputs/{run}/config.yaml"))
    except KeyError:
        return {}
    out = {f"{a}.{b}": cfg.get(a, {}).get(b) for a, b in KEY_CFG if isinstance(cfg.get(a), dict)}
    out["seed"] = cfg.get("seed")
    return out


def fmt_time(s):
    if s is None:
        return "?"
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m" if s >= 3600 else f"{s / 60:.0f}m"


def collect(path):
    rid = os.path.splitext(os.path.basename(path))[0]
    with zipfile.ZipFile(path) as z:
        info = read_json(z, "run_info.json") or {}
        runs = sorted({n.split("/")[1] for n in z.namelist() if n.startswith("outputs/") and n.count("/") >= 2})
        data = {"id": rid, "created": info.get("created"), "preset": info.get("preset"), "order": info.get("order"),
                "gpus_per_run": info.get("gpus_per_run"), "overrides": info.get("common_overrides"),
                "exit_codes": info.get("exit_codes"), "code_build": info.get("code_build"), "runs": {}}
        extra = {}
        for run in runs:
            res = read_json(z, f"outputs/{run}/results.json")
            if res is None:
                continue
            exp = next((e for e in info.get("experiments") or [] if e[0] == run), None)
            protocols = {p: v.get("summary") for p, v in (res.get("protocols") or {}).items()}
            summ = res.get("summary") or {}
            if not protocols and "Transfer" in summ:  # zip từ code cũ, chưa có nhiều protocol
                protocols = {"(cũ)": summ}
            if not protocols and "zero_shot_mean" in summ:
                protocols = {"zero-shot": {"Avg": summ["zero_shot_mean"]}}
            data["runs"][run] = {
                "config_file": exp[1] if exp else None, "extra_overrides": exp[2] if exp else None,
                "key_config": key_config(z, run), "seconds": duration(z, f"outputs/{run}/log.jsonl"),
                "main_protocol": res.get("main_protocol"), "summary": res.get("summary"),
                "protocols": protocols,
                "per_dataset": res.get("per_dataset"), "acc_matrix": res.get("acc_matrix"),
                "router": res.get("router"), "zero_shot": res.get("zero_shot")}
            if f"outputs/{run}/evaluator.pt" in z.namelist():
                extra[run] = z.read(f"outputs/{run}/evaluator.pt")
    return data, extra


def f2(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "-"


def render(d, note):
    lines = [marker(d["id"]), f"## {d['id']}", ""]
    meta = [f"- **Preset:** {d['preset']} ({len(d['order'] or [])} dataset: {', '.join(d['order'] or [])})",
            f"- **Tạo lúc:** {d['created']} | GPU mỗi run: {d['gpus_per_run']}"]
    cb = d.get("code_build")
    if cb:
        meta.append(f"- **Code:** `{cb.get('id')}` (commit {cb.get('commit')}, nhánh {cb.get('branch')}"
                    + (", có thay đổi chưa commit" if cb.get("dirty") else "") + ")")
    if d.get("exit_codes"):
        bad = {k: v for k, v in d["exit_codes"].items() if v}
        meta.append("- **Exit code:** " + ("tất cả 0" if not bad else f"LỖI {bad}"))
    if note:
        meta.append(f"- **Ghi chú:** {note}")
    lines += meta + ["", "### Phương pháp và thời gian", "",
                     "| run | config | override thêm | thay đổi chính | thời gian train+eval |", "|---|---|---|---|---|"]
    for name, r in d["runs"].items():
        kc = r["key_config"]
        key = ", ".join(f"{k}={kc[k]}" for k in ("pact.use_marg", "pact.use_pa", "adapter.gate_mode", "head.type",
                                                 "train.lr", "train.iterations", "seed") if k in kc)
        lines.append(f"| {name} | {r['config_file']} | {' '.join(r['extra_overrides'] or []) or '-'} | {key} | {fmt_time(r['seconds'])} |")
    lines += ["", "### Kết quả (Transfer / Avg / Last)", "",
              "| run | " + " | ".join(sorted({p for r in d["runs"].values() for p in r["protocols"]})) + " |",
              "|---|" + "---|" * len({p for r in d["runs"].values() for p in r["protocols"]})]
    protos = sorted({p for r in d["runs"].values() for p in r["protocols"]})
    for name, r in d["runs"].items():
        cells = []
        for p in protos:
            s = (r["protocols"] or {}).get(p)
            cells.append(f"{f2(s.get('Transfer'))} / {f2(s.get('Avg'))} / {f2(s.get('Last'))}" if s else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Last theo từng dataset (protocol chính):", ""]
    for name, r in d["runs"].items():
        last = ((r.get("per_dataset") or {}).get("Last")) or {}
        if last:
            lines.append(f"- **{name}** ({r.get('main_protocol')}): " + ", ".join(f"{k} {f2(v)}" for k, v in last.items()))
    checks = [f"{name}: {invariance(r.get('acc_matrix'))}" for name, r in d["runs"].items() if r.get("acc_matrix")]
    if checks:
        lines += ["", "Cột đã học bất biến theo Bảng 8 (protocol chính): " + "; ".join(checks)]
    lines += ["", "**Kết luận:** _(điền tay)_", ""]
    return "\n".join(lines) + "\n"


def invariance(acc):
    """Điểm của task j ở mọi hàng t >= j phải bằng nhau (snapshot đóng băng)."""
    try:
        bad = [(t, j) for t in range(len(acc)) for j in range(t + 1) if abs(acc[t][j] - acc[j][j]) > 1e-6]
    except (TypeError, IndexError):
        return "không kiểm tra được"
    return "có" if not bad else f"KHÔNG ({len(bad)} ô lệch, ví dụ hàng {bad[0][0]} cột {bad[0][1]})"


def write_all(d, extra, note):
    os.makedirs(os.path.join(EXP, "runs"), exist_ok=True)
    with open(os.path.join(EXP, "runs", d["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    for run, blob in extra.items():
        p = os.path.join(EXP, "artifacts", d["id"], run)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "evaluator.pt"), "wb") as f:
            f.write(blob)
    new = not os.path.exists(LOG)
    with open(LOG, "a", encoding="utf-8") as f:
        if new:
            f.write(HEADER)
        f.write("\n" + render(d, note))


def verified(d, extra):
    try:
        with open(LOG, encoding="utf-8") as f:
            ok = marker(d["id"]) in f.read()
        with open(os.path.join(EXP, "runs", d["id"] + ".json"), encoding="utf-8") as f:
            ok = ok and json.load(f)["id"] == d["id"]
        ok = ok and all(os.path.getsize(os.path.join(EXP, "artifacts", d["id"], r, "evaluator.pt")) > 0 for r in extra)
        return ok
    except (OSError, ValueError, KeyError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--note", default="")
    ap.add_argument("--delete", action="store_true", help="xoá zip sau khi đã ghi và kiểm tra lại")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(EXP, exist_ok=True)
    rc = 0
    for path in a.zips:
        try:
            d, extra = collect(path)
        except (OSError, zipfile.BadZipFile) as e:
            print(f"[lỗi] {path}: {e}")
            rc = 1
            continue
        if not d["runs"]:
            print(f"[bỏ qua] {path}: không có results.json nào (không xoá)")
            rc = 1
            continue
        done = os.path.exists(LOG) and marker(d["id"]) in open(LOG, encoding="utf-8").read()
        if done:
            print(f"[đã có] {d['id']}")
        else:
            write_all(d, extra, a.note)
            print(f"[ghi] {d['id']}: {', '.join(d['runs'])}")
        if a.delete:
            if verified(d, extra):
                os.remove(path)
                print(f"[xoá] {path}")
            else:
                print(f"[giữ] {path}: kiểm tra lại không đạt, không xoá")
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
