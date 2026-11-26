"""So sánh PACT với isolation baseline từ experiments/runs/*.json (do tools/log_run.py ghi) và áp
quy tắc quyết định của kế hoạch nghiên cứu.

    python tools/compare_runs.py                       # mọi preset có đủ cặp PACT / isolation
    python tools/compare_runs.py --preset full --write # chỉ preset full, ghi experiments/COMPARE.md

Một run được coi là:
    PACT       config_file chứa "mtil_order1"          (L_marg + L_pa bật)
    isolation  config_file chứa "isolation_baseline"   (bỏ L_marg và L_pa)
Chỉ so các run cùng preset, cùng thứ tự dataset, cùng seed. Mỗi (preset, loại, seed) lấy run mới nhất.

Chỉ số quyết định: Last của protocol `oracle_zeroshot` (đánh giá theo snapshot, biết task-id), hiệu PACT - isolation:
    trung bình các seed < +0.5 và cùng dấu ở mọi seed  -> ràng buộc không đóng góp vào Last
    trung bình các seed > +1.0 và cùng dấu ở mọi seed  -> ràng buộc có giá trị
    còn lại, hoặc chỉ 1 seed                           -> chưa kết luận (thêm seed hoặc chạy full)
Ngưỡng 0.5 / 1.0 do nhóm đề xuất, chưa có nguồn xác nhận; sửa bằng --low / --high.
Cũng báo Transfer (oracle_latest và router) so với zero-shot: Transfer zero-shot = trung bình điểm zero-shot của
các dataset từ thứ 2 trở đi trong thứ tự học.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kind_of(run):
    cfg = run.get("config_file") or ""
    if "isolation_baseline" in cfg:
        return "iso"
    if "mtil_order1" in cfg:
        return "pact"
    return None


def load_runs(runs_dir):
    """{(preset, tuple(order), kind, seed): (created, id, run)} giữ run mới nhất."""
    best = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("duplicate_of"):  # a re-run of the same config / seed (marked by hand), not an extra seed
            continue
        for name, run in (d.get("runs") or {}).items():
            kind = kind_of(run)
            seed = (run.get("key_config") or {}).get("seed")
            if kind is None or seed is None or not run.get("protocols"):
                continue
            key = (d.get("preset"), tuple(d.get("order") or ()), kind, int(seed))
            stamp = d.get("created") or ""
            if key not in best or stamp >= best[key][0]:
                best[key] = (stamp, d["id"], name, run, d.get("code_build"))
    return best


def proto(run, p, metric):
    s = (run.get("protocols") or {}).get(p)
    return None if not s else s.get(metric)


def zero_shot_transfer(run):
    zs = run.get("zero_shot")
    return sum(zs[1:]) / len(zs[1:]) if zs and len(zs) > 1 else None


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def signed(x):
    return "-" if x is None else f"{x:+.2f}"


def decide(diffs, low, high):
    if not diffs:
        return "không có cặp nào"
    mean = sum(diffs) / len(diffs)
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs) or all(abs(d) < 1e-9 for d in diffs)
    if len(diffs) < 2:
        return f"CHƯA KẾT LUẬN: mới có 1 seed (chênh {mean:+.2f}); cần thêm seed trước khi áp quy tắc"
    if mean < low and same_sign:
        return (f"Ràng buộc PACT không đóng góp vào Last (chênh trung bình {mean:+.2f} < +{low}, cùng dấu ở mọi seed): "
                "đi nhánh router / đánh giá trung thực (A1, A2, B1, B2)")
    if mean > high and same_sign:
        return (f"Ràng buộc PACT có giá trị (chênh trung bình {mean:+.2f} > +{high}, cùng dấu ở mọi seed): "
                "ưu tiên hợp nhất adapter (A3) và ablation nhánh conditional (B3)")
    return (f"CHƯA KẾT LUẬN: chênh trung bình {mean:+.2f} nằm giữa +{low} và +{high}, hoặc các seed trái dấu "
            f"({', '.join(f'{d:+.2f}' for d in diffs)}); thêm seed hoặc chạy full")


def compare(runs, preset_filter, low, high):
    groups = {}
    for (preset, order, kind, seed), val in runs.items():
        if preset_filter and preset != preset_filter:
            continue
        groups.setdefault((preset, order), {}).setdefault(seed, {})[kind] = val
    lines = []
    for (preset, order), seeds in sorted(groups.items(), key=lambda kv: str(kv[0])):
        pairs = {s: v for s, v in seeds.items() if "pact" in v and "iso" in v}
        lines += [f"## Preset `{preset}` ({len(order)} dataset: {', '.join(order)})", ""]
        for s, v in sorted(seeds.items()):
            if s not in pairs:
                missing = "isolation" if "iso" not in v else "PACT"
                lines.append(f"- seed {s}: thiếu run {missing}, bỏ qua")
        if not pairs:
            lines += ["", "Chưa có cặp PACT / isolation cùng seed.", ""]
            continue
        lines += ["", "| seed | PACT Last | iso Last | chênh Last | PACT Transfer (oracle_latest) | iso Transfer (oracle_latest) | zero-shot Transfer | PACT router Last | iso router Last |",
                  "|---|---|---|---|---|---|---|---|---|"]
        diffs, notes = [], []
        for s, v in sorted(pairs.items()):
            pa, io = v["pact"][3], v["iso"][3]
            lp, li = proto(pa, "oracle_zeroshot", "Last"), proto(io, "oracle_zeroshot", "Last")
            d = None if lp is None or li is None else lp - li
            if d is not None:
                diffs.append(d)
            lines.append(f"| {s} | {f(lp)} | {f(li)} | {signed(d)} | {f(proto(pa, 'oracle_latest', 'Transfer'))} | "
                         f"{f(proto(io, 'oracle_latest', 'Transfer'))} | {f(zero_shot_transfer(pa))} | "
                         f"{f(proto(pa, 'router', 'Last'))} | {f(proto(io, 'router', 'Last'))} |")
            for kind in ("pact", "iso"):
                cb = v[kind][4]
                notes.append((s, kind, v[kind][1], (cb or {}).get("id")))
            ca, cb2 = (v["pact"][4] or {}).get("code_hash"), (v["iso"][4] or {}).get("code_hash")
            if ca and cb2 and ca != cb2:
                lines.append(f"|  | CẢNH BÁO seed {s}: hai run dùng bản code khác nhau ({ca} và {cb2}), so sánh có thể không công bằng | | | | | | | |")
        lines += ["", "**Quy tắc quyết định:** " + decide(diffs, low, high), ""]
        # Transfer so với zero-shot theo run
        lines += ["Transfer so với zero-shot (điểm > 0 nghĩa là vượt zero-shot):", ""]
        for s, v in sorted(pairs.items()):
            zs = zero_shot_transfer(v["pact"][3])
            for kind, label in (("pact", "PACT"), ("iso", "isolation")):
                r = v[kind][3]
                cells = []
                for p in ("oracle_latest", "router", "latest"):
                    t = proto(r, p, "Transfer")
                    cells.append(f"{p} {f(t)} ({signed(None if t is None or zs is None else t - zs)})")
                lines.append(f"- seed {s}, {label}: " + "; ".join(cells) + f"; zero-shot {f(zs)}")
        lines += ["", "Nguồn: " + ", ".join(f"{k[1]} seed {k[0]}: {k[2]}" + (f" (code {k[3]})" if k[3] else "") for k in notes), ""]
    if not lines:
        lines = ["Không tìm thấy run PACT / isolation nào trong experiments/runs/."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "experiments", "runs"))
    ap.add_argument("--preset", default=None)
    ap.add_argument("--low", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=1.0)
    ap.add_argument("--write", action="store_true", help="ghi thêm vào experiments/COMPARE.md")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    text = compare(load_runs(a.runs_dir), a.preset, a.low, a.high)
    print(text)
    if a.write:
        path = os.path.join(ROOT, "experiments", "COMPARE.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# So sánh PACT và isolation\n\nSinh bởi `tools/compare_runs.py` từ `experiments/runs/*.json`.\n\n" + text)
        print(f"written: {path}")


if __name__ == "__main__":
    main()
