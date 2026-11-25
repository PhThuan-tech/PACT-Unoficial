"""Dựng bảng kết quả từ outputs của PACT (chỉ đọc, không đụng tới pact/).

    python tools/make_tables.py outputs/                  # mọi run trong outputs/*/results.json
    python tools/make_tables.py outputs/mtil_order1       # một run
    python tools/make_tables.py outputs/ --out reports/tables   # ghi thêm .md + .csv

Mỗi thư mục run cần results.json (config.yaml và log.jsonl nếu có thì dùng thêm).
Run zero-shot (adapter.type=none) được nhận ra tự động và cho ra bảng Z.

Bảng xuất ra:
  Z. Zero-shot so với Bảng 1 (lệch, số mẫu train/test, số lớp)      [run zero-shot]
  1. Điểm ngay sau khi học từng task (đường chéo acc_matrix) so với Bảng 8
  2. Transfer / Avg / Last của cả 4 protocol
  3. Quyết định của router (dataset được gửi tới đâu, sau mỗi task)

Run "sơ bộ" (preset quick: ít dataset, eval.max_samples, ít iteration) được gắn nhãn,
không so số tuyệt đối với paper.
"""

import argparse
import csv
import json
import os
import sys

# ---------------------------------------------------------------- mốc của paper
PAPER_ZS = {"Aircraft": 24.3, "Caltech101": 88.4, "CIFAR100": 68.2, "DTD": 44.6, "EuroSAT": 54.9,
            "Flowers": 71.0, "Food": 88.5, "MNIST": 59.4, "OxfordPet": 89.0, "StanfordCars": 64.7,
            "SUN397": 65.2}                                    # Bảng 1, CLIP zero-shot
PAPER_TABLE8 = {"Aircraft": 65.58, "Caltech101": 96.60, "CIFAR100": 91.21, "DTD": 80.53,
                "EuroSAT": 97.98, "Flowers": 98.59, "Food": 93.81, "MNIST": 99.53,
                "OxfordPet": 95.66, "StanfordCars": 89.90, "SUN397": 86.60}  # Bảng 8 (supp)
PAPER_PACT = {"Transfer": 71.76, "Avg": 81.05, "Last": 90.54}  # Bảng 1, PACT Order-I
PAPER_ZS_TRANSFER = 69.4                                        # zero-shot Transfer, Order-I
FULL_ORDER_I = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food",
                "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
EXPECTED_SIZES = {  # (train, test), giống tools/kaggle_data.py
    "Aircraft": (3334, 3333), "Caltech101": (6941, 1736), "CIFAR100": (50000, 10000),
    "DTD": (1880, 1880), "EuroSAT": (21600, 5400), "Flowers": (1020, 6149),
    "Food": (75750, 25250), "MNIST": (60000, 10000), "OxfordPet": (3680, 3669),
    "StanfordCars": (8144, 8041), "SUN397": (87003, 21751)}
PROTOCOL_ORDER = ["router", "oracle_zeroshot", "oracle_latest", "latest"]
PROTOCOL_NOTE = {
    "router": "không biết task-ID",
    "oracle_zeroshot": "Bảng 8; Transfer ≈ zero-shot theo định nghĩa",
    "oracle_latest": "dùng để xét Transfer > zero-shot",
    "latest": "không snapshot: xét quên + Transfer",
}


# ---------------------------------------------------------------- đọc dữ liệu
def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    cfg = {}
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:  # yaml thiếu hoặc file hỏng: vẫn dựng bảng được
            print(f"[warn] không đọc được {cfg_path}: {exc}", file=sys.stderr)
    sizes = {}
    log_path = os.path.join(run_dir, "log.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "dataset" in rec and "train" in rec and "test" in rec:
                    sizes[rec["dataset"]] = (rec["train"], rec["test"], rec.get("classes"))
    return {"name": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir,
            "res": res, "cfg": cfg, "sizes": sizes}


def find_runs(paths):
    runs = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "results.json")):
            runs.append(p)
        elif os.path.isdir(p):
            for d in sorted(os.listdir(p)):
                if os.path.isfile(os.path.join(p, d, "results.json")):
                    runs.append(os.path.join(p, d))
        else:
            print(f"[warn] bỏ qua {p}: không có results.json", file=sys.stderr)
    return [load_run(r) for r in runs]


def preliminary_reasons(run):
    """Lý do run này chỉ là sơ bộ (rỗng = thiết lập đầy đủ)."""
    order, cfg = run["res"].get("order", []), run["cfg"]
    reasons = []
    if order != FULL_ORDER_I:
        reasons.append(f"{len(order)} dataset (không phải Order-I đủ 11)")
    max_samples = cfg.get("eval", {}).get("max_samples")
    if max_samples:
        reasons.append(f"eval.max_samples={max_samples}")
    iters = cfg.get("train", {}).get("iterations")
    if iters and iters != 3000 and "zero_shot_by_template" not in run["res"]:
        reasons.append(f"{iters} iteration (paper: 3000)")
    return reasons


# ---------------------------------------------------------------- tiện ích bảng
def fmt(x, nd=2, sign=False):
    if x is None:
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Report:
    def __init__(self, out_dir=None):
        self.parts, self.out_dir = [], out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def add(self, text):
        self.parts.append(text)

    def table(self, title, header, rows, csv_name=None, note=None):
        self.add(f"\n#### {title}\n")
        self.add(md_table(header, rows))
        if note:
            self.add(f"\n{note}")
        if self.out_dir and csv_name:
            with open(os.path.join(self.out_dir, csv_name), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

    def text(self):
        return "\n".join(self.parts) + "\n"


# ---------------------------------------------------------------- bảng Z: zero-shot
def table_zero_shot(rep, run):
    res = run["res"]
    order, by = res["order"], res["zero_shot_by_template"]
    main = res["zero_shot"]
    rows, flagged = [], []
    for i, n in enumerate(order):
        paper = PAPER_ZS.get(n)
        delta = main[i] - paper if paper is not None else None
        tr, te, cls = run["sizes"].get(n, (None, None, None))
        exp = EXPECTED_SIZES.get(n)
        size_note = "" if exp is None or tr is None or (tr, te) == exp else f" (kỳ vọng {exp[0]}/{exp[1]})"
        status = "OK"
        if delta is not None and abs(delta) > 1.5:
            status = "LỆCH > 1.5"
        elif delta is not None and abs(delta) > 1.0:
            status = "lệch 1.0–1.5: ghi nguyên nhân"
        if status != "OK":
            flagged.append(n)
        rows.append([n, fmt(paper, 1), fmt(by["first"][i]), fmt(by["ensemble"][i]),
                     fmt(delta, sign=True), f"{tr}/{te}{size_note}" if tr is not None else "–",
                     cls if cls is not None else "–", status])
    mean = sum(main) / len(main)
    paper_mean = [PAPER_ZS[n] for n in order if n in PAPER_ZS]
    paper_mean = sum(paper_mean) / len(paper_mean) if paper_mean else None
    rows.append(["**Trung bình**", fmt(paper_mean, 2), fmt(sum(by["first"]) / len(order)),
                 fmt(sum(by["ensemble"]) / len(order)),
                 fmt(mean - paper_mean if paper_mean else None, sign=True), "", "", ""])
    missing = [n for n in FULL_ORDER_I if n not in order]
    note = (f"Cột lệch tính trên template dùng khi eval ({run['cfg'].get('eval', {}).get('template_mode', 'ensemble')}). "
            f"Cần ghi nguyên nhân: {', '.join(flagged) if flagged else 'không có'}. ")
    if missing:
        note += f"**Chưa chạy zero-shot:** {', '.join(missing)}."
    rep.table(f"Bảng Z – Zero-shot so với Bảng 1 ({run['name']})",
              ["Dataset", "Paper", "first", "ensemble", "Δ", "train/test", "Lớp", "Trạng thái"],
              rows, f"{run['name']}_zero_shot.csv", note)


# ---------------------------------------------------------------- bảng 1: đường chéo vs Bảng 8
def table_diag(rep, run, prelim):
    res, order = run["res"], run["res"]["order"]
    protos = res.get("protocols", {})
    oz = protos.get("oracle_zeroshot", {}).get("acc_matrix")
    ol = protos.get("oracle_latest", {}).get("acc_matrix")
    lt = protos.get("latest", {}).get("acc_matrix")
    if not oz:
        rep.add("\n_(không có protocol oracle_zeroshot: bỏ qua bảng 1)_")
        return
    steps = len(oz)
    rows = []
    for j, n in enumerate(order):
        if j >= steps:
            rows.append([n, "chưa học", "", "", "", ""])
            continue
        d = oz[j][j]
        paper = PAPER_TABLE8.get(n)
        last_latest = lt[-1][j] if lt else None
        rows.append([n, fmt(d), fmt(ol[j][j]) if ol else "–", fmt(paper),
                     fmt(d - paper if paper is not None and not prelim else None, sign=True),
                     fmt(last_latest)])
    stable = all(oz[i][j] == oz[j][j] for i in range(steps) for j in range(min(i + 1, len(order))))
    note = ("Kiểm tra cài đặt: cột đã học của oracle_zeroshot bất biến = "
            f"**{'OK' if stable else 'KHÔNG'}** (chỉ chứng minh snapshot đúng, không chứng minh không quên). "
            "Cột 'latest (hàng cuối)' cho thấy mô hình mới nhất còn giữ được bao nhiêu.")
    if prelim:
        note += " Run sơ bộ: không tính Δ với Bảng 8."
    rep.table(f"Bảng 1 – Điểm ngay sau khi học từng task ({run['name']})",
              ["Dataset", "oracle_zeroshot", "oracle_latest", "Bảng 8", "Δ", "latest (hàng cuối)"],
              rows, f"{run['name']}_table8.csv", note)


# ---------------------------------------------------------------- bảng 2: Transfer/Avg/Last
def table_summary(rep, run, prelim):
    res, order = run["res"], run["res"]["order"]
    protos = res.get("protocols") or {res.get("main_protocol", "main"): res}
    zs = res.get("zero_shot")
    steps = len(next(iter(protos.values()))["acc_matrix"])
    done = steps >= len(order)
    # Transfer chỉ tính trên dataset 2..min(steps+1, T): mốc zero-shot lấy đúng các dataset đó
    span = zs[1:min(steps, len(order) - 1) + 1] if zs else []
    zs_transfer = sum(span) / len(span) if span else None
    ref = PAPER_ZS_TRANSFER if (not prelim and done) else zs_transfer
    names = [p for p in PROTOCOL_ORDER if p in protos] + [p for p in protos if p not in PROTOCOL_ORDER]
    rows = []
    for p in names:
        s = protos[p]["summary"]
        if p in ("oracle_latest", "latest") and ref is not None:
            verdict = "vượt" if s["Transfer"] > ref else "KHÔNG vượt"
            verdict += f" ({fmt(s['Transfer'] - ref, sign=True)})"
        else:
            verdict = "không dùng làm tiêu chí"
        row = [p, fmt(s["Transfer"]), fmt(s["Avg"]), fmt(s["Last"]), verdict, PROTOCOL_NOTE.get(p, "")]
        if steps < len(order):
            row[-1] += f" – mới xong {steps}/{len(order)} task"
        rows.append(row)
    if not prelim:
        rows.append(["*paper (PACT)*", fmt(PAPER_PACT["Transfer"]), fmt(PAPER_PACT["Avg"]),
                     fmt(PAPER_PACT["Last"]), "", "Bảng 1"])
    if not prelim and done:
        src = "(paper, Order-I)."
    elif prelim:
        src = "(trung bình zero-shot của dataset 2..T trong chính run này, vì run sơ bộ)."
    else:
        src = (f"(zero-shot đo trong run, trên đúng {len(span)} dataset mà Transfer đang tính, "
               "vì run chưa xong; chưa so được với 69.4).")
    note = f"Mốc Transfer zero-shot: {fmt(ref)} " + src
    if zs_transfer is not None and not prelim and done:
        note += f" Zero-shot Transfer đo được trong run: {fmt(zs_transfer)}."
    rep.table(f"Bảng 2 – Transfer / Avg / Last theo protocol ({run['name']})",
              ["Protocol", "Transfer", "Avg", "Last", "Transfer > zero-shot?", "Ghi chú"],
              rows, f"{run['name']}_protocols.csv", note)


# ---------------------------------------------------------------- bảng 3: router
def _dest(entry):
    parts = sorted(entry.get("to", {}).items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} {v:.0%}" if v < 1 else k for k, v in parts) or "–"


def table_router(rep, run):
    res, order = run["res"], run["res"]["order"]
    log = res.get("router") or []
    if not log:
        rep.add("\n_(không có router log)_")
        return
    rows, warnings, seen = [], [], set()

    def warn(key, msg):  # chỉ báo lần đầu mỗi vấn đề xuất hiện
        if key not in seen:
            seen.add(key)
            warnings.append(msg)

    for t, routing in enumerate(log):
        learned = set(order[:t + 1])
        cells = []
        for n in order:
            e = routing.get(n)
            if e is None:
                cells.append("–")
                continue
            cell = "→ " + _dest(e)
            if n in learned and e.get("correct", 0) < 1:
                cell += f" ⚠ đúng {e.get('correct', 0):.0%}"
                warn((n, "learned"), f"từ sau task {t + 1}: {n} (đã học) chỉ {e.get('correct', 0):.0%} "
                     "batch về đúng task")
            stray = {k: v for k, v in e.get("to", {}).items() if k != "CLIP"}
            if n not in learned and stray:
                cell += " ⚠"
                warn((n, "stray", t), f"sau task {t + 1}: {n} (chưa học) bị gửi tới "
                     + ", ".join(f"{k} {v:.0%}" for k, v in stray.items()) + " thay vì CLIP")
            cells.append(cell)
        rows.append([f"sau {order[t]}" if t < len(order) else f"hàng {t + 1}"] + cells)
    note = ("Mỗi ô: tỷ lệ batch test được gửi tới snapshot nào (CLIP = mô hình đóng băng). "
            "⚠ = dataset đã học không về đúng task, hoặc dataset chưa học bị gửi nhầm vào một task.")
    if warnings:
        note += "\n\nCảnh báo:\n" + "\n".join(f"- {w}" for w in warnings)
    rep.table(f"Bảng 3 – Quyết định của router ({run['name']})",
              ["Sau task"] + order, rows, f"{run['name']}_router.csv", note)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="thư mục run (có results.json) hoặc thư mục outputs/")
    ap.add_argument("--out", default=None, help="ghi tables.md và các .csv vào thư mục này")
    args = ap.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        sys.exit("không tìm thấy results.json nào")
    rep = Report(args.out)
    for run in runs:
        res = run["res"]
        is_zs = "zero_shot_by_template" in res
        prelim = preliminary_reasons(run)
        cfg = run["cfg"]
        lr = cfg.get("train", {}).get("lr")
        meta = [f"seed={cfg.get('seed')}" if cfg else None, f"lr={lr}" if lr and not is_zs else None,
                f"head={cfg.get('head', {}).get('type')}" if cfg and not is_zs else None,
                f"gate={cfg.get('adapter', {}).get('gate_mode')}" if cfg and not is_zs else None]
        rep.add(f"\n## {run['name']}" + ("  — SƠ BỘ" if prelim and not is_zs else ""))
        rep.add(" · ".join(m for m in meta if m) or "(không có config.yaml)")
        if prelim and not is_zs:
            rep.add(f"\n> Sơ bộ: {'; '.join(prelim)}. Chỉ xem cơ chế và thứ tự, không so số với paper.")
        if is_zs:
            table_zero_shot(rep, run)
            continue
        table_diag(rep, run, bool(prelim))
        table_summary(rep, run, bool(prelim))
        table_router(rep, run)

    text = rep.text()
    print(text)
    if args.out:
        with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[đã ghi] {args.out}/tables.md và các file .csv", file=sys.stderr)


if __name__ == "__main__":
    main()
