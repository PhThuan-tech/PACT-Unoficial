"""Generate kaggle/pact_demo.ipynb:  python kaggle/make_notebook.py kaggle/pact_demo.ipynb"""
import json
import os
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r"""
# PACT (unofficial) – demo trên Kaggle, chạy offline, nhiều GPU

Notebook này chạy lại repo PACT (CVPR 2026, bản tự cài đặt) trên Kaggle: **Internet tắt**, GPU 4×L4 (hoặc 2×T4).

**Input cần gắn (Add Input):**

| Input | Nội dung |
|---|---|
| `pact-code` (bạn tự upload) | file `pact-code.zip` (code + wheel, ~2 MB), tạo bằng `python tools/prepare_upload.py`. Upload lại mỗi khi code đổi |
| `pact-kit` hoặc `pact-clip` | trọng số CLIP (upload một lần). Dataset `pact-kit` cũ dùng được; code cũ bên trong sẽ bị bỏ qua vì notebook chọn bản code mới nhất |
| Dataset ảnh | Một bản Kaggle cho mỗi dataset cần dùng (gợi ý ở bảng dưới). Script tự dò file đặc trưng, nên **không cần đúng cấu trúc thư mục** |

Preset `quick` chỉ cần **Aircraft, DTD, EuroSAT, OxfordPet**. Preset `mid` cần thêm **Caltech101, CIFAR100, MNIST, StanfordCars** (8 dataset, 2000 iter, bỏ Flowers vì chưa có bản Kaggle đúng cấu trúc). Preset `full` cần đủ 11 dataset MTIL.

**Cấu hình mặc định bám theo supplementary của paper** (xem `reports/Giải mã chi tiết PACT.md`):
- Đánh giá theo snapshot của từng task (Bảng 8: độ chính xác task đã học không đổi).
- Gate cố định tăng theo tầng, tầng cuối 0.5.
- Hai head học được cho mỗi task.
- α̃ bật bằng công tắc cứng (Eq. 40–44).
- Label smoothing 0.2, eval bằng ensemble template, fp32.

Mỗi run tính cùng lúc 4 giao thức đánh giá: `router`, `oracle_latest`, `oracle_zeroshot` và `latest` (cách cũ).

| Dataset | Dataset trên Kaggle (đã kiểm 25/11, copy tên rồi Add Input) |
|---|---|
| Aircraft | `howardkhh/fgvc-aircraft` |
| Caltech101 | `imbikramsaha/caltech-101` |
| CIFAR100 | `fedesoriano/cifar100` |
| DTD | `jmexpert/describable-textures-dataset-dtd` |
| EuroSAT | `apollo2506/eurosat-dataset` |
| Flowers | `muhammadtalib/flowers-102` |
| Food | `dansbecker/food-101` |
| MNIST | `hojjatk/mnist-dataset` |
| OxfordPet | `julinmaloof/the-oxfordiiit-pet-dataset` |
| StanfordCars | `rickyyyyyyy/torchvision-stanford-cars` |
| SUN397 | `hiuphmnhtrung/sun397` |
| TinyImageNet (CIL) | `akash2sharma/tiny-imagenet` (chưa kiểm) |

Code: `lnhquckhnh/pact-code`, CLIP: `lnhquckhnh/pact-clip`

**Chạy dài (full):** dùng *Save Version → Save & Run All*. Phiên Kaggle có giới hạn thời gian; checkpoint được lưu sau mỗi task (xem ô cuối để chạy tiếp).
""")

code(r"""
# ===================== CẤU HÌNH =====================
PRESET = "full"          # "dry": 11 dataset, 20 iter (dry-run) | "quick": 4 dataset, 1000 iter | "mid": 8 dataset, 2000 iter | "full": thiết lập paper (11 dataset, 3000 iter)
GPUS_PER_RUN = 2        # số GPU cho MỖI cấu hình (1, 2 hoặc 4); GPUS_PER_RUN x số cấu hình <= số GPU. Cả nhóm dùng cùng một giá trị
ALL_EXPERIMENTS = [     # (tên, file config, override thêm). Mỗi run độc lập (chạy ở phiên Kaggle riêng hoặc song song), dùng GPUS_PER_RUN GPU
    ("pact_s42",   "configs/mtil_order1.yaml",                 ["seed=42"]),     # PACT đầy đủ
    ("iso_s42",    "configs/analysis/isolation_baseline.yaml", ["seed=42"]),     # bỏ L_marg và L_pa, vẫn đánh giá theo snapshot
    ("pact_s2026", "configs/mtil_order1.yaml",                 ["seed=2026"]),
    ("iso_s2026",  "configs/analysis/isolation_baseline.yaml", ["seed=2026"]),
]
RUN_ONLY = ["pact_s42"]  # MỖI NGƯỜI ĐỔI DÒNG NÀY: pact_s42 (Phát), iso_s42, pact_s2026, iso_s2026; None = chạy hết (cần đủ GPU)
EXPERIMENTS = [e for e in ALL_EXPERIMENTS if RUN_ONLY is None or e[0] in RUN_ONLY]
assert EXPERIMENTS, f"RUN_ONLY={RUN_ONLY} không khớp tên run nào"
RESUME_FROM = None      # chạy tiếp: thư mục outputs/ của version trước (đã gắn làm input)
RUN_ZERO_SHOT_CHECK = False   # zero-shot 11 dataset đã kiểm tra (Khánh, 25/11); đặt True nếu đổi dữ liệu hoặc CLIP

WORK = "/kaggle/working"
REPO = f"{WORK}/PACT"
DATA_ROOT = f"{WORK}/data"
OUT = f"{WORK}/outputs"

PRESETS = {
    "quick": dict(order=["Aircraft", "DTD", "EuroSAT", "OxfordPet"],
                  extra=["train.iterations=1000", "eval.max_samples=1000", "train.log_every=25"]),
    # mid: 8 dataset nhỏ/vừa có bản Kaggle dò được, bỏ Flowers (chưa có bản đúng cấu trúc), Food và SUN397 (lớn). Giữ thứ tự tương đối của Order-I.
    # 8 task x 2000 iter = 16k iter, khoảng 48% chi phí của full (33k iter), khoảng 3 giờ GPU mỗi run (ước lượng, chưa đo).
    "mid":   dict(order=["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "MNIST", "OxfordPet", "StanfordCars"],
                  extra=["train.iterations=2000", "eval.max_samples=2000", "train.log_every=50"]),
    "full":  dict(order=["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food",
                         "MNIST", "OxfordPet", "StanfordCars", "SUN397"],
                  extra=[]),
    # dry: dry-run trước full. Cùng 11 dataset và cùng đường đi code, nhưng 20 iter/task và 200 ảnh test, vài phút.
    # Kiểm tra: đọc đủ 11 dataset, ghi checkpoint / results.json, --resume (chạy hai lần, lần hai phải báo "resumed after 11 tasks").
    # Kết quả KHÔNG có giá trị khoa học; đặt RUN_ZERO_SHOT_CHECK = False khi dry-run (đã có kết quả zero-shot).
    "dry":   dict(order=["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food",
                         "MNIST", "OxfordPet", "StanfordCars", "SUN397"],
                  extra=["train.iterations=20", "eval.max_samples=200", "train.log_every=5"]),
}
ORDER = PRESETS[PRESET]["order"]
print("preset:", PRESET, "| datasets:", ORDER)
""")

code(r"""
# ===================== 1. Tìm pact-kit, cài wheel offline, giải nén repo =====================
import os, sys, json, time, shutil, zipfile, subprocess, importlib

def walk_limited(root, max_depth=6, max_entries=300):
    # os.walk giới hạn độ sâu, bỏ qua thư mục ảnh lớn (không quét hàng trăm nghìn file)
    base = root.rstrip("/").count("/")
    for path, dirs, files in os.walk(root):
        if path.count("/") - base >= max_depth or len(dirs) + len(files) > max_entries:
            dirs[:] = []
        yield path, dirs, files


CLIP_NAME = "ViT-B-16-quickgelu-openai.pt"
KIT_WHEELS = ("open_clip_torch-", "ftfy-")   # chỉ cài đúng 2 wheel này, bỏ qua wheel của input khác
KIT_ZIPS = ("pact-kit.zip", "pact-code.zip", "pact-clip.zip")

def scan(roots):
    # CLIP: lấy ở bất kỳ input nào. Code: mọi thư mục PACT/ (cũ trong pact-kit, mới trong pact-code)
    found = dict(clip=None, codes=[], zips=[])
    for root in roots:
        if not os.path.isdir(root): continue
        for path, dirs, files in walk_limited(root):
            if found["clip"] is None and CLIP_NAME in files:
                found["clip"] = os.path.join(path, CLIP_NAME)
            if os.path.isfile(os.path.join(path, "pact", "engine.py")):
                found["codes"].append(path)
            found["zips"] += [os.path.join(path, f) for f in files if f in KIT_ZIPS]
    return found

def build_id(code_dir):
    try: return json.load(open(os.path.join(code_dir, "BUILD_INFO.json"))).get("id", "?")
    except Exception: return "?"

def build_time(code_dir):   # bản code mới nhất thắng; bản cũ không có BUILD_INFO.json
    try: return json.load(open(os.path.join(code_dir, "BUILD_INFO.json")))["built"]
    except Exception: return ""

ROOTS = ["/kaggle/input", f"{WORK}/_kits"]
found = scan(ROOTS)
extracted = False
for zp in found["zips"]:   # Kaggle giữ nguyên file zip: pact-code (nhỏ) luôn giải nén,
    name = os.path.basename(zp)[:-4]   # zip lớn (pact-kit, pact-clip) chỉ khi còn thiếu CLIP/code
    dest = f"{WORK}/_kits/{name}"
    if os.path.isdir(dest) or not (name == "pact-code" or not found["clip"] or not found["codes"]):
        continue
    print("giải nén", zp, "->", dest)
    with zipfile.ZipFile(zp) as z: z.extractall(dest)
    extracted = True
if extracted:
    found = scan(ROOTS)
assert found["clip"], "Không thấy trọng số CLIP: gắn dataset pact-kit (hoặc pact-clip)"
assert found["codes"], "Không thấy code: gắn dataset pact-code (hoặc pact-kit)"
clip_w = found["clip"]
clip_ref = os.path.join(os.path.dirname(clip_w), "reference.pt")
code_src = max(found["codes"], key=build_time)
CODE_ROOT = os.path.dirname(code_src)          # dataset chứa code; wheel lấy ở đây trước

def find_wheels(roots):
    wheels, wheel_dirs = [], []
    for root in roots:
        for path, dirs, files in walk_limited(root):
            if any(os.path.isfile(os.path.join(path, p, "__init__.py")) for p in ("open_clip", "ftfy")):
                wheel_dirs.append(path)        # Kaggle đã giải nén luôn file .whl
            wheels += [os.path.join(path, f) for f in files
                       if f.endswith(".whl") and f.startswith(KIT_WHEELS)]
    return wheels, wheel_dirs
wheels, wheel_dirs = find_wheels([CODE_ROOT])
if not (wheels or wheel_dirs):
    wheels, wheel_dirs = find_wheels(ROOTS)
wheels = list({os.path.basename(w): w for w in wheels}.values())   # mỗi file 1 lần

print("code     :", code_src, "| build:", build_time(code_src) or "(bản cũ)", "| id:", build_id(code_src),
      f"| {len(found['codes'])} bản tìm thấy")
print("wheels   :", [os.path.basename(w) for w in wheels] or wheel_dirs)
print("clip     :", clip_w, "| reference:", os.path.exists(clip_ref))
assert os.path.exists(clip_ref), "thiếu clip/reference.pt cạnh trọng số CLIP"

if os.path.exists(REPO): shutil.rmtree(REPO)
shutil.copytree(code_src, REPO)
os.chdir(REPO); sys.path.insert(0, REPO)

if wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q",
                    *wheels], check=True)
elif wheel_dirs:   # wheel đã bị giải nén thành thư mục: dùng trực tiếp qua PYTHONPATH
    sys.path[:0] = wheel_dirs
    os.environ["PYTHONPATH"] = os.pathsep.join(wheel_dirs + [os.environ.get("PYTHONPATH", "")])
importlib.invalidate_caches()
missing = []
for mod in ["torch", "torchvision", "open_clip", "ftfy", "wcwidth", "regex", "timm", "huggingface_hub", "scipy", "yaml"]:
    try:
        m = importlib.import_module(mod); print(f"  {mod:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        missing.append(mod); print(f"  {mod:16s} THIẾU: {e}")
assert not missing, f"Thiếu thư viện {missing}: thêm wheel của chúng vào pact-kit/wheels"
""")

code(r"""
# ===================== 2. Phần cứng =====================
import torch
n_gpu = torch.cuda.device_count()
print("GPU:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
print("bf16:", n_gpu > 0 and torch.cuda.is_bf16_supported(), "| CPU:", os.cpu_count(),
      "| disk trống (GB):", round(shutil.disk_usage(WORK).free / 2**30, 1))
n_procs = GPUS_PER_RUN * len(EXPERIMENTS)
assert n_gpu >= n_procs, f"Cần {n_procs} GPU (GPUS_PER_RUN x số cấu hình), máy có {n_gpu}"
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) // n_procs))
print("num_workers mỗi process:", NUM_WORKERS)
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"])
""")

code(r"""
# ===================== 3. Kiểm tra trọng số CLIP (đúng QuickGELU, file không hỏng) =====================
r = subprocess.run([sys.executable, "-W", "ignore", "tools/clip_offline.py", "verify",
                    "--weights", clip_w, "--reference", clip_ref], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
assert r.returncode == 0, "Trọng số CLIP không khớp reference"
""")

code(r"""
# ===================== 4. Dò dataset trong /kaggle/input, dựng data_root (symlink) =====================
# Nếu một dataset không được nhận ra, thêm: "--map", "Tên=/kaggle/input/<thư mục>" để chỉ chỗ tìm.
r = subprocess.run([sys.executable, "tools/kaggle_data.py", "--input", "/kaggle/input",
                    "--data-root", DATA_ROOT, "--datasets", *ORDER, "--check"],
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
assert r.returncode == 0, "Có dataset thiếu/lỗi (xem ở trên)"
""")

code(r"""
# ===================== 5. Tham số chung cho mọi run =====================
COMMON = [f"data_root={DATA_ROOT}", "download=false", f"num_workers={NUM_WORKERS}",
          "model.arch=ViT-B-16-quickgelu", f"model.pretrained={clip_w}",
          "mtil.order=[" + ",".join(ORDER) + "]", *PRESETS[PRESET]["extra"]]
print("\n".join(COMMON))
""")

code(r"""
# ===================== 6. Zero-shot: so với dòng CLIP zero-shot ở Bảng 1 =====================
# Kiểm tra dữ liệu, tên lớp, template, trọng số trước khi train (rẻ, 1 GPU, toàn bộ tập test).
# Đo cả 2 cách dựng text classifier: template đầu ("first") và ensemble mọi template (mặc định eval).
PAPER_ZS = {"Aircraft": 24.3, "Caltech101": 88.4, "CIFAR100": 68.2, "DTD": 44.6, "EuroSAT": 54.9,
            "Flowers": 71.0, "Food": 88.5, "MNIST": 59.4, "OxfordPet": 89.0, "StanfordCars": 64.7,
            "SUN397": 65.2}
if RUN_ZERO_SHOT_CHECK:
    zs_out = f"{OUT}/zero_shot"
    cmd = [sys.executable, "-m", "pact.run_mtil", "--set", *COMMON, "adapter.type=none",
           f"output_dir={zs_out}", "eval.max_samples=null"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    else:
        res = json.load(open(f"{zs_out}/results.json"))
        by = res["zero_shot_by_template"]
        print(f"{'dataset':14s}{'first':>8s}{'ensemble':>10s}{'paper':>8s}{'lệch(ens)':>11s}")
        for i, n in enumerate(res["order"]):
            a1, a2 = by["first"][i], by["ensemble"][i]
            flag = "  <-- lệch > 1.5 điểm" if abs(a2 - PAPER_ZS[n]) > 1.5 else ""
            print(f"{n:14s}{a1:8.2f}{a2:10.2f}{PAPER_ZS[n]:8.1f}{a2 - PAPER_ZS[n]:+11.2f}{flag}")
        m1 = sum(abs(by["first"][i] - PAPER_ZS[n]) for i, n in enumerate(res["order"]))
        m2 = sum(abs(by["ensemble"][i] - PAPER_ZS[n]) for i, n in enumerate(res["order"]))
        print(f"tổng |lệch|: first={m1:.2f}  ensemble={m2:.2f}  -> khớp paper hơn: "
              f"{'ensemble' if m2 <= m1 else 'first (cân nhắc thêm eval.template_mode=first vào COMMON)'}")
""")

code(r"""
# ===================== 7. Khởi chạy các cấu hình song song =====================
os.makedirs(OUT, exist_ok=True)
if RESUME_FROM:   # chép checkpoint của version trước để chạy tiếp
    for name, _, _ in EXPERIMENTS:
        src = os.path.join(RESUME_FROM, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(OUT, name), dirs_exist_ok=True); print("resume:", name)

PROCS = {}
for i, (name, config, extra) in enumerate(EXPERIMENTS):
    gpus = ",".join(str(g) for g in range(i * GPUS_PER_RUN, (i + 1) * GPUS_PER_RUN))
    run_dir = os.path.join(OUT, name); os.makedirs(run_dir, exist_ok=True)
    launcher = ([sys.executable, "-m"] if GPUS_PER_RUN == 1 else
                ["torchrun", "--standalone", f"--nproc_per_node={GPUS_PER_RUN}", "-m"])
    cmd = [*launcher, "pact.run_mtil", "--config", config, "--resume",
           "--set", *COMMON, f"output_dir={run_dir}", *extra]
    stdout = open(os.path.join(run_dir, "stdout.log"), "a")
    PROCS[name] = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, cwd=REPO,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus})
    print(f"[{name}] GPU {gpus} | log: {run_dir}/stdout.log")
""")

code(r"""
# ===================== 8. Theo dõi (ô này chặn tới khi xong: cần cho Save & Run All) =====================
def last_record(name):
    path = os.path.join(OUT, name, "log.jsonl")
    if not os.path.exists(path): return {}
    with open(path) as f: lines = f.readlines()
    return json.loads(lines[-1]) if lines else {}

def tail(name, n=40):
    with open(os.path.join(OUT, name, "stdout.log")) as f: return "".join(f.readlines()[-n:])

ITERS = next((int(o.split("=")[1]) for o in PRESETS[PRESET]["extra"] if o.startswith("train.iterations=")), 3000)

def progress(name):
    # tiến độ huấn luyện 0..1 theo (task đã xong x iter + iter hiện tại); eval không được tính
    path = os.path.join(OUT, name, "log.jsonl")
    if not os.path.exists(path): return 0.0
    with open(path) as f: lines = f.readlines()
    for line in reversed(lines):
        rec = json.loads(line)
        if "it" in rec and rec.get("task") in ORDER:
            return (ORDER.index(rec["task"]) * ITERS + rec["it"] + 1) / (len(ORDER) * ITERS)
        if "after_task" in rec and rec["after_task"] == ORDER[-1]:
            return 1.0
    return 0.0

INTERVAL = 120   # giây
KEYS = ("task", "it", "L_cond", "phi", "alpha_tilde", "after_task", "mean")
T0, P0 = time.time(), {n: progress(n) for n in PROCS}   # mốc đầu (nếu resume thì không tính phần đã xong)
while True:
    print(time.strftime("%H:%M:%S"), "-" * 60)
    for name, p in PROCS.items():
        rec = last_record(name)
        brief = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k in KEYS}
        state = "đang chạy" if p.poll() is None else f"xong (exit {p.returncode})"
        prog = progress(name)
        eta = ""
        if p.poll() is None and prog > P0[name]:
            remain = (1 - prog) * (time.time() - T0) / (prog - P0[name])
            eta = f" | ETA ~{remain / 60:.0f} phút (chưa tính eval)"
        print(f"  [{name}] {state} {prog:.0%}{eta} {brief}")
    if all(p.poll() is not None for p in PROCS.values()): break
    time.sleep(INTERVAL)
for name, p in PROCS.items():
    if p.returncode: print(f"\n==== {name} LỖI (exit {p.returncode}) ====\n{tail(name)}")
""")

code(r"""
# ===================== 8b. Đặc trưng CLIP đóng băng cho router theo từng ảnh (analysis/router_persample.py) =====================
# Chạy sau khi các run xong, 1 GPU, vài phút. Ghi outputs/router_features/router_features.npz (float16, vài chục MB),
# nằm trong zip kết quả. Phân tích (CPU, vài giây):  python analysis/router_persample.py analyze <file>.npz
EXTRACT_ROUTER_FEATURES = True
if EXTRACT_ROUTER_FEATURES:
    feat_dir = os.path.join(OUT, "router_features"); os.makedirs(feat_dir, exist_ok=True)
    feat_path = os.path.join(feat_dir, "router_features.npz")
    cmd = [sys.executable, "analysis/router_persample.py", "extract", "--config", "configs/mtil_order1.yaml",
           "--out", feat_path, "--set", *COMMON, "eval.max_samples=2000"]   # tối đa 2000 ảnh test/dataset: full 11 dataset sẽ vượt MAX_MB của zip
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    print(r.stdout[-2500:], r.stderr[-2500:])
    if r.returncode == 0:
        r = subprocess.run([sys.executable, "analysis/router_persample.py", "analyze", feat_path],
                           cwd=REPO, capture_output=True, text=True)
        print(r.stdout[-5000:], r.stderr[-1500:])
""")

code(r"""
# ===================== 9. Đồ thị loss / alpha / gate =====================
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(EXPERIMENTS), 3, figsize=(16, 3.6 * len(EXPERIMENTS)), squeeze=False)
for row, (name, _, _) in enumerate(EXPERIMENTS):
    path = os.path.join(OUT, name, "log.jsonl")
    if not os.path.exists(path): continue
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if "it" in r]
    if not recs: continue
    tasks = list(dict.fromkeys(r["task"] for r in recs)); iters = max(r["it"] for r in recs) + 1
    step = [tasks.index(r["task"]) * iters + r["it"] for r in recs]
    for key, col in [("L_cond", 0), ("L_marg", 0), ("L_pa", 0), ("alpha", 1), ("alpha_tilde", 1), ("phi", 2)]:
        pts = [(s, r[key]) for s, r in zip(step, recs) if key in r]
        if pts: axes[row][col].plot(*zip(*pts), label=key, lw=1)
    for ax, title in zip(axes[row], ["loss", "alpha / alpha_tilde", "phi (Eq. 43)"]):
        ax.set_title(f"{name}: {title}"); ax.legend(fontsize=8)
        for t in range(1, len(tasks)): ax.axvline(t * iters, color="gray", lw=0.4)
plt.tight_layout(); plt.show()
""")

code(r"""
# ===================== 10. Kết quả: mọi giao thức đánh giá, so với paper =====================
PAPER_SUMMARY = (71.76, 81.05, 90.54)   # PACT, MTIL Order-I (Bảng 1)
PAPER_LAST = {"Aircraft": 65.58, "Caltech101": 96.60, "CIFAR100": 91.21, "DTD": 80.53, "EuroSAT": 97.98,
              "Flowers": 98.59, "Food": 93.81, "MNIST": 99.53, "OxfordPet": 95.66, "StanfordCars": 89.90,
              "SUN397": 86.60}          # Bảng 8 (supp), hàng cuối = độ chính xác ngay sau khi học task đó
for name, _, _ in EXPERIMENTS:
    path = os.path.join(OUT, name, "results.json")
    if not os.path.exists(path): print(f"{name}: chưa có results.json"); continue
    res = json.load(open(path)); order = res["order"]
    print(f"===== {name}")
    for proto, r in res.get("protocols", {"main": res}).items():
        sm = r["summary"]
        line = f"  {proto:16s} Transfer={sm['Transfer']:6.2f}  Avg={sm['Avg']:6.2f}  Last={sm['Last']:6.2f}"
        if PRESET == "full": line += "   | paper %.2f / %.2f / %.2f" % PAPER_SUMMARY
        print(line)
    acc = res.get("protocols", {}).get("oracle_zeroshot", {}).get("acc_matrix")
    if acc:   # Bảng 8: cột task đã học phải giữ nguyên ở các hàng sau
        stable = all(acc[i][j] == acc[j][j] for i in range(len(acc)) for j in range(i + 1))
        print("  Bảng 8 (cột đã học bất biến):", "OK" if stable else "KHÔNG")
        print(f"  {'dataset':14s}{'ngay sau học':>14s}{'paper':>8s}")
        for j, n in enumerate(order):
            print(f"  {n:14s}{acc[j][j]:14.2f}{PAPER_LAST[n]:8.2f}")
    if res.get("router"):
        print("  router (hàng cuối):", {n: v["to"] for n, v in res["router"][-1].items()})
if PRESET in ("quick", "mid", "dry"):
    print(f"\n({PRESET}: {len(ORDER)} dataset, {PRESETS[PRESET]['extra'][0].split('=')[1]} iter thay vì 11 dataset, 3000 iter: Transfer/Avg/Last không so trực tiếp "
          "với paper; cột 'ngay sau học' so được nhưng sẽ thấp hơn vì train ngắn hơn)")
""")

code(r"""
# ===================== 11. Đóng gói kết quả thành 1 file zip để gửi =====================
# Gồm: log.jsonl, results.json, config.yaml, stdout.log của mọi run (cả zero-shot, chẩn đoán),
# đồ thị ô 9, nguồn dữ liệu, thông tin môi trường, evaluator.pt (vài MB, cho analysis/router_offline.py).
# KHÔNG gồm checkpoint (state.pt) trừ run được liệt kê dưới đây. Checkpoint của MỌI run vẫn nằm trong
# /kaggle/working/outputs (tab Output của notebook), dùng lại được qua RESUME_FROM mà không cần tải về.
import os, sys, json, time, zipfile, platform, subprocess
INCLUDE_CHECKPOINTS = ["pact_s42"]   # True: mọi run | False hoặc []: không run nào | danh sách tên run
WORK = globals().get("WORK", "/kaggle/working")
OUT, DATA_ROOT = globals().get("OUT", f"{WORK}/outputs"), globals().get("DATA_ROOT", f"{WORK}/data")
MAX_MB = 50   # bỏ qua file lẻ lớn hơn mức này (trừ checkpoint khi đã bật)

if "fig" in globals():   # đồ thị từ ô 9
    fig.savefig(f"{OUT}/curves.png", dpi=110, bbox_inches="tight")

def versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "torchvision", "open_clip", "numpy"):
        try: out[mod] = __import__(mod).__version__
        except Exception as e: out[mod] = f"n/a ({e})"
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception: pass
    return out

procs = globals().get("PROCS", {})
info = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "preset": globals().get("PRESET"), "order": globals().get("ORDER"),
    "gpus_per_run": globals().get("GPUS_PER_RUN"), "experiments": globals().get("EXPERIMENTS"),
    "common_overrides": globals().get("COMMON"), "num_workers": globals().get("NUM_WORKERS"),
    "exit_codes": {n: p.returncode for n, p in procs.items()},
    "code_build": (lambda f: json.load(open(f)) if os.path.exists(f) else None)(f"{globals().get('REPO', '')}/BUILD_INFO.json"),
    "versions": versions(), "cpu_count": os.cpu_count(),
}
try:
    info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
except Exception: pass

bundle = f"{WORK}/pact_results_{info['preset'] or 'run'}_{time.strftime('%Y%m%d_%H%M')}.zip"
skipped = []
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, dirs, files in os.walk(OUT):
        for f in sorted(files):
            full = os.path.join(folder, f)
            is_ckpt = f.endswith(".pt") and f != "evaluator.pt"   # evaluator.pt: nhỏ, cho phân tích router
            run_name = os.path.relpath(full, OUT).split(os.sep)[0]
            keep_ckpt = INCLUDE_CHECKPOINTS is True or (isinstance(INCLUDE_CHECKPOINTS, (list, tuple))
                                                        and run_name in INCLUDE_CHECKPOINTS)
            if (is_ckpt and not keep_ckpt) or \
               (not is_ckpt and os.path.getsize(full) > MAX_MB * 2**20):
                skipped.append(os.path.relpath(full, OUT)); continue
            z.write(full, os.path.join("outputs", os.path.relpath(full, OUT)))
    if os.path.exists(f"{DATA_ROOT}/_sources.json"):
        z.write(f"{DATA_ROOT}/_sources.json", "data_sources.json")
    z.writestr("run_info.json", json.dumps(info, indent=2, ensure_ascii=False, default=str))

with zipfile.ZipFile(bundle) as z: names = z.namelist()
print(f"{bundle}: {os.path.getsize(bundle) / 2**20:.1f} MB, {len(names)} file")
print("bỏ qua:", skipped or "không có")
print("Tải về: thanh bên phải -> Output -> /kaggle/working -> file .zip, hoặc bấm link dưới.")
from IPython.display import FileLink, display
cwd = os.getcwd(); os.chdir(WORK); display(FileLink(os.path.basename(bundle))); os.chdir(cwd)
""")

md(r"""
## Chạy tiếp khi phiên Kaggle hết giờ

1. Version vừa chạy có `outputs/<tên cấu hình>/state.pt` (lưu sau mỗi task).
2. Tạo version mới, *Add Input* → chọn **output của notebook này** (version trước).
3. Đặt `RESUME_FROM = "/kaggle/input/<tên notebook>/outputs"` ở ô cấu hình, giữ nguyên `PRESET` và `EXPERIMENTS`, rồi *Save & Run All*.

## Chọn số GPU cho mỗi cấu hình

`train.batch_size` (mặc định 64) là batch **toàn cục**, chia đều cho các GPU. ViT không có BatchNorm nên chạy 2 hay 4 GPU vẫn tối ưu cùng một hàm mục tiêu như 1 GPU, chỉ khác thứ tự lấy mẫu ngẫu nhiên. Nếu CPU nạp ảnh không kịp (`GPU-Util` trong `nvidia-smi` thấp), thêm GPU cho một cấu hình sẽ không nhanh hơn; khi đó nên chạy nhiều cấu hình song song.
""")

# ----------------------------------------------------------------------------- phase B variant
# python kaggle/make_notebook.py kaggle/pact_phase_b.ipynb --phase-b
# Same pipeline, but: preset mid (8 datasets), 4 runs x 1 GPU in parallel (lr 5e-5, seed 42), each changing ONE
# factor against pact_base; afterwards every run is re-evaluated with the cond / marg branches (phase A,
# analysis/reeval_branch.py) on its own GPU, and a comparison table is printed.
PHASE_B = "--phase-b" in sys.argv


def _cell(marker):
    hits = [i for i, c in enumerate(cells) if marker in c["source"]]
    assert len(hits) == 1, (marker, hits)
    return hits[0]


def _replace(i, old, new):
    assert old in cells[i]["source"], (i, old[:60])
    cells[i]["source"] = cells[i]["source"].replace(old, new)


def _insert_after(i, kind, src):
    cells.insert(i + 1, {"cell_type": kind, "metadata": {}, **({"execution_count": None, "outputs": []}
                                                              if kind == "code" else {}),
                         "source": src.strip("\n")})


if PHASE_B:
    cells[0]["source"] = (
        "# PACT – giai đoạn B: thu hẹp khoảng cách với paper (preset mid, 4 run song song)\n\n"
        "Mỗi run đổi **một** yếu tố so với `pact_base` (lr 5e-5, seed 42, 8 dataset, 2000 iter/task):\n\n"
        "| run | thay đổi | nguyên nhân kiểm |\n|---|---|---|\n"
        "| `pact_base` | PACT, lr 5e-5 | baseline mới |\n"
        "| `iso_base` | bỏ L_marg, L_pa | ràng buộc đóng góp bao nhiêu ở lr mới |\n"
        "| `pact_late` | α̃ bật ở 60% thay vì 20% | ràng buộc bật quá sớm |\n"
        "| `pact_softpa` | `pa_sigma_rel` 0.022 → 0.05 | L_pa kìm quá mạnh |\n\n"
        "Sau khi train, ô 8c đánh giá lại mọi run bằng nhánh `cond` và `marg` (giai đoạn A, "
        "`analysis/reeval_branch.py`), ô 10b in bảng so sánh. Chạy: *Save Version → Save & Run All* "
        "(khoảng 3.5–4 giờ, 4×L4).\n\n---\n\n" + cells[0]["source"])

    i = _cell("# ===================== CẤU HÌNH")
    src = cells[i]["source"]
    a, b = src.index("PRESET = "), src.index("RESUME_FROM = ")
    cells[i]["source"] = src[:a] + '''PRESET = "mid"           # giai đoạn B: 8 dataset, 2000 iter/task (rẻ hơn full khoảng một nửa)
GPUS_PER_RUN = 1         # 4 run x 1 GPU, chạy song song trên 4xL4
LR = ["train.lr=5e-5", "seed=42"]   # lr đã chốt sau pilot; giữ giống nhau ở mọi run
ALL_EXPERIMENTS = [      # mỗi run đổi đúng MỘT yếu tố so với pact_base
    ("pact_base",   "configs/mtil_order1.yaml",                 [*LR]),
    ("iso_base",    "configs/analysis/isolation_baseline.yaml", [*LR]),
    # latest_frac phải >= warmup_frac, nếu không công tắc bật theo deadline 50% trước khi hết warmup
    ("pact_late",   "configs/mtil_order1.yaml",                 [*LR, "pact.schedule.warmup_frac=0.6",
                                                                  "pact.schedule.latest_frac=0.8"]),
    ("pact_softpa", "configs/mtil_order1.yaml",                 [*LR, "pact.pa_sigma_rel=0.05"]),
]
RUN_ONLY = None          # None = chạy cả 4 (cần 4 GPU); hoặc danh sách tên run
EXPERIMENTS = [e for e in ALL_EXPERIMENTS if RUN_ONLY is None or e[0] in RUN_ONLY]
assert EXPERIMENTS, f"RUN_ONLY={RUN_ONLY} không khớp tên run nào"
''' + src[b:]
    _replace(i, "RUN_ZERO_SHOT_CHECK = False", "RUN_ZERO_SHOT_CHECK = False   # zero-shot đã kiểm (Khánh, 25/11)")

    j = _cell("EXTRACT_ROUTER_FEATURES = True")
    _replace(j, "EXTRACT_ROUTER_FEATURES = True", "EXTRACT_ROUTER_FEATURES = False   # giai đoạn B: không cần")
    _insert_after(j, "code", r'''
# ===================== 8c. Giai đoạn A: đánh giá lại mọi run bằng nhánh cond / marg =====================
# Mỗi run trên GPU riêng (song song), 2000 ảnh test/dataset. Ghi outputs/<run>/reeval_branch.json (nằm trong zip).
#   cond/cond: trung bình adapter 1..k, head cond (như run đã báo cáo)
#   marg/marg: chỉ adapter k, head marg  |  marg/cond: chỉ adapter k, head cond (iso không train head marg)
RUN_REEVAL = True
if RUN_REEVAL:
    procs = {}
    for i, (name, _, _) in enumerate(EXPERIMENTS):
        run_dir = os.path.join(OUT, name)
        if not os.path.exists(os.path.join(run_dir, "state.pt")):
            print(f"[{name}] không có state.pt, bỏ qua"); continue
        log_f = open(os.path.join(run_dir, "reeval_branch.log"), "w")
        cmd = [sys.executable, "analysis/reeval_branch.py", "--run-dir", run_dir, "--max-samples", "2000",
               "--set", *COMMON]
        procs[name] = (subprocess.Popen(cmd, cwd=REPO, stdout=log_f, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i % max(n_gpu, 1))}), log_f)
        print(f"[{name}] reeval trên GPU {i % max(n_gpu, 1)}")
    for name, (p, log_f) in procs.items():
        p.wait(); log_f.close()
        print(f"\n===== {name} (exit {p.returncode})")
        print("".join(open(os.path.join(OUT, name, "reeval_branch.log")).readlines()[-22:]))
''')

    k = _cell("# ===================== 10. Kết quả")
    _insert_after(k, "code", r'''
# ===================== 10b. So sánh giai đoạn B (mỗi run so với pact_base) =====================
def load(name, fname="results.json"):
    path = os.path.join(OUT, name, fname)
    return json.load(open(path)) if os.path.exists(path) else None

base = load("pact_base")
print(f"{'run':13s}{'Transfer(o_latest)':>20s}{'Last(oracle)':>14s}{'Avg(o_latest)':>15s}{'Δ Transfer':>12s}{'Δ Last':>9s}")
for name, _, _ in EXPERIMENTS:
    r = load(name)
    if not r: print(f"{name:13s} chưa có kết quả"); continue
    t = r["protocols"]["oracle_latest"]["summary"]; o = r["protocols"]["oracle_zeroshot"]["summary"]
    line = f"{name:13s}{t['Transfer']:20.2f}{o['Last']:14.2f}{t['Avg']:15.2f}"
    if base and name != "pact_base":
        bt = base["protocols"]["oracle_latest"]["summary"]; bo = base["protocols"]["oracle_zeroshot"]["summary"]
        line += f"{t['Transfer'] - bt['Transfer']:+12.2f}{o['Last'] - bo['Last']:+9.2f}"
    print(line)
zs = base["zero_shot"] if base else None
if zs: print(f"{'zero-shot':13s}{sum(zs[1:]) / (len(zs) - 1):20.2f}   (Transfer của CLIP gốc trên cùng thứ tự)")

print("\nĐộ chính xác ngay sau khi học (đường chéo oracle_zeroshot):")
runs = [(n, load(n)) for n, _, _ in EXPERIMENTS]
runs = [(n, r) for n, r in runs if r]
if runs:
    order = runs[0][1]["order"]
    print(f"{'dataset':14s}" + "".join(f"{n:>13s}" for n, _ in runs))
    for j, d in enumerate(order):
        print(f"{d:14s}" + "".join(f"{r['protocols']['oracle_zeroshot']['acc_matrix'][j][j]:13.2f}" for _, r in runs))

print("\nGiai đoạn A (2000 ảnh test/dataset): Last theo nhánh lúc test")
print(f"{'run':13s}{'cond/cond':>11s}{'marg/marg':>11s}{'marg/cond':>11s}")
for name, _, _ in EXPERIMENTS:
    rb = load(name, "reeval_branch.json")
    if not rb: print(f"{name:13s} chưa có reeval_branch.json"); continue
    vals = [rb["results"].get(v, {}).get("summary", {}).get("oracle_zeroshot", {}).get("Last") for v in
            ("cond/cond", "marg/marg", "marg/cond")]
    print(f"{name:13s}" + "".join(f"{x:11.2f}" if x is not None else f"{'-':>11s}" for x in vals))
print("\nĐọc: Δ Last > 0 mà Δ Transfer không giảm -> thay đổi đó thu hẹp khoảng cách. marg/* > cond/cond rõ rệt ->"
      " nhánh cond pha loãng adapter, cách đánh giá của paper có thể là marg.")
''')

    m = _cell("INCLUDE_CHECKPOINTS = ")
    _replace(m, 'INCLUDE_CHECKPOINTS = ["pact_s42"]', 'INCLUDE_CHECKPOINTS = []   # giai đoạn B: state.pt vẫn nằm trong tab Output')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
