# PACT (unofficial)

Bản tự cài đặt lại paper **PACT: Phase-Like Transition Constraints in Adapter-Based Continual Learning of Vision-Language Models** (Wang, Ding, Han, CVPR 2026). Repo chính thức của tác giả hiện vẫn trống, nên code này được viết **dựa trên main paper và supplementary** (Appendix 6.3, phân tích convex mixing, Bảng 8). Những chỗ cả hai không nói rõ được ghi ở mục "Giả định". Phân tích chi tiết nằm ở `reports/Giải mã chi tiết PACT.md`.

## 1. Cài đặt

Cần GPU. Cài PyTorch bản CUDA theo hướng dẫn trên pytorch.org, sau đó:

```bash
pip install -r requirements.txt
```

Kiểm tra nhanh logic (chạy trên CPU, dùng một CLIP giả rất nhỏ, không tải gì):

```bash
python tests/smoke_test.py
```

## 2. Dữ liệu

Mặc định dữ liệu nằm ở `data/` (đổi bằng `--set data_root=D:/datasets`). torchvision sẽ tự tải phần lớn dataset. Một số phải tải tay:

- **StanfordCars**: torchvision không tải được nữa, cần đặt thủ công vào `data_root/stanford_cars/`.
- **Caltech101**: tải qua Google Drive, đôi khi lỗi.
- **TinyImageNet** (cho CIL): giải nén `tiny-imagenet-200/` vào `data_root/`.

Cách chia dữ liệu giống ZSCL / MoE-Adapters:
- Caltech101, EuroSAT, SUN397 chia 80/20 với seed 42.
- Tên lớp và prompt lấy theo CLIP.
- Thứ tự lớp CIL nằm trong `configs/class_orders/`.

## 3. Chạy

```bash
# MTIL (bảng chính của paper)
python -m pact.run_mtil --config configs/mtil_order1.yaml

# Few-shot MTIL
python -m pact.run_mtil --config configs/mtil_fewshot5.yaml

# CIL
python -m pact.run_cil --config configs/cil_cifar100_10steps.yaml

# Ablation (Bảng 5)
python -m pact.run_mtil --config configs/ablation/lora_per_task.yaml
```

- **Ghi đè tham số:** `--set train.batch_size=16 train.grad_accum=4 eval.max_samples=2000`
- **Chạy tiếp sau khi bị dừng:** thêm `--resume`. Trạng thái được lưu sau mỗi task.
- **GPU 8GB** (ví dụ RTX 4060): dùng `train.batch_size=16 train.grad_accum=4`.
- **Nhiều GPU cho một cấu hình:** `torchrun --standalone --nproc_per_node=2 -m pact.run_mtil --config ...`. `train.batch_size` là batch toàn cục, chia đều cho các GPU. Gradient được all-reduce, eval chia shard rồi cộng dồn nên độ chính xác khớp với chạy 1 GPU. Chỉ rank 0 ghi log và checkpoint.
- **Chỉ zero-shot:** `--set adapter.type=none` (đánh giá một lần, không train).
- **Trọng số CLIP:** mặc định là `ViT-B-16-quickgelu`. Trọng số OpenAI được train với QuickGELU, nhưng open_clip ≥ 2.2x dựng `ViT-B-16` bằng GELU thường kể cả khi `pretrained=openai` (chỉ phát warning), làm độ chính xác tụt mà không báo lỗi.

### Chạy trên Kaggle (offline, nhiều GPU)

1. Trên máy có mạng: `python tools/prepare_upload.py`. Lệnh này tạo ra:
   - `dist/pact-code.zip` (~2 MB): code, wheel `open_clip` + `ftfy`, và notebook.
   - `dist/pact-clip.zip` (~570 MB): trọng số CLIP kèm file kiểm tra.
   - `dist/pact_demo.ipynb`.

   Khi chỉ sửa code, chạy `--skip-clip --skip-wheels` để không phải tải lại gì.
2. Kaggle chỉ nhận file, không nhận thư mục. Upload `pact-clip.zip` **một lần**, và `pact-code.zip` mỗi khi code đổi (tạo version mới của dataset). Rồi import `dist/pact_demo.ipynb`. Nếu gắn nhiều bản code cùng lúc, notebook chọn bản mới nhất nhờ `PACT/BUILD_INFO.json`.
3. Trong notebook: gắn `pact-kit` và các dataset ảnh, rồi chạy từ trên xuống. Các ô đầu sẽ kiểm tra CLIP (`tools/clip_offline.py verify`), dò và dựng dữ liệu (`tools/kaggle_data.py`), và so zero-shot với Bảng 1 trước khi train.

`tools/kaggle_data.py` nhận ra từng dataset theo file đặc trưng, không theo tên thư mục, rồi dựng `data_root` bằng symlink đúng cấu trúc torchvision. Nhờ vậy các bản Kaggle có cấu trúc khác nhau vẫn dùng được.

Kết quả nằm trong `output_dir`:
- `log.jsonl`: log train (các loss, α, α̃, φ, d_min, d_mlp) và kết quả sau mỗi task (kèm quyết định của router).
- `results.json`:
  - `acc_matrix`, `per_dataset`, `summary`: giao thức chính (phần tử đầu của `eval.protocols`).
  - `protocols`: ma trận và Transfer / Avg / Last của **từng** giao thức.
  - `router`: router đã gửi từng dataset tới đâu.
- `state.pt`: checkpoint gồm adapter, gate, head và cache của evaluator.

## 4. Đối chiếu paper và code

| Paper | Code |
|---|---|
| Adapter song song với MLP, cùng kiến trúc, copy trọng số MLP, gate sigmoid (SAI 1–4) | `pact/adapters.py` (`MLPAdapter`, `PACTMLP`) |
| Chèn adapter ở layer 3, 6, 9, 12 (SAI 5) | `adapter.layers` |
| Nhánh conditional (trung bình các adapter + 0.1ε), nhánh marginal (Eq. 4, 5) | `PACTMLP.forward`, `model.logits(..., branch)` |
| `L_pa` (Eq. 9) | `pact/losses.py` |
| Probe bank, khoảng cách d_j, α = exp(−d/τ) (Eq. 11–14) | `pact/phase.py: compute_alpha` |
| α̃ chỉ bật sau khi hội tụ, `L = L_cond + α̃(L_marg + L_pa)` (Eq. 15) | `PostConvergenceSchedule`, `engine.train_task` |
| Transfer / Avg / Last | `pact/metrics.py` |
| α̃ = 𝟙[φ ≤ γ]·α, φ theo cửa sổ (supp. Eq. 40–44) | `PostConvergenceSchedule` (mode `window`) |
| g_t cố định tăng theo tầng, tầng cuối 0.5 (supp., "convex mixing") | `adapter.gate_mode: layerwise` |
| "Two lightweight heads" được cập nhật (supp.) | `model.TaskHead`, `head.type: learned` |
| Bảng 8: độ chính xác task đã học không đổi, "routing algorithm" (supp.) | `pact/evaluation.py` (snapshot theo task, router) |

## 5. Giả định (paper không nói rõ, cần tinh chỉnh)

Đây là những điểm supplementary đã làm rõ so với main paper, và là mặc định mới:

- **Cách đánh giá MTIL (quan trọng nhất).** Bảng 8 của supplement cho thấy độ chính xác của mỗi task đã học **giữ nguyên tuyệt đối** ở mọi bước sau. Phần chứng minh lại nhắc tới một "routing algorithm" không được định nghĩa. Hai điều này chỉ khớp khi dữ liệu của task k được đánh giá bằng **snapshot của chính task k**, gồm adapter 1..k, gate k, head k, tất cả đã đóng băng.

  Cách chọn snapshot lúc test thì paper không mô tả, nên mỗi run tính cùng lúc 4 giao thức (`eval.protocols`, phần tử đầu là kết quả chính):

  | Giao thức | Dữ liệu đã học | Dữ liệu chưa học |
  |---|---|---|
  | `router` | Mỗi batch test được gửi tới task có prototype gần nhất (đặc trưng CLIP đóng băng) | Về CLIP đóng băng nếu dưới ngưỡng, giống DDAS của MoE-Adapters |
  | `oracle_zeroshot` | Biết task-ID, dùng snapshot của task đó | CLIP đóng băng |
  | `oracle_latest` | Biết task-ID, dùng snapshot của task đó | Mô hình mới nhất |
  | `latest` | Mô hình mới nhất (cách cũ của repo, gây quên) | Mô hình mới nhất |

  Ngưỡng của router là quantile `eval.router.quantile` (5%) của cosine giữa ảnh train và prototype, đây là giả định. Các snapshot đã đóng băng nên kết quả được cache: mỗi tổ hợp (dataset, snapshot, head) chỉ đánh giá một lần.
- **Gate g_t**: dùng lịch cố định tăng theo tầng, tầng cuối 0.5, không học. Dạng hàm thì paper không nêu; mặc định tuyến tính theo độ sâu, cho 0.125 / 0.25 / 0.375 / 0.5 ở tầng 3 / 6 / 9 / 12 (`gate_power`). Vẫn có các lựa chọn khác:
  - `gate_mode: constant`: g cố định ở mọi tầng; supplement cho tốt nhất ở 0.1.
  - `gate_mode: learned`: gate sigmoid học được, chính là bản cũ với `gate_init_mode: preserve` (g_t = g_{t−1}·t/(t−1), chặn ở 0.95). Config `configs/ablation/learned_gate.yaml`.
- **Head**: supplement nói "only the current adapter and two lightweight heads are updated". Vì vậy mỗi task có hai head, một cho nhánh cond và một cho nhánh marg. Dạng head thì paper không nêu: ở đây là classifier cosine trên các lớp của task, khởi tạo từ text embedding (ensemble template). Lúc test, dữ liệu của task k dùng head cond k. Dữ liệu chưa học dùng text embedding đóng băng. Đặt `head.type=text` để quay về head text đóng băng.
- **α̃** theo Appendix 6.3: φ = w1·√v/(ℓ̄+ε) + w2·|m| trên W loss gần nhất của L_cond. Khi φ ≤ γ thì bật cứng α̃ = α và giữ luôn (latch). Không có ramp, vì Fig. 6 cho thấy loss "nhảy" lúc PACT bật. Các giá trị W = 100, w1 = w2 = 0.5, γ = 0.1 là giả định. Ngoài ra không bật trong 20% đầu, và nếu φ chưa bao giờ ≤ γ thì bắt buộc bật ở 50% số iteration. φ được ghi vào log để chỉnh γ. Cách cũ (EMA + ramp) vẫn còn ở `pact.schedule.mode=stability`.
- **Protocol kế thừa từ ZSCL / MoE-Adapters**:
  - Label smoothing 0.2.
  - Train dùng template đầu, eval dùng ensemble mọi template trong `prompts.md` của CLIP.
  - Eval chạy fp32 (cho phép TF32).
  - Few-shot seed 42.
  - Template CIL "a bad photo of a {}.".

Các giả định còn lại:

- **Gate khi chèn adapter** (chỉ khi `gate_mode: learned`): task 1 dùng γ = `gate_init`. Từ task t ≥ 2, g_t = g_{t−1}·t/(t−1) để nhánh conditional không đổi lúc chèn, và bị chặn ở `gate_max_init`.
- **`L_pa`**: coi Q là Gaussian đẳng hướng với độ lệch chuẩn tỉ lệ với độ lớn trọng số. Khi đó KL = ‖θ−θ₀‖² / (2σ_rel²‖θ₀‖²), tức là độ trôi tương đối, chuẩn hoá giống d_j ở Eq. 12. Đo trên CLIP ViT-B/16 thật cho thấy d ≈ 80–125 × r², với r là độ trôi tương đối. Vì vậy mặc định σ_rel = 0.022 để L_pa ≈ d/τ = −ln α so với MLP. Giá trị 0.22 dùng trước đó làm L_pa yếu khoảng 100 lần.
- **Hiệu chỉnh τ và `pa_sigma_rel`**: log có `d_min`, `d_mlp` và `L_pa`. Nên chọn τ cùng bậc với `d_min` điển hình ở cuối task, rồi đặt `pa_sigma_rel ≈ √(τ/200)` (với d ≈ 100·r²).
- **Task 1** chưa có adapter cũ, nên α được tính so với MLP pretrained.
- **Nhiễu 0.1ε** chỉ thêm khi train.
- **CIL** không có task-ID, nên dùng nhánh cond với mọi adapter và head text đóng băng (`cil.head: text`). Tuỳ chọn `cil.head: concat` ghép head cond của các task theo thứ tự lớp.
- **Siêu tham số chưa biết**: lr, τ, số probe, W, γ, dạng lịch gate. Tất cả nằm trong `pact/config.py` và được đánh dấu `ASSUMPTION`. lr 1e-5 (lr full fine-tune của ZSCL) chưa chắc đúng cho adapter, nên notebook có quét thử 5e-5.

Trước khi train, nên chạy riêng zero-shot (`--set adapter.type=none`) và so với dòng CLIP zero-shot ở Bảng 1: 24.3 / 88.4 / 68.2 / 44.6 / 54.9 / 71.0 / 88.5 / 59.4 / 89.0 / 64.7 / 65.2. Chế độ này đo cả template đầu lẫn ensemble (`zero_shot_by_template`). Dataset nào lệch hơn khoảng 1 điểm thì cần xem lại split, tên lớp hoặc template của dataset đó.

Mốc kiểm tra theo Bảng 8 (Order-I) là độ chính xác ngay sau khi học mỗi task: Aircraft 65.58, Caltech101 96.60, CIFAR100 91.21, DTD 80.53, EuroSAT 97.98, Flowers 98.59, Food 93.81, MNIST 99.53, OxfordPet 95.66, StanfordCars 89.90, SUN397 86.60. Một cấu hình được xem là đúng hướng khi thoả hai điều kiện: các cột đã học không đổi, và Transfer cao hơn mức zero-shot (69.4).

Sau đó kiểm tra theo thứ tự các dòng của Bảng 5: LoRA mỗi task (59.23 / 59.82 / 79.54), rồi thêm gate (67.50 / 75.94 / 85.60), rồi PACT đầy đủ (71.76 / 81.05 / 90.54).

## 6. Mở rộng

- Đổi cách tính khoảng cách giữa adapter hoặc công thức α: `pact/phase.py`
- Đổi cách gộp adapter (thay vì lấy trung bình): `PACTMLP.forward`
- Thêm loss mới: `engine.train_task`
- Thêm dataset hoặc thứ tự task: `pact/data/`, `mtil.order` (đã có `I` và `II`)

## 7. Trạng thái

- Logic đã được kiểm tra bằng `tests/smoke_test.py`:
  - SAI giữ nguyên dự đoán lúc vừa chèn adapter, ở task 1 và cả task 2, 3 (gate g_t = g_{t−1}·t/(t−1)). Trường hợp gate bị chặn ở `gate_max_init` cũng có test.
  - L_pa bằng 0 lúc chèn và bằng 0.5 khi độ trôi tương đối đúng bằng σ_rel.
- `tests/ddp_test.py`: 2 process (gloo, CPU). Kiểm tra tham số giống hệt nhau giữa các rank, eval chia shard bằng eval 1 process, và log chỉ ghi một lần.
- `tests/kaggle_data_test.py`: dựng một `/kaggle/input` giả với cấu trúc lộn xộn, kiểm tra script dò và dựng đúng cả 12 dataset, nạp được 8 dataset bằng loader torchvision, và bỏ qua Fashion-MNIST mồi nhử.
  - Adapter và gate cũ được đóng băng.
  - α bằng 1 lúc vừa chèn adapter.
  - Lưu và nạp lại checkpoint cho kết quả giống hệt.
  - Các biến thể ablation đều chạy được.
- Smoke test còn kiểm tra:
  - Gate theo tầng là 0.125 / 0.25 / 0.375 / 0.5 và bị đóng băng.
  - Head được train, head cũ bị đóng băng.
  - Cột đã học bất biến theo Bảng 8, kể cả khi tính lại từ đầu.
  - Cache của evaluator và checkpoint có head.
  - Công tắc window bật ở warmup khi loss ổn định, và bật theo deadline khi loss quá nhiễu.
- `tests/ddp_test.py` kiểm tra evaluator theo snapshot và router chạy trên 2 process cho kết quả giống hệt 1 process.
- Đã chạy với CLIP thật ở preset quick, theo cách đánh giá cũ (`latest`), gate học được, 300 iteration. Các cấu hình PACT cho Avg ≈ 61 và Last ≈ 65. Cách đánh giá đó làm quên mạnh, và đây là lý do có chế độ snapshot.

Nguồn: tên lớp lấy từ OpenAI CLIP `data/prompts.md` (MIT). Thứ tự lớp CIL lấy từ ZSCL / MoE-Adapters4CL.
