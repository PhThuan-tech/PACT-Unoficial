# Nhật ký thí nghiệm

Mỗi mục do `tools/log_run.py` ghi từ một file zip kết quả. Số liệu đầy đủ nằm ở `experiments/runs/<id>.json`.
Các số của preset khác nhau (quick / mid / full) không so trực tiếp với nhau hay với paper.

<!-- run:pact_results_quick_20261119_1508 -->
## pact_results_quick_20261119_1508

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-19 15:08:13 | GPU mỗi run: 2
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| baseline_gate | configs/ablation/sai124_gate.yaml | - | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 8m |
| pact_full | configs/mtil_order1.yaml | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 8m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 4m |

### Kết quả (Transfer / Avg / Last)

| run | (cũ) | zero-shot |
|---|---|---|
| baseline_gate | 61.73 / 57.89 / 61.05 | - |
| pact_full | 58.98 / 60.16 / 63.42 | - |
| zero_shot | - | - / 52.70 / - |

Last theo từng dataset (protocol chính):

- **baseline_gate** (None): Aircraft 24.30, DTD 46.30, EuroSAT 83.50, OxfordPet 90.10
- **pact_full** (None): Aircraft 27.00, DTD 50.10, EuroSAT 87.30, OxfordPet 89.30

**Kết luận:** _(điền tay)_


<!-- run:pact_results_quick_20261119_1542 -->
## pact_results_quick_20261119_1542

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-19 15:42:14 | GPU mỗi run: 2
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| baseline_gate | configs/ablation/sai124_gate.yaml | - | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 6m |
| pact_full | configs/mtil_order1.yaml | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 7m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 2m |

### Kết quả (Transfer / Avg / Last)

| run | (cũ) | zero-shot |
|---|---|---|
| baseline_gate | 61.73 / 57.89 / 61.05 | - |
| pact_full | 59.71 / 60.94 / 64.70 | - |
| zero_shot | - | - / 52.70 / - |

Last theo từng dataset (protocol chính):

- **baseline_gate** (None): Aircraft 24.30, DTD 46.30, EuroSAT 83.50, OxfordPet 90.10
- **pact_full** (None): Aircraft 28.90, DTD 50.70, EuroSAT 89.10, OxfordPet 90.10

**Kết luận:** _(điền tay)_


<!-- run:pact_results_quick_20261119_1600 -->
## pact_results_quick_20261119_1600

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-19 16:00:52 | GPU mỗi run: 1
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| baseline_gate | None | - | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 6m |
| pact_fixedgate | configs/mtil_order1.yaml | adapter.gate_init_mode=fixed | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 13m |
| pact_full | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 7m |
| pact_pa010 | configs/mtil_order1.yaml | pact.pa_sigma_rel=0.01 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 12m |
| pact_pa022 | configs/mtil_order1.yaml | pact.pa_sigma_rel=0.022 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 12m |
| pact_pa022_fixedgate | configs/mtil_order1.yaml | pact.pa_sigma_rel=0.022 adapter.gate_init_mode=fixed | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 13m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=None, train.lr=1e-05, train.iterations=300, seed=42 | 2m |

### Kết quả (Transfer / Avg / Last)

| run | (cũ) | zero-shot |
|---|---|---|
| baseline_gate | 61.73 / 57.89 / 61.05 | - |
| pact_fixedgate | 60.06 / 61.16 / 65.40 | - |
| pact_full | 59.71 / 60.94 / 64.70 | - |
| pact_pa010 | 60.08 / 60.46 / 64.38 | - |
| pact_pa022 | 59.49 / 60.67 / 64.55 | - |
| pact_pa022_fixedgate | 60.15 / 60.94 / 65.15 | - |
| zero_shot | - | - / 52.70 / - |

Last theo từng dataset (protocol chính):

- **baseline_gate** (None): Aircraft 24.30, DTD 46.30, EuroSAT 83.50, OxfordPet 90.10
- **pact_fixedgate** (None): Aircraft 26.10, DTD 52.20, EuroSAT 92.00, OxfordPet 91.30
- **pact_full** (None): Aircraft 28.90, DTD 50.70, EuroSAT 89.10, OxfordPet 90.10
- **pact_pa010** (None): Aircraft 28.10, DTD 51.10, EuroSAT 87.90, OxfordPet 90.40
- **pact_pa022** (None): Aircraft 28.20, DTD 51.30, EuroSAT 88.30, OxfordPet 90.40
- **pact_pa022_fixedgate** (None): Aircraft 26.70, DTD 51.40, EuroSAT 92.10, OxfordPet 90.40

**Kết luận:** _(điền tay)_


<!-- run:pact_results_quick_20261124_0347 -->
## pact_results_quick_20261124_0347

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-24 03:47:49 | GPU mỗi run: 1
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| pact_learned_gate | configs/ablation/learned_gate.yaml | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=learned, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 46m |
| pact_lr5e-5 | configs/mtil_order1.yaml | train.lr=5e-5 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=1000, seed=42 | 46m |
| pact_paper | configs/mtil_order1.yaml | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 46m |
| pact_text_head | configs/mtil_order1.yaml | head.type=text | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=text, train.lr=1e-05, train.iterations=1000, seed=42 | 46m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 3m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router | zero-shot |
|---|---|---|---|---|---|
| pact_learned_gate | 59.42 / 64.99 / 67.92 | 59.42 / 69.19 / 77.75 | 62.67 / 70.39 / 77.75 | 60.22 / 69.47 / 77.75 | - |
| pact_lr5e-5 | 63.32 / 68.83 / 72.67 | 63.32 / 73.71 / 81.62 | 62.67 / 73.56 / 81.62 | 62.57 / 73.53 / 81.62 | - |
| pact_paper | 62.33 / 65.37 / 68.80 | 62.33 / 69.84 / 77.45 | 62.67 / 69.99 / 77.45 | 62.32 / 69.86 / 77.45 | - |
| pact_text_head | 62.16 / 62.81 / 65.97 | 62.16 / 66.93 / 73.95 | 62.67 / 67.12 / 73.95 | 62.28 / 66.98 / 73.95 | - |
| zero_shot | - | - | - | - | - / 53.32 / - |

Last theo từng dataset (protocol chính):

- **pact_learned_gate** (router): Aircraft 52.00, DTD 72.80, EuroSAT 94.90, OxfordPet 91.30
- **pact_lr5e-5** (router): Aircraft 60.40, DTD 77.30, EuroSAT 96.00, OxfordPet 92.80
- **pact_paper** (router): Aircraft 50.50, DTD 72.10, EuroSAT 95.60, OxfordPet 91.60
- **pact_text_head** (router): Aircraft 44.40, DTD 65.60, EuroSAT 95.10, OxfordPet 90.70

**Kết luận:** _(điền tay)_


<!-- run:pact_results_quick_20261124_1527 -->
## pact_results_quick_20261124_1527

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-24 15:27:34 | GPU mỗi run: 1
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| pact_learned_gate | configs/ablation/learned_gate.yaml | seed=2026 train.lr=1e-5 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=learned, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=2026 | 46m |
| pact_lr5e-5 | configs/mtil_order1.yaml | seed=2026 train.lr=5e-5 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=1000, seed=2026 | 45m |
| pact_paper | configs/mtil_order1.yaml | seed=2026 train.lr=1e-5 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=2026 | 45m |
| pact_text_head | configs/mtil_order1.yaml | seed=2026 train.lr=1e-5 head.type=text | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=text, train.lr=1e-05, train.iterations=1000, seed=2026 | 45m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 2m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router | zero-shot |
|---|---|---|---|---|---|
| pact_learned_gate | 59.17 / 64.79 / 67.90 | 59.17 / 69.00 / 77.57 | 62.67 / 70.29 / 77.57 | 60.18 / 69.36 / 77.57 | - |
| pact_lr5e-5 | 62.78 / 68.61 / 72.60 | 62.78 / 73.55 / 81.62 | 62.67 / 73.59 / 81.62 | 62.37 / 73.47 / 81.62 | - |
| pact_paper | 62.19 / 65.30 / 68.78 | 62.19 / 69.75 / 77.30 | 62.67 / 69.94 / 77.30 | 62.28 / 69.80 / 77.30 | - |
| pact_text_head | 62.04 / 62.71 / 65.73 | 62.04 / 66.91 / 73.88 | 62.67 / 67.12 / 73.88 | 62.28 / 66.98 / 73.88 | - |
| zero_shot | - | - | - | - | - / 53.32 / - |

Last theo từng dataset (protocol chính):

- **pact_learned_gate** (router): Aircraft 51.80, DTD 72.90, EuroSAT 94.50, OxfordPet 91.10
- **pact_lr5e-5** (router): Aircraft 60.60, DTD 77.20, EuroSAT 96.00, OxfordPet 92.70
- **pact_paper** (router): Aircraft 51.00, DTD 71.40, EuroSAT 95.40, OxfordPet 91.40
- **pact_text_head** (router): Aircraft 44.20, DTD 66.20, EuroSAT 94.80, OxfordPet 90.30

**Kết luận:** _(điền tay)_


<!-- run:pact_results_quick_20261125_0140 -->
## pact_results_quick_20261125_0140

- **Preset:** quick (4 dataset: Aircraft, DTD, EuroSAT, OxfordPet)
- **Tạo lúc:** 2026-11-25 01:40:29 | GPU mỗi run: 2
- **Exit code:** tất cả 0

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| isolation | configs/analysis/isolation_baseline.yaml | - | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 16m |
| pact_paper | configs/mtil_order1.yaml | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 23m |
| zero_shot | None | - | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=1000, seed=42 | 2m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router | zero-shot |
|---|---|---|---|---|---|
| isolation | 61.19 / 66.04 / 70.20 | 61.19 / 70.66 / 79.50 | 62.67 / 71.24 / 79.50 | 61.72 / 70.88 / 79.50 | - |
| pact_paper | 62.42 / 65.26 / 68.70 | 62.42 / 69.71 / 77.20 | 62.67 / 69.83 / 77.20 | 62.42 / 69.73 / 77.20 | - |
| zero_shot | - | - | - | - | - / 53.32 / - |

Last theo từng dataset (protocol chính):

- **isolation** (router): Aircraft 51.00, DTD 76.20, EuroSAT 97.70, OxfordPet 93.10
- **pact_paper** (router): Aircraft 50.60, DTD 71.20, EuroSAT 95.50, OxfordPet 91.50

Cột đã học bất biến theo Bảng 8 (protocol chính): isolation: có; pact_paper: có (kiểm tra từ `acc_matrix` đã lưu)

Router offline (`analysis/router_offline.py`, cả hai run giống nhau): dữ liệu đã học gửi đúng 100% ở delta từ -0.10 đến +0.05, sụp còn 15% ở +0.10. Unseen gửi nhầm sang snapshot: 16.7% ở delta 0, 0% ở +0.05.

Quét mịn delta (bước 0.01 từ 0.02 đến 0.10, ngày 2026-11-26): cả hai phía cùng đúng trong khoảng **delta 0.03 đến 0.06** (dữ liệu đã học 100% đúng, unseen 0% gửi nhầm, Transfer 62.67 = zero-shot). Ở delta 0.07 dữ liệu đã học còn 71.2% đúng (Last PACT 65.23, isolation 67.00); 0.08: 60.0%; 0.09: 46.2%; 0.10: 15.0%. Delta 0.02 còn 4.2% unseen bị gửi nhầm. Tức khoảng ngưỡng tốt rộng khoảng 0.04 đơn vị cosine, cách ngưỡng mặc định (delta 0) khoảng 0.03; 4 dataset rất khác nhau, router theo batch. Ghi chú: ở delta 0.03 đến 0.06 Transfer bằng đúng zero-shot, nghĩa là router "hoàn hảo" ở đây chỉ trả unseen về CLIP chứ không tạo ra lợi ích Transfer.

**Kết luận:** Chưa kết luận được, chỉ là tín hiệu sơ bộ (1 seed, 4 task, `eval.max_samples=1000`). Isolation có Last cao hơn PACT 2.3 điểm (79.5 so với 77.2), cao hơn ở cả ba task so sánh được, nên ràng buộc PACT chưa cho thấy đóng góp vào Last. Transfer: `oracle_latest` 62.42 (PACT) và 61.19 (isolation), đều thấp hơn zero-shot 62.67. Router theo batch đúng 100% là kết quả dễ (4 dataset rất khác nhau, đo theo batch), không phải bằng chứng router tốt. Việc tiếp theo: chạy `mid` hai seed để xác nhận.

<!-- run:pact_results_full_20261125_1155 -->
## pact_results_full_20261125_1155

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-25 11:55:56 | GPU mỗi run: 2
- **Code:** `20261125-1441-627d990-c92d780a` (commit 627d990, nhánh main)
- **Exit code:** tất cả 0
- **Ghi chú:** iso_s42, full, 2 GPU, seed 42 (chạy bởi Thuận)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| iso_s42 | configs/analysis/isolation_baseline.yaml | seed=42 | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=3000, seed=42 | 3h43m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| iso_s42 | 65.82 / 72.33 / 79.75 | 65.82 / 77.25 / 87.64 | 69.47 / 78.84 / 87.64 | 66.53 / 73.34 / 80.20 |

Last theo từng dataset (protocol chính):

- **iso_s42** (router): Aircraft 63.01, Caltech101 97.47, CIFAR100 72.39, DTD 78.94, EuroSAT 85.91, Flowers 83.18, Food 91.50, MNIST 55.56, OxfordPet 94.52, StanfordCars 82.34, SUN397 77.42

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): iso_s42: có [oracle_zeroshot]

Chi tiết (đọc từ `results.json`, ngày 2026-11-25; chưa có run PACT cùng seed để so):

- **Thời gian:** 3h43m trên 2 GPU (ước lượng trước đó 2.5-3.5 giờ).
- **Snapshot hợp lệ:** cả `oracle_zeroshot` và `oracle_latest` bất biến 0 ô lệch (Bảng 8 đạt ở dạng này). Protocol `router` lệch 4 ô là bình thường vì quyết định router thay đổi theo hàng.
- **Zero-shot khớp Khánh:** Transfer zero-shot 69.47.
- **Đường chéo (isolation) so Bảng 8 của PACT:** trung bình 87.64 so với 90.54 (-2.90). Lệch nhiều nhất: SUN397 77.42 (-9.18), StanfordCars 82.34 (-7.56), CIFAR100 85.07 (-6.14), Flowers 96.00 (-2.59), Aircraft 63.01 (-2.57), Food 91.50 (-2.31); Caltech101 +0.87; EuroSAT -0.17, MNIST -0.33, OxfordPet -0.84, DTD -1.12. Chưa kết luận được ràng buộc PACT đóng góp bao nhiêu cho khoảng cách này, vì thiếu `pact_s42` cùng seed. Siêu tham số chưa biết (lr 1e-5 là giả định) cũng có thể là nguyên nhân.
- **Transfer:** `oracle_latest` 65.82, thấp hơn zero-shot 69.47 (-3.65): mô hình mới nhất không có ràng buộc làm hại dữ liệu chưa học. `oracle_zeroshot` bằng zero-shot theo định nghĩa. Cần xem PACT có giữ được Transfer không.
- **Router của repo kém hơn nhiều ở 11 dataset so với 4 dataset:** Last `router` 80.20 so với 87.64 của `oracle_zeroshot` (-7.4); Transfer `router` 66.53. Hàng cuối: MNIST bị gửi 100% sang snapshot Caltech101 (55.56 so với 99.20), EuroSAT 100% sang SUN397 (85.91 so với 97.81), CIFAR100 sang SUN397 (90%) và Caltech101 (10%) (72.39 so với 85.07), Flowers sang SUN397 (80%) và Caltech101 (20%) (83.18 so với 96.00). Kết quả "router đúng 100%" ở preset `quick` (4 dataset) không còn đúng khi 11 dataset.
- **Router theo từng ảnh** (`analysis/router_persample.py`, 2000 ảnh test mỗi dataset, hàng t=1..10 trung bình): `proto` (router của repo) gửi đúng 69.0% (từng ảnh) / 75.5% (theo batch), gửi nhầm unseen sang snapshot 58.7% / 51.6%, AUROC 0.813 / 0.874. Điểm Mahalanobis (kiểu DIKI) tốt nhất: gửi đúng 92.6% / 100%, unseen gửi nhầm 27.0% / 22.0%, AUROC 0.951 / 0.996, FPR95 0.247 / 0.010. Thứ tự AUROC (từng ảnh): maha 0.951 > knn 0.911 > proto 0.813 > maxcos 0.796 > mcm 0.582. Theo batch, tỷ lệ gửi đúng cao hơn từng ảnh 6-19 điểm ở mọi điểm số, nhưng gửi nhầm unseen chỉ giảm với proto, knn, maha (maxcos và mcm không giảm), nên một phần chất lượng của router theo batch đến từ giả định một dataset mỗi batch.
- **Cần làm tiếp:** (1) chạy `pact_s42` và hai seed 2026 để áp quy tắc quyết định bằng `tools/compare_runs.py`; (2) thử Mahalanobis làm router trong run thật (cần đo Transfer/Last, vì `router_persample.py` chỉ đo chất lượng chọn task); (3) đo lại bằng hai seed.

**Kết luận:** Isolation baseline cho Last 87.64 (snapshot đúng, 3h43m); router mặc định của repo là điểm yếu lớn ở 11 dataset (mất 7.4 điểm Last, Transfer dưới zero-shot) và điểm Mahalanobis cải thiện rõ về chất lượng chọn task; so với PACT phải chờ `pact_s42`.


<!-- run:pact_results_full_20261125_1941 -->
## pact_results_full_20261125_1941

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-25 19:41:02 | GPU mỗi run: 2
- **Code:** `20261125-2125-20242ff-032555cc` (commit 20242ff, nhánh main)
- **Exit code:** tất cả 0
- **Ghi chú:** pact_s42, full, 2 GPU, seed 42, lr 1e-5 (Phát; zip gốc bị nén lồng thư mục, đã nén lại phẳng để ghi log)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| pact_s42 | configs/mtil_order1.yaml | seed=42 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=3000, seed=42 | 4h38m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| pact_s42 | 68.63 / 72.67 / 77.78 | 68.63 / 76.36 / 83.74 | 69.47 / 76.74 / 83.74 | 68.76 / 73.60 / 77.74 |

Last theo từng dataset (protocol chính):

- **pact_s42** (router): Aircraft 59.92, Caltech101 96.49, CIFAR100 74.18, DTD 72.82, EuroSAT 82.50, Flowers 77.17, Food 90.35, MNIST 58.63, OxfordPet 92.89, StanfordCars 74.85, SUN397 75.33

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): pact_s42: có [oracle_zeroshot]

Chi tiết (so với `iso_s42`, cùng seed 42, cùng lr 1e-5, cấu hình chỉ khác `pact.use_marg` / `pact.use_pa`):

- **Ràng buộc PACT đổi Last lấy Transfer.** Last (`oracle_zeroshot`) 83.74 so với 87.64 của iso (-3.90), thấp hơn ở cả 11 dataset: StanfordCars -7.5, DTD -6.5, Flowers -6.0, CIFAR100 -5.0, EuroSAT -4.5, MNIST -4.1, Aircraft -3.1, SUN397 -2.1, OxfordPet -1.9, Food -1.2, Caltech101 -1.0. Transfer (`oracle_latest`) 68.63 so với 65.82 (+2.81), cao hơn ở 10/10 dataset chưa học (EuroSAT 53.3 so với 47.0, CIFAR100 67.1 so với 62.1), nhưng vẫn dưới zero-shot 69.47 (-0.84).
- **Cơ chế:** α̃ bật ở iteration 600 (20%) ở cả 11 task (`converged`, φ luôn dưới γ = 0.1). Sau đó α khoảng 0.90-0.97 và `d_mlp` chỉ 0.002-0.01: L_pa giữ adapter sát MLP gốc trong 80% thời gian còn lại. Loss cuối task của PACT cao hơn iso ở mọi task.
- **So với paper (71.76 / 81.05 / 90.54):** còn -3.13 Transfer, -4.69 Avg, -6.80 Last. Lệch lớn nhất: StanfordCars -15.1, SUN397 -11.3, CIFAR100 -11.2 (iso cũng thấp ở ba dataset này: chưa train đủ).
- **Nhiễu seed:** iso_s42 so với iso_s2026 chênh khoảng 0.05 Last, nên -3.90 lớn hơn nhiễu khoảng 80 lần.
- **Lưu ý:** hai run dùng hai bản code khác nhau (`627d990` và `20242ff`). Commit `20242ff` chưa có trên GitHub, cần Phát push để đối chiếu. Cả hai run dùng lr 1e-5, không phải 5e-5 đã chốt sau pilot.

**Kết luận:** Last cao chủ yếu nhờ cô lập task (snapshot), không nhờ ràng buộc. Ràng buộc PACT giúp Transfer nhưng làm giảm Last. Quy tắc quyết định chính thức cần `pact_s2026`.


<!-- run:pact_results_full_20261126_0927 -->
## pact_results_full_20261126_0927

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-26 09:27:30 | GPU mỗi run: 2
- **Code:** `20261125-1441-627d990-c92d780a` (commit 627d990, nhánh main)
- **Exit code:** tất cả 0
- **Ghi chú:** BẢN TRÙNG của iso_s2026 (Thuận chạy nhầm config, định chạy pact_s2026 thay Đức Anh). Bản chính thức: `pact_results_full_20261125_1326` (Khánh).

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| iso_s2026 | configs/analysis/isolation_baseline.yaml | seed=2026 | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=3000, seed=2026 | 3h42m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| iso_s2026 | 65.81 / 72.31 / 79.75 | 65.81 / 77.20 / 87.63 | 69.47 / 78.81 / 87.63 | 66.49 / 73.31 / 79.52 |

Last theo từng dataset (protocol chính):

- **iso_s2026** (router): Aircraft 62.56, Caltech101 97.52, CIFAR100 64.90, DTD 78.99, EuroSAT 86.13, Flowers 83.20, Food 91.52, MNIST 55.43, OxfordPet 94.82, StanfordCars 82.34, SUN397 77.37

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): iso_s2026: có [oracle_zeroshot]

Chi tiết: run trùng config và seed với `iso_s2026` của Khánh (`pact_results_full_20261125_1326`), chạy lại độc lập (bản build khác: `c92d780a` so với `032555cc`, cùng commit 627d990).

- So với bản của Khánh: mọi ô của cả 4 ma trận protocol trùng khít (chênh lớn nhất 0.0).
- Thời gian: 3h42m (bản của Khánh 3h38m).

**Kết luận:** pipeline tất định. Cùng code, config và seed thì cho kết quả trùng từng bit giữa hai phiên Kaggle khác nhau. Run này không thêm thông tin seed: phân tích iso_s2026 xem ở mục `pact_results_full_20261125_1326`. Seed 2026 vẫn thiếu `pact_s2026` (Đức Anh).


<!-- run:pact_results_mid_20261126_1233 -->
## pact_results_mid_20261126_1233

- **Preset:** mid (8 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, MNIST, OxfordPet, StanfordCars)
- **Tạo lúc:** 2026-11-26 12:33:31 | GPU mỗi run: 1
- **Code:** `20261126-1548-cdfa82a+dirty-37276d26` (commit cdfa82a, nhánh analysis/phase-a-b, có thay đổi chưa commit)
- **Exit code:** tất cả 0
- **Ghi chú:** Giai đoạn B: preset mid (8 dataset, 2000 iter), lr 5e-5, seed 42, 4 run x 1 GPU, kèm giai đoạn A (reeval_branch)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| iso_base | configs/analysis/isolation_baseline.yaml | train.lr=5e-5 seed=42 | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=2000, seed=42 | 2h19m |
| pact_base | configs/mtil_order1.yaml | train.lr=5e-5 seed=42 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=2000, seed=42 | 3h22m |
| pact_late | configs/mtil_order1.yaml | train.lr=5e-5 seed=42 pact.schedule.warmup_frac=0.6 pact.schedule.latest_frac=0.8 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=2000, seed=42 | 2h51m |
| pact_softpa | configs/mtil_order1.yaml | train.lr=5e-5 seed=42 pact.pa_sigma_rel=0.05 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=2000, seed=42 | 3h21m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| iso_base | 59.72 / 67.36 / 80.44 | 59.72 / 74.16 / 89.28 | 67.32 / 77.85 / 89.28 | 60.31 / 68.97 / 78.51 |
| pact_base | 66.69 / 71.83 / 79.76 | 66.69 / 75.52 / 85.38 | 67.32 / 75.75 / 85.38 | 66.79 / 72.56 / 79.19 |
| pact_late | 65.86 / 70.69 / 78.54 | 65.86 / 74.78 / 84.96 | 67.32 / 75.39 / 84.96 | 66.06 / 71.66 / 78.44 |
| pact_softpa | 65.18 / 71.00 / 80.08 | 65.18 / 75.73 / 87.09 | 67.32 / 76.75 / 87.09 | 65.60 / 72.30 / 79.73 |

Last theo từng dataset (protocol chính):

- **iso_base** (router): Aircraft 66.05, Caltech101 97.81, CIFAR100 51.00, DTD 80.27, EuroSAT 98.85, MNIST 50.35, OxfordPet 95.45, StanfordCars 88.30
- **pact_base** (router): Aircraft 61.15, Caltech101 96.95, CIFAR100 68.05, DTD 76.70, EuroSAT 94.85, MNIST 61.80, OxfordPet 93.15, StanfordCars 80.85
- **pact_late** (router): Aircraft 60.85, Caltech101 96.72, CIFAR100 67.50, DTD 73.83, EuroSAT 93.45, MNIST 58.30, OxfordPet 93.30, StanfordCars 83.55
- **pact_softpa** (router): Aircraft 64.05, Caltech101 97.41, CIFAR100 65.40, DTD 77.18, EuroSAT 96.85, MNIST 59.45, OxfordPet 93.75, StanfordCars 83.75

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): iso_base: có [oracle_zeroshot]; pact_base: có [oracle_zeroshot]; pact_late: có [oracle_zeroshot]; pact_softpa: có [oracle_zeroshot]

Chi tiết (preset mid: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, MNIST, OxfordPet, StanfordCars; 2000 iter/task; lr 5e-5; seed 42; zero-shot Transfer trên thứ tự này 67.32). Mỗi run đổi một yếu tố so với `pact_base`:

| run | Transfer (oracle_latest) | Last (oracle_zeroshot) | Δ Transfer | Δ Last | α̃ bật ở | thời gian |
|---|---|---|---|---|---|---|
| pact_base | 66.69 | 85.38 | – | – | it 400 (20%) | 202 phút |
| iso_base | 59.72 | 89.28 | -6.97 | +3.90 | – | 139 phút |
| pact_late (warmup 0.6, latest 0.8) | 65.86 | 84.96 | -0.83 | -0.43 | it 1200 (60%) | 172 phút |
| pact_softpa (pa_sigma_rel 0.05) | 65.18 | 87.09 | -1.50 | +1.70 | it 400 (20%) | 202 phút |

- **Ở lr 5e-5, bỏ ràng buộc làm Transfer sụp:** iso EuroSAT 40.7 (zero-shot 55.2), StanfordCars 49.1 (64.3), CIFAR100 57.7 (69.3). Ràng buộc PACT cần thiết cho Transfer, đánh đổi khoảng 3.9 Last.
- **`pact_late`:** kém hơn baseline ở cả Transfer và Last: bật α̃ muộn không giúp. Loại.
- **`pact_softpa`:** dịch dọc đường đánh đổi (+1.7 Last, -1.5 Transfer), Avg gần như bằng (75.73 so với 75.52). Không trội hơn baseline.
- **Giai đoạn A (`reeval_branch`, 2000 ảnh test/dataset), Last / Transfer (oracle_latest):**
  - pact_base: cond/cond 85.38 / 66.69; marg/marg 87.00 / 64.08; marg/cond 83.97 / 64.08.
  - pact_softpa: cond/cond 87.09 / 65.18; marg/marg 88.61 / 58.19.
  - pact_late: cond/cond 84.96 / 65.86; marg/marg 87.17 / 63.26.
  - iso_base: cond/cond 89.28 / 59.72; marg/marg 35.54 / 37.02 (iso không train adapter đứng riêng nên marg sụp: đúng dự đoán).
  - Đường chéo pact_base, cond -> marg: CIFAR100 81.6 -> 85.0, EuroSAT 94.8 -> 97.7, MNIST 97.8 -> 99.2, OxfordPet 93.2 -> 94.6, StanfordCars 80.8 -> 84.8; Aircraft, Caltech101, DTD gần như không đổi.
- **Pha loãng adapter được xác nhận:** nhánh cond (trung bình adapter 1..k) làm giảm điểm task đã học, nhất là các task sau. Kết hợp "đã học dùng marg, chưa học dùng cond" (protocol mới `oracle_hybrid`, commit `38e87dc`): pact_base Last 87.0 / Transfer 66.7; pact_softpa Last 88.6 / Transfer 65.2 (gần Last của iso 89.3, Transfer cao hơn iso 5.5).
- **Code:** build `cdfa82a+dirty` (working tree còn `experiments/LOG.md`, `tools/log_run.py` chưa commit; không ảnh hưởng train).

**Kết luận:** (1) Ở lr 5e-5 ràng buộc PACT cần thiết cho Transfer; bật muộn không giúp, nới L_pa chỉ đổi Transfer lấy Last. (2) Nhánh cond pha loãng adapter: đánh giá dữ liệu đã học bằng nhánh marg tăng khoảng 1.6 Last mà không đổi Transfer; đây có thể là cách đánh giá của paper. (3) Mọi cấu hình vẫn có Transfer dưới zero-shot, chưa giải thích được. Bước tiếp: chạy full với lr 5e-5 và `oracle_hybrid` (lượt sau).


<!-- run:pact_results_full_20261125_1326 -->
## pact_results_full_20261125_1326

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-25 13:26:21 | GPU mỗi run: 2
- **Code:** `20261125-1633-627d990-032555cc` (commit 627d990, nhánh main)
- **Exit code:** tất cả 0
- **Ghi chú:** iso_s2026, full, 2 GPU, seed 2026, lr 1e-5 (Khánh chạy; bản chính thức của iso_s2026)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| iso_s2026 | configs/analysis/isolation_baseline.yaml | seed=2026 | pact.use_marg=False, pact.use_pa=False, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=3000, seed=2026 | 3h38m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| iso_s2026 | 65.81 / 72.31 / 79.75 | 65.81 / 77.20 / 87.63 | 69.47 / 78.81 / 87.63 | 66.49 / 73.31 / 79.52 |

Last theo từng dataset (protocol chính):

- **iso_s2026** (router): Aircraft 62.56, Caltech101 97.52, CIFAR100 64.90, DTD 78.99, EuroSAT 86.13, Flowers 83.20, Food 91.52, MNIST 55.43, OxfordPet 94.82, StanfordCars 82.34, SUN397 77.37

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): iso_s2026: có [oracle_zeroshot]

Chi tiết (so với `iso_s42`, cấu hình chỉ khác seed, cùng commit `627d990`):

- Seed có tác dụng thật: loss theo iteration khác nhau ở mọi bản ghi.
- Kết quả gần như trùng: `oracle_zeroshot` Last 87.64 -> 87.63, Avg 78.84 -> 78.81; `oracle_latest` Transfer 65.82 -> 65.81. Chênh từng dataset tối đa 0.45 (Aircraft). `router` Last 80.20 -> 79.52 (-0.68), vì router theo batch nhạy hơn.
- Bảng 8: cột đã học bất biến (OK). Zero-shot trùng với seed 42.
- Thời gian: 3h38m train+eval trên 2 GPU (seed 42: 3h43m).
- Bản chạy lại của Thuận (`pact_results_full_20261126_0927`, cùng config và seed) trùng khít từng ô, tức là pipeline tất định.

**Kết luận:** nhiễu do seed ở preset full khoảng ±0.05 điểm (Last, Transfer), nhỏ hơn mọi khác biệt giữa các cấu hình đang so. Còn thiếu `pact_s2026` để `tools/compare_runs.py` kết luận với 2 seed.


<!-- run:pact_results_full_20261126_1523 -->
## pact_results_full_20261126_1523

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-26 15:23:53 | GPU mỗi run: 4
- **Code:** `20261125-1441-627d990-c92d780a` (commit 627d990, nhánh main)
- **Exit code:** tất cả 0
- **Ghi chú:** pact_s2026, full, 2 GPU, seed 2026, lr 1e-5 (Đức Anh)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| pact_s2026 | configs/mtil_order1.yaml | seed=2026 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=1e-05, train.iterations=3000, seed=2026 | 2h32m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|
| pact_s2026 | 68.64 / 72.66 / 77.80 | 68.64 / 76.37 / 83.75 | 69.47 / 76.75 / 83.75 | 68.79 / 73.66 / 77.39 |

Last theo từng dataset (protocol chính):

- **pact_s2026** (router): Aircraft 59.92, Caltech101 96.54, CIFAR100 69.75, DTD 73.09, EuroSAT 82.57, Flowers 77.18, Food 90.37, MNIST 58.72, OxfordPet 93.05, StanfordCars 74.72, SUN397 75.41

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): pact_s2026: có [oracle_zeroshot]

Chi tiết (so với `iso_s2026` cùng seed và `pact_s42`; chạy 4 GPU, các run kia 2 GPU):

- **PACT so với iso, seed 2026:** `oracle_zeroshot` Last 83.75 so với 87.63 (-3.87); `oracle_latest` Transfer 68.64 so với 65.81 (+2.83). Seed 42 cho -3.90 và +2.81. PACT thấp hơn iso ở cả 11 dataset; chênh nhiều nhất ở StanfordCars (-7.6), DTD (-6.4), Flowers (-5.9), CIFAR100 (-5.0).
- **Nhiễu seed của PACT:** so với `pact_s42`, Last 83.74 -> 83.75 và Transfer 68.63 -> 68.64; mỗi dataset chênh tối đa 0.13 (StanfordCars).
- **Bản code:** `pact_s42` chạy commit 20242ff, các run còn lại chạy 627d990 (hai hash build). Kết quả PACT trùng nhau giữa hai bản code, nên cảnh báo của `compare_runs` không ảnh hưởng kết luận.
- **Thời gian:** 2h32m train+eval trên 4 GPU, so với 4h38m trên 2 GPU (`pact_s42`), nhanh hơn khoảng 1.8 lần.
- Transfer vẫn thấp hơn zero-shot 69.47 khoảng 0.8 điểm.

**Kết luận:** đã đủ 4 run full (PACT và iso, seed 42 và 2026). Theo `tools/compare_runs.py`, chênh Last trung bình -3.88 và cùng dấu ở cả 2 seed, nên ràng buộc PACT (L_marg, L_pa) **không đóng góp vào Last** ở cấu hình lr 1e-5: nó đổi Last lấy Transfer (+2.8) nhưng chưa vượt được zero-shot. Bước tiếp theo: chạy full với lr 5e-5 và oracle_hybrid, vì giai đoạn B cho thấy hai thay đổi này thu hẹp khoảng cách.


<!-- run:pact_results_full_20261127_1253 -->
## pact_results_full_20261127_1253

- **Preset:** full (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)
- **Tạo lúc:** 2026-11-27 12:53:13 | GPU mỗi run: 4
- **Code:** `20261127-1646-0a25c79+dirty-524af0df` (commit 0a25c79, nhánh analysis/phase-a-b, có thay đổi chưa commit)
- **Exit code:** tất cả 0
- **Ghi chú:** pact_lr5e5_s42, full, 4 GPU L4, seed 42, lr 5e-5, có oracle_hybrid (run full nền mới, Thuận; zip gốc lồng thư mục, đã nén lại phẳng để ghi log)

### Phương pháp và thời gian

| run | config | override thêm | thay đổi chính | thời gian train+eval |
|---|---|---|---|---|
| pact_lr5e5_s42 | configs/mtil_order1.yaml | seed=42 train.lr=5e-5 | pact.use_marg=True, pact.use_pa=True, adapter.gate_mode=layerwise, head.type=learned, train.lr=5e-05, train.iterations=3000, seed=42 | 2h36m |

### Kết quả (Transfer / Avg / Last)

| run | latest | oracle_hybrid | oracle_latest | oracle_zeroshot | router |
|---|---|---|---|---|---|
| pact_lr5e5_s42 | 68.74 / 74.58 / 82.20 | 68.74 / 78.30 / 88.03 | 68.74 / 77.75 / 86.83 | 69.47 / 78.12 / 86.83 | 68.94 / 74.90 / 81.73 |

Last theo từng dataset (protocol chính):

- **pact_lr5e5_s42** (router): Aircraft 62.17, Caltech101 97.06, CIFAR100 77.03, DTD 77.29, EuroSAT 89.20, Flowers 87.01, Food 91.29, MNIST 60.28, OxfordPet 94.14, StanfordCars 83.60, SUN397 79.92

Cột đã học bất biến theo Bảng 8 (kiểm tra trên oracle_zeroshot; router có thể đổi quyết định giữa các hàng): pact_lr5e5_s42: có [oracle_zeroshot]

Chi tiết (so với `pact_s42` / `pact_s2026` lr 1e-5 và `iso_s42` / `iso_s2026` lr 1e-5, cùng code trừ lr):

- **lr 5e-5 tăng Last, giữ Transfer:** `oracle_zeroshot` Last 86.83 so với 83.74 (+3.09); `oracle_latest` Transfer 68.74 so với 68.63 (+0.11). Kết quả giai đoạn B (mid) chuyển được sang full.
- **`oracle_hybrid` thêm +1.20 Last (88.03) mà không đổi Transfer.** Tăng nhiều ở các task sau: StanfordCars 83.60 → 87.64, CIFAR100 82.28 → 85.58, EuroSAT 94.93 → 97.61, MNIST 97.20 → 98.91, Food 91.29 → 92.00. Xác nhận pha loãng adapter ở nhánh cond trên preset full.
- **So với isolation lr 1e-5 (87.64 / 65.82):** hybrid cao hơn +0.39 Last và +2.92 Transfer. Chưa so được với isolation ở lr 5e-5 (chưa chạy full); ở preset mid, iso lr 5e-5 vẫn hơn pact hybrid 2.3 Last nhưng Transfer thấp hơn 7.0.
- **So với paper (71.76 / 81.05 / 90.54), dùng hybrid:** còn −3.02 Transfer, −2.75 Avg, −2.51 Last (trước đây −6.80 Last). Lệch Last lớn nhất: SUN397 −5.91, CIFAR100 −5.63, DTD −3.72, Aircraft −3.56, Flowers −3.11, StanfordCars −2.26; Caltech101 +0.52.
- **Transfer vẫn dưới zero-shot 69.47 (−0.73).** Thấp nhất ở EuroSAT 52.86 (zero-shot 55.19), SUN397 63.36 (65.10), StanfordCars 63.04 (64.76); cao hơn ở MNIST (+1.55) và DTD (+0.78).
- **Router của repo vẫn mất 5.1 Last** (81.73 so với 86.83): MNIST 60.28 (oracle 97.20), Flowers 87.01 (95.46), EuroSAT 89.20 (94.93), CIFAR100 77.03 (82.28).
- **Thời gian:** 2h36m trên 4×L4.
- **Thiếu `state.pt`** trong zip (chỉ có trong tab Output của Kaggle), cần tải riêng cho phân tích offline. Zip có `router_features.npz` (39.6 MB) cho phân tích router theo từng ảnh.

**Phân tích offline trên checkpoint (28/11, `kaggle/pact_analysis.ipynb`, 4×L4, 2000 ảnh test/dataset, 80 phút; kết quả ở `experiments/artifacts/pact_results_full_20261127_1253/pact_lr5e5_s42/analysis_20261128/`):**

- **`oracle_snapshot` (Transfer):** chọn snapshot tốt nhất cho dữ liệu chưa học (biết trước đáp án) chỉ đạt 69.89 so với zero-shot 69.43 (+0.46, trong nhiễu). Chỉ MNIST được lợi rõ (+2.95, 6/7 snapshot hơn zero-shot); DTD +0.9. Kết luận: không cách gửi dữ liệu chưa học sang snapshot nào nâng được Transfer vượt zero-shot; trần thực tế của Transfer là trả dữ liệu chưa học về CLIP (69.43), hiện mất 0.67 so với trần đó.
- **`alignment_drift`:** với text đóng băng, đặc trưng ảnh của task đã học ở snapshot cuối chỉ đạt 66.18 (gần zero-shot), còn head học được đạt 81.85 (+15.7): phần lớn Last đến từ head, adapter không kéo ảnh lại gần text. Trên dữ liệu chưa học, Transfer giảm nhiều nhất ở EuroSAT (−2.24), SUN397 (−1.92), CIFAR100 (−1.60), StanfordCars (−1.24); tăng ở MNIST (+1.86), DTD (+0.83). Cosine ảnh–text đúng lớp chỉ giảm 0.003–0.03; mức giảm độ chính xác tương quan vừa với độ giảm `pos` (r = 0.47) và `self_sim` (r = 0.48), không tương quan với modality gap (r = −0.17) hay CKA (r = 0.08). Ca nặng nhất: StanfordCars dưới snapshot 1 (chỉ adapter Aircraft) −6.8, EuroSAT dưới snapshot 2 −4.6. Chạy 77 phút, chủ yếu vì chưa bật TF32 (đã sửa sau lần chạy này).
- **`probe_distance`:** probe Gaussian cho d 0.0027–0.0064, không τ nào tạo được cả melted lẫn frozen. Token ảnh thật cho d lớn hơn khoảng 10 lần (0.016–0.08), τ = 0.05 cho α trải 0.20–0.85. Ở cả hai, adapter cách nhau xa hơn cách MLP (contrast 1.5–1.7) và không có cụm theo domain như Hình 3 (Aircraft–StanfordCars không gần nhau; Food là "láng giềng gần nhất" của MNIST, OxfordPet, StanfordCars, SUN397). Theo token của chính domain, adapter lệch khỏi MLP mạnh hơn hẳn trung bình (Flowers 0.18 so với 0.05, DTD 0.17 so với 0.04, EuroSAT 0.11 so với 0.04), trừ Food (0.029), SUN397 (0.016), Caltech101 (0.039): ba adapter gần như không học thêm, khớp với giả thuyết dataset lớn nhận quá ít epoch (SUN397 thiếu 5.9 Last so với paper).
- **`reeval_branch`:** marg/marg cho Last 87.75 (= hybrid), Transfer 66.38 (−2.4 so với cond). Dùng adapter k với head cond (marg/cond) làm Last tụt còn 82.76 (Flowers 74.2, StanfordCars 72.4): head gắn chặt với nhánh đã train cùng.
- **`router_persample`:** như phân tích trước (cùng đặc trưng): Mahalanobis chọn đúng 92.6% ảnh đã học, nhưng vẫn gửi nhầm 75% ảnh Flowers và 62.5% ảnh MNIST chưa học sang snapshot. Kết hợp với `oracle_snapshot`: mọi lần gửi nhầm dữ liệu chưa học đều là thiệt.

**Kết luận:** đây là mốc so sánh mới của nhóm ở preset full: **68.74 / 78.30 / 88.03** (`oracle_hybrid`), 1 seed (nhiễu seed ở full khoảng ±0.05). lr 5e-5 và đánh giá task đã học qua nhánh marg thu hẹp khoảng cách Last với paper từ −6.8 xuống −2.5. Còn hai khoảng cách chưa giải thích được: Transfer dưới zero-shot và router mất 5.1 Last. Muốn kết luận ràng buộc PACT có lợi hay không ở lr 5e-5 cần thêm `iso_lr5e5_s42` full.

