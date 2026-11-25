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

**Kết luận:** _(điền tay)_

