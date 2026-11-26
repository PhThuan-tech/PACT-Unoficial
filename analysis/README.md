# Phân tích (nhánh `analysis/isolation-and-router`)

Các thí nghiệm phụ để đọc đúng kết quả PACT. Không đổi mặc định nào của `pact/`.

## Hướng 2: baseline cô lập theo task

Câu hỏi: Last cao trong Bảng 8 đến từ ràng buộc PACT (L_marg, L_pa, α̃), hay chỉ từ việc mỗi task được đánh giá bằng snapshot riêng?

- Config: `configs/analysis/isolation_baseline.yaml`. Giống `pact_paper`, chỉ tắt L_marg và L_pa.
- Chạy cùng preset và seed với `pact_paper`. Trong notebook, thêm vào `EXPERIMENTS`:
  `("isolation", "configs/analysis/isolation_baseline.yaml", [])`.
- So hai run:
  - Last của `oracle_zeroshot` (đánh giá theo snapshot).
  - Transfer của `oracle_latest` (mô hình mới nhất trên dataset chưa học).

Đọc kết quả:
- Chênh trong mức nhiễu giữa các seed: ràng buộc PACT gần như không đóng góp vào Last.
- `pact_paper` cao hơn rõ: ràng buộc có tác dụng.

## Hướng 3: router offline

Câu hỏi: không có task-ID thì router chọn đúng task được bao nhiêu, và ngưỡng ảnh hưởng thế nào?

```bash
python analysis/router_offline.py outputs/pact_paper/evaluator.pt
```

`evaluator.pt` nhỏ (vài MB), do `run_mtil.py` ghi sau mỗi task, và luôn có trong zip kết quả của notebook. Script cũng đọc được `state.pt`. Script không cần GPU. Nó in ra:
1. Tỉ lệ batch được gửi đúng task, về CLIP, hoặc sang task khác, theo từng mức dịch ngưỡng `delta` (`delta > 0` là chặt hơn).
2. Transfer, Avg, Last ứng với từng `delta`. Một mức `delta` chỉ tính được điểm khi các tổ hợp (dataset, snapshot, head) mà router chọn đã được đánh giá trong run. Cột `missing` cho biết phần thiếu.
3. Ma trận nhầm lẫn ở `delta = 0`.
4. Kiểm tra: ở `delta = 0`, kết quả phải trùng protocol `router` của run.

Kết quả ghi vào `router_offline.json` và `router_offline.png`, đặt cạnh file đầu vào.

Test: `python tests/router_offline_test.py`.

Lưu ý: chỉ các run được chạy bằng code của nhánh này mới có `evaluator.pt`. Run chạy bằng code `main` hiện tại thì phải dùng `state.pt` (đặt `INCLUDE_CHECKPOINTS = True` ở ô cuối notebook).

## Hướng 4: router theo từng ảnh (`router_persample.py`)

Câu hỏi: router theo batch (đúng 100% với dữ liệu đã học) tốt đến đâu khi bỏ giả định "mỗi batch một dataset"?

```bash
# GPU, một lần (notebook ô 8b tự chạy sau khi các run xong): đặc trưng CLIP đóng băng -> .npz
python analysis/router_persample.py extract --config configs/mtil_order1.yaml --out router_features.npz --set <như run>
# CPU, vài giây
python analysis/router_persample.py analyze router_features.npz --alpha 0.05
```

So 5 cách tính điểm (proto, knn, maha, maxcos, mcm), chế độ `sample` và `batch`: tỷ lệ gửi đúng, gửi nhầm unseen, AUROC, FPR95, từng dataset. Test: `python tests/router_persample_test.py`.

## Hướng 5: oracle best snapshot (`oracle_snapshot.py`)

Câu hỏi: có snapshot nào giúp dataset chưa học tốt hơn CLIP đóng băng không? Nếu không, không router nào nâng được Transfer vượt zero-shot.

```bash
python analysis/oracle_snapshot.py --run-dir outputs/pact_s42 --max-samples 2000 --set data_root=... model.pretrained=...
```

Cần `state.pt` và `config.yaml` của run (GPU). In bảng zero-shot / latest / oracle_best theo từng dataset. Test lõi: `python tests/oracle_snapshot_test.py`.

## So sánh PACT và isolation (`tools/compare_runs.py`)

Đọc `experiments/runs/*.json` (do `tools/log_run.py` ghi), ghép cặp PACT / isolation cùng preset và seed, tính chênh Last của `oracle_zeroshot`, Transfer so với zero-shot, rồi áp quy tắc quyết định (dưới +0.5 và cùng dấu: ràng buộc không đóng góp; trên +1.0 và cùng dấu: có giá trị; còn lại: chưa kết luận). Cảnh báo nếu hai run dùng bản code khác nhau. Test: `python tests/compare_runs_test.py`.

## Giai đoạn A: đánh giá lại theo nhánh lúc test (`reeval_branch.py`)

Câu hỏi: nhánh cond (trung bình adapter 1..k) có "pha loãng" adapter của task đang đánh giá không? Đây có phải nguyên nhân khiến các task cuối (StanfordCars, SUN397) lệch xa paper nhất?

```bash
python analysis/reeval_branch.py --run-dir outputs/pact_s42 --max-samples 2000 --set data_root=... model.pretrained=...
```

Script cần `state.pt` và `config.yaml` của run, và chạy trên GPU (khoảng 10 phút trên một L4). Nó đánh giá lại ba biến thể:
- `cond/cond`: cách run đã báo cáo.
- `marg/marg`: chỉ adapter k, head marg.
- `marg/cond`: chỉ adapter k, head cond. Biến thể này dành cho iso, vì iso không train head marg.

Kết quả gồm Transfer / Avg / Last (`oracle_zeroshot`, `oracle_latest`) và độ chính xác ngay sau khi học từng task, ghi vào `<run>/reeval_branch.json`. Đặt `--max-samples 0` thì `cond/cond` trùng đúng `results.json` của run. Test: `python tests/reeval_branch_test.py`.

Với `pact_s42` và `iso_s42` (đã chạy xong): mở một notebook Kaggle, gắn output của version đã chạy (output này có `state.pt`) cùng `pact-code` và dữ liệu, chạy ô 1–5 của notebook rồi chạy lệnh trên với `--run-dir` trỏ vào thư mục run trong `/kaggle/input/...`.

## Giai đoạn B: notebook `kaggle/pact_phase_b.ipynb`

Tạo bằng `python kaggle/make_notebook.py kaggle/pact_phase_b.ipynb --phase-b`. Lệnh `tools/prepare_upload.py` cũng chép file này vào `dist/`.

Notebook chạy preset mid (8 dataset, 2000 iteration mỗi task) với lr 5e-5 và seed 42. Có 4 run, mỗi run 1 GPU, chạy song song:

| run | so với `pact_base` |
|---|---|
| `pact_base` | baseline |
| `iso_base` | bỏ L_marg và L_pa |
| `pact_late` | α̃ bật ở 60% |
| `pact_softpa` | `pa_sigma_rel` = 0.05 |

`pact_late` phải đặt thêm `latest_frac = 0.8`. Nếu không, deadline 50% sẽ bật công tắc trước khi hết warmup.

Sau khi train xong:
- Ô 8c chạy giai đoạn A cho mọi run.
- Ô 10b in bảng: Δ Transfer và Δ Last so với `pact_base`, độ chính xác ngay sau khi học từng task, và Last theo từng nhánh.
