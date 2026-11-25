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
