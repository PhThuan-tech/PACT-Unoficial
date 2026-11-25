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
