# So sánh PACT và isolation

Sinh bởi `tools/compare_runs.py` từ `experiments/runs/*.json`.

## Preset `full` (11 dataset: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397)


| seed | PACT Last | iso Last | chênh Last | PACT Transfer (oracle_latest) | iso Transfer (oracle_latest) | zero-shot Transfer | PACT router Last | iso router Last |
|---|---|---|---|---|---|---|---|---|
| 42 | 83.74 | 87.64 | -3.90 | 68.63 | 65.82 | 69.47 | 77.74 | 80.20 |
|  | CẢNH BÁO seed 42: hai run dùng bản code khác nhau (032555cc và c92d780a), so sánh có thể không công bằng | | | | | | | |
| 2026 | 83.75 | 87.63 | -3.87 | 68.64 | 65.81 | 69.47 | 77.39 | 79.52 |
|  | CẢNH BÁO seed 2026: hai run dùng bản code khác nhau (c92d780a và 032555cc), so sánh có thể không công bằng | | | | | | | |

**Quy tắc quyết định:** Ràng buộc PACT không đóng góp vào Last (chênh trung bình -3.88 < +0.5, cùng dấu ở mọi seed): đi nhánh router / đánh giá trung thực (A1, A2, B1, B2)

Transfer so với zero-shot (điểm > 0 nghĩa là vượt zero-shot):

- seed 42, PACT: oracle_latest 68.63 (-0.84); router 68.76 (-0.70); latest 68.63 (-0.84); zero-shot 69.47
- seed 42, isolation: oracle_latest 65.82 (-3.64); router 66.53 (-2.94); latest 65.82 (-3.64); zero-shot 69.47
- seed 2026, PACT: oracle_latest 68.64 (-0.83); router 68.79 (-0.68); latest 68.64 (-0.83); zero-shot 69.47
- seed 2026, isolation: oracle_latest 65.81 (-3.66); router 66.49 (-2.98); latest 65.81 (-3.66); zero-shot 69.47

Nguồn: pact seed 42: pact_results_full_20261125_1941 (code 20261125-2125-20242ff-032555cc), iso seed 42: pact_results_full_20261125_1155 (code 20261125-1441-627d990-c92d780a), pact seed 2026: pact_results_full_20261126_1523 (code 20261125-1441-627d990-c92d780a), iso seed 2026: pact_results_full_20261125_1326 (code 20261125-1633-627d990-032555cc)

