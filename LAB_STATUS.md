# Trạng thái Lab Day 2

Được kiểm tra ngày 2026-10-04 trên máy không có CUDA.

| Hạng mục | Trạng thái | Bằng chứng hoặc phần còn thiếu |
|---|---|---|
| Fold 0 và chống leakage | Hoàn thành | train 10.501, val 3.501, test 3.507; giao bằng 0; hợp 17.509 |
| EDA phân bố lớp và ảnh mẫu | Hoàn thành | `eda/class_distribution_fold0.png`, `eda/sample_images_fold0.png`, `eda/class_counts_fold0.csv` |
| Pipeline/code khung | Hoàn thành để chạy GPU | `code/` và `starter/`; notebook có các bước thực nghiệm và `eval.py` |
| So sánh 5 backbone | Chưa chạy | Không có `runs/`, `curves/` hay chỉ số validation |
| Ablation ≥3 trục + run kết hợp | Chưa chạy | Có plan T00–T07, chưa có output thực nghiệm |
| Inference ≥4 phương pháp, ECE, latency | Chưa chạy | Có code I00–I04; chưa có checkpoint/GPU để đo |
| Chung kết T00/I00 và F01, 3 seed | Chưa chạy | Không có `predictions/` hay `eval_out/` |
| `eval.py score` và `eval.py grade` | Chưa chạy | Cần predictions test hợp lệ |
| `results.xlsx` | Chỉ là template | Đủ 7 sheet nhưng chưa chứa bất kỳ số liệu run thật nào |
| `curves/`, `predictions/`, `eval_out/` | Chưa có | Sinh ra sau khi chạy notebook GPU |
| `report.md` | Khung sẵn sàng | Cần chèn bảng/kết quả chạy thật, confusion matrix và phân tích lỗi |
| README tái lập | Khung sẵn sàng | Cần thay placeholder bằng link Kaggle/Colab |

## Việc còn phải chạy trên GPU

1. Mở `code/lab_day2.ipynb` trong Kaggle hoặc Colab có CUDA.
2. Chạy Bước 0, sau đó bật lần lượt `RUN_BACKBONE_SWEEP`, `RUN_TRAINING_ABLATION` và `RUN_INFERENCE_STUDY`.
3. Chọn cấu hình bằng validation, rồi chỉ bật `RUN_FINAL_TEST` một lần để chạy T00/I00 và F01 ở seed 0, 1, 2.
4. Gửi lại `predictions/`, `eval_out/`, `runs/` hoặc link notebook. Khi đó có thể hoàn thiện các sheet, biểu đồ, report và kiểm tra điều kiện nộp.
