# Hướng dẫn tái lập Lab Day 2

## Môi trường

- Python 3.10+; PyTorch có CUDA; `timm`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`.
- Dataset DeepWeeds: `data/images/` và `data/labels/` với các CSV fold 0 nguyên bản.
- Notebook chạy lại: **[điền link Kaggle hoặc Colab sau khi upload]**.

## Thứ tự chạy

1. Mở `starter/lab_day2.ipynb` từ thư mục gốc repo trên một GPU.
2. Chạy Bước 0, xem EDA, augmentation và overfit batch trước khi bật các cờ `RUN_*`.
3. Bật từng cờ theo [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): backbone, ablation, inference, rồi final test.
4. Chỉ bật `RUN_FINAL_TEST` sau khi chọn mọi cấu hình bằng validation.
5. Chạy `python eval.py score ...` và `python eval.py grade ...` được tạo ở cuối notebook.
6. Đối chiếu `eval_out/` với `results.xlsx` và hoàn thiện `report.md`.

## Quy tắc tái lập

- CSV fold 0 không được sửa, và không gộp validation vào train.
- Seed chung kết: 0, 1, 2. Test chỉ chạy một lần mỗi seed.
- Mọi file dự đoán final và baseline phải nằm trong `predictions/` và đúng định dạng của `eval.py`.
