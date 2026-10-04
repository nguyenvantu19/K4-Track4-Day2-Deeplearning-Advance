# Kế hoạch chạy thí nghiệm DeepWeeds

Chạy theo đúng thứ tự dưới đây. Chỉ dùng validation để chọn cấu hình; không bật `RUN_FINAL_TEST` trước bước 4.

## Bước 0 — Pipeline

- [x] Fold 0: train 10.501, val 3.501, test 3.507; giao các tập bằng 0; hợp bằng 17.509.
- [ ] Chạy notebook trên GPU để lưu biểu đồ EDA, augmentation, initial loss và overfit một batch.

## Bước 1 — Backbones

Giữ nguyên baseline `T00`, seed 0, 12 epoch. Bật `RUN_BACKBONE_SWEEP=True`.

| exp_id | backbone | Nhóm yêu cầu |
|---|---|---|
| B01 | resnet50 | ResNet |
| B02 | resnext50 | ResNeXt |
| B03 | convnext_tiny | ConvNeXt |
| B04 | deit_small | Transformer |
| B05 | mobilenetv3 | Mạng nhẹ |

Chọn backbone đi tiếp từ macro-F1 validation và đánh đổi params/GMAC/latency, không chỉ điểm cao nhất.

## Bước 2 — Ablation train

Chạy trên backbone được chọn, seed 0. Mỗi thí nghiệm chỉ thay đổi một yếu tố so với `T00`; bật `RUN_TRAINING_ABLATION=True`.

| exp_id | Trục | Thay đổi |
|---|---|---|
| T00 | Baseline | finetune, basic aug, CE |
| T01 | Khởi tạo | scratch |
| T02 | Khởi tạo | frozen backbone |
| T03 | Augmentation | color jitter |
| T04 | Augmentation | CutMix, alpha=1.0 |
| T05 | Loss | label smoothing=0.1 |
| T06 | Loss | focal, gamma=2.0 |
| T07 | Regularization | EMA decay=0.999 |

Sau đó, thêm một run kết hợp hai lựa chọn tốt nhất (ví dụ augmentation + loss) nếu nó chưa nằm trong bảng.

## Bước 3 — Inference (validation)

Bật `RUN_INFERENCE_STUDY=True` sau khi có checkpoint. Notebook đã có năm dòng: `I00` 1-view, `I01` TTA flip/probability, `I02` TTA flip/logit, `I03` temperature scaling, `I04` TTA đa tỉ lệ. Đo latency riêng cho từng dòng, batch 1, warmup 10 và 50 lần đo.

## Bước 4 — Chung kết

1. Ghi cố định backbone, training override và inference được chọn từ validation.
2. Chạy `T00/I00` và `F01` với seed 0, 1, 2.
3. Bật `RUN_FINAL_TEST=True` đúng một lần khi đã chốt cấu hình.
4. Chạy `eval.py score` và `eval.py grade`; sao chép các số được tính lại vào `results.xlsx` và `report.md`.

## Bước 5 — Nộp

- [ ] `results.xlsx` có số từ các run thật.
- [ ] `curves/` có một biểu đồ riêng cho mọi B/T/F run.
- [ ] `predictions/` có T00 và F01 cho đủ ba seed.
- [ ] `report.md` được điền và xuất PDF nếu giảng viên yêu cầu.
- [ ] Sao chép `starter/` thành `code/` sau khi chốt code.
- [ ] Điền link Kaggle/Colab vào `SUBMISSION_README.md`.
