# Báo cáo Lab Day 2 — DeepWeeds

> Chỉ thay các mục trong ngoặc vuông bằng số liệu từ run thật và `eval.py`.

## 1. Tóm tắt

[Nêu bài toán, số thí nghiệm, cấu hình tốt nhất, macro-F1/top-1/ECE test dạng mean ± std và kết luận chính trong tối đa 10 dòng.]

## 2. Dữ liệu và thiết lập

- Dataset: DeepWeeds, 9 lớp; fold 0 nguyên bản.
- Split đã kiểm tra: train 10.501, validation 3.501, test 3.507; các giao bằng 0; tổng 17.509.
- Mất cân bằng: lớp Negative chiếm khoảng 52%; do đó macro-F1 là chỉ số chọn cấu hình.
- EDA đã tạo: `eda/class_distribution_fold0.png`, `eda/sample_images_fold0.png` và `eda/class_counts_fold0.csv`.
- Phần cứng, phiên bản thư viện, seed: [điền].
- Pipeline sanity checks: initial loss [điền], overfit batch [điền], augmentation [tham chiếu hình].

## 3. So sánh backbone

[Chèn bảng trích từ sheet Backbones: B01–B05, params, GMAC, macro-F1/top-1 validation, thời gian train và latency.]

[Nêu backbone được chọn và đánh đổi accuracy/latency/model size.]

## 4. Công thức huấn luyện

[Chèn bảng ablation T00–T07 và run kết hợp. Mỗi nhận xét phải đối chiếu delta với độ lệch chuẩn của các seed ở vòng final.]

## 5. Suy luận và độ trễ

[Chèn bảng I00–I04. Báo p50/p95/p99 batch 1, GPU, dtype, warmup=10 và số lần đo=50.]

[Báo ECE trước/sau temperature scaling, trong đó nhiệt độ được fit trên validation. Chèn scatter macro-F1 validation và p95 latency.]

## 6. Chung kết và phân tích lỗi

- Baseline T00/I00 và final F01 chạy seed [0, 1, 2].
- Kết quả test từ `eval.py`: macro-F1 [điền] ± [điền], top-1 [điền] ± [điền], ECE [điền].
- Chèn confusion matrix, bảng precision/recall/F1 theo lớp và nêu cụ thể Chinee apple, Snake weed.
- Xem một số ảnh đoán sai: [mô tả giả thuyết nguyên nhân].

## 7. Kết luận và khuyến nghị triển khai

[Cấu hình offline tốt nhất là gì? Tốt hơn baseline bao nhiêu? Chênh lệch có lớn hơn std không?]

[Nêu một cấu hình p95 ≤ 100 ms hoặc giải thích vì sao chưa đạt; đề xuất cấu hình cho robot.] 

## 8. Hạn chế và bước tiếp theo

- Kết quả final chỉ dùng [điền] seed và [điền] fold.
- Split ngẫu nhiên không theo địa điểm có thể làm test lạc quan hơn triển khai ngoài thực địa.
- [Nêu giới hạn GPU, thí nghiệm thất bại, rủi ro lệch mùa/ánh sáng/miền dữ liệu.]
