# Proposal Gap Analysis and Improvement Plan

## Quy tắc hoàn thành

Dự án chỉ được đánh dấu hoàn thành khi toàn bộ pipeline đã chạy trên dữ liệu đầy đủ,
checkpoint và metric được lưu bằng JSON, các kết quả được đo lại độc lập, và mọi sai
lệch so với proposal được ghi rõ. Expected output là mục tiêu nghiên cứu, không phải
số liệu được phép điền thủ công.

## KPI cần đối chiếu

| Nhóm | Mục tiêu proposal | Kết quả đo thật | Trạng thái |
|---|---:|---:|---|
| Enhancer parameters | <= 0.35 M | Chờ benchmark cuối | Đang chạy |
| Enhancer overhead | <= 6 ms | Chờ benchmark CUDA/TensorRT | Đang chạy |
| End-to-end latency p50 | <= 36 ms | Chờ benchmark | Đang chạy |
| End-to-end latency p95 | <= 42 ms | Chờ benchmark | Đang chạy |
| Clean mAP50-95 degradation | <= 0.005 | Chờ đánh giá B0/LADD-UAV | Đang chạy |
| Low-light/cross-domain detection | Cải thiện so với baseline | Chờ Stage C/D và evaluation | Đang chạy |

## Kết quả production đã khóa

- Stage B enhancer: 40/40 epoch trên toàn bộ LOL-v1.
- Best Stage B tại epoch 25: PSNR 16.1580 dB, SSIM 0.7573.
- Detector B2: 100/100 epoch trên toàn bộ VisDrone-LLMix, seed 3407.
- Best B2 tại epoch 88: mAP50 0.2434, mAP50-95 0.13586.
- Stage C: đang chạy trên toàn bộ LLMix bằng GPU.

## Vòng xử lý nếu chưa đạt KPI

### 1. Chất lượng detection low-light chưa đạt

1. Kiểm tra metric theo LL1/LL2/LL3 và theo kích thước tiny/small để xác định miền lỗi.
2. Tune loss theo từng thành phần thay vì tăng đồng loạt: detection, NWD, heatmap,
   foreground edge và reconstruction.
3. Dùng curriculum LL1 -> LL2 -> LL3 rồi fine-tune lại trên LLMix để giảm sốc miền.
4. Thử P2 detection head cho vật thể nhỏ; giữ nhánh này chỉ khi lợi ích mAP bù được
   latency và số tham số.
5. Thử scale-aware NWD hoặc tăng trọng số riêng cho object có cạnh dưới 16 px.
6. Hard-example replay từ các frame false negative có illumination thấp nhất.

### 2. Clean mAP giảm quá 0.005

1. Tăng identity loss trên 20% ảnh clean trong LLMix.
2. Thêm clean-teacher distillation giữa B0 và LADD-UAV.
3. Hiệu chỉnh gate threshold để bypass mạnh hơn trên ảnh đủ sáng.
4. Freeze detector backbone ở các vòng đầu Stage C và chỉ mở dần neck/head.
5. Tăng tỷ lệ clean có kiểm soát, nhưng giữ một ablation riêng để báo cáo công bằng.

### 3. Enhancer hoặc latency vượt ngân sách

1. Profile riêng gate, enhancer và detector bằng CUDA events sau warm-up.
2. Giảm fusion blocks hoặc channel width theo thứ tự 96 -> 80 -> 64.
3. Fuse Conv/BatchNorm, dùng FP16 và export ONNX/TensorRT với shape cố định 640.
4. Dùng hard gate khi inference và bỏ hoàn toàn enhancer ở nhánh bypass.
5. Cache illumination statistics và tránh đồng bộ CPU-GPU trong `drop_engine`.
6. Chỉ chấp nhận quantization nếu mAP50-95 giảm không vượt ngưỡng đã khai báo.

### 4. Cross-domain ExDark/AU-AIR/UAVDT kém

1. Báo cáo overlap classes riêng trước khi kết luận domain gap.
2. Hiệu chỉnh class mapping và confidence trên validation, không tune trên test.
3. Domain-balanced fine-tuning bằng sampler theo dataset/sequence.
4. Style augmentation theo thống kê sáng thực của ExDark thay vì chỉ gamma tổng hợp.
5. Test-time adaptation chỉ được đưa vào kết quả chính nếu latency vẫn đạt proposal.

## Thứ tự quyết định

1. Hoàn thành Stage C/D và đánh giá nguyên bản.
2. Chạy ablation loss/gate có chi phí thấp.
3. Chạy curriculum và clean distillation nếu metric detection/clean chưa đạt.
4. Thử P2/NWD scale-aware nếu tiny-object metric vẫn là nút thắt.
5. Tối ưu export và benchmark lại latency sau khi khóa checkpoint tốt nhất.
6. Ghi vòng được chọn, vòng bị loại và lý do vào JSON cùng tài liệu tổng kết.

## Artifact bắt buộc cho mỗi vòng

- Config đầy đủ và seed.
- `epoch_history.json` và `results.json`.
- `best.pt`, `last.pt` cùng JSON sidecar.
- Metric per-domain, per-class, tiny/small/medium/large.
- Latency p50/p95, VRAM peak, parameter count và môi trường phần mềm/phần cứng.
- Quyết định giữ hoặc loại vòng thử nghiệm dựa trên KPI.
