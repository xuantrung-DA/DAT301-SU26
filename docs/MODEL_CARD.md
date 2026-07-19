# LADD-UAV Model Card

## Intended use

Low-light small-object detection trong ảnh UAV, ưu tiên VisDrone/LowLight-VisDrone ở input detector 640×640. Output gồm boxes, class IDs, confidence, gate mode/probability, illumination/noise proxy và latency logs.

## Architecture

- Adaptive gate: bypass/light/full.
- 3-scale depthwise-separable bounded residual enhancer, channels 24/48/96.
- DG-RDM với GT Gaussian box heatmap khi train và learned objectness heatmap khi inference.
- YOLO11n detector; slim P2 và NWD là ablation có điều kiện.

## Constraints

- Generator path ≤0.35M parameters.
- Enhancer overhead target ≤6 ms p50.
- Clean mAP drop không quá 0.005.
- Không claim GAN-based perceptual quality; main objective là detection.

## Known limitations

- Synthetic-to-real generalization phải được xác nhận riêng trên ExDark/AU-AIR.
- Untrained gate dùng illumination prior an toàn nhưng chỉ checkpoint Stage C/D mới có task-specific correction.
- NWD candidate matching trong implementation là auxiliary nearest-candidate approximation; phải ablate A2 trước khi giữ trong final model.
- TensorRT/INT8 gates chỉ được claim sau parity và mAP validation trên đúng hardware protocol.
- Không dùng cụm “the first”; claim an toàn là “we propose and evaluate”.

## Production selection (19/07/2026)

Production dùng cấu hình `configs/final_routed.yaml`: slim-P2 B2 cho frame có mean luminance dưới 0.30 và B0 clean detector cho frame sáng hơn. Mỗi frame chỉ chạy một detector. Confidence khóa ở 0.20, NMS IoU 0.70, input 640×640.

Enhancer/DG-RDM không nằm trong production checkpoint hiện tại: full validation cho thấy M0/M1/M2 làm giảm recall ở mọi bin độ sáng và learned gate collapse về `full`. Các module vẫn được giữ làm research ablation, nhưng không được claim là nguyên nhân tạo gain của kết quả final.

Ba-seed LL2: mAP50 `0.13831±0.00018`, mAP50-95 `0.07418±0.00080`, AP-small@50 `0.05631±0.00104`, recall `0.27458±0.01378`, small recall `0.14904±0.01181`. Clean mAP50 Ultralytics `0.23285±0.00075`. PyTorch CUDA latency 640×640 p50/p95 `11.03/16.14 ms`; TensorRT và real-domain transfer chưa được claim.
