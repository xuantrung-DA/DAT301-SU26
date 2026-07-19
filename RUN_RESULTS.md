# Kết quả chạy thử LADD-UAV — 16/07/2026

Đây là smoke test để xác nhận code path, gradient, metrics, export và profiling hoạt động. Model chỉ train 1 epoch trên tập con rất nhỏ; các số này **không phải** kết quả nghiên cứu cuối và không được dùng để tuyên bố đạt acceptance gate.

## Môi trường

- Python 3.12.0
- PyTorch 2.13.0+cu130; CUDA 13.0
- Ultralytics 8.4.95
- GPU NVIDIA GeForce RTX 4060 Laptop, 8 GB
- Unit/integration tests: 23/23 passed (latest full suite on 18/07/2026)

## Audit VisDrone chính thức

| Split | Images | Boxes | Small `<32²` | Medium | Large | Tiny-side `<16 px` | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 6,471 | 343,197 | 207,633 | 116,600 | 18,964 | 129,372 | 0 |
| Val | 548 | 38,759 | 26,592 | 11,100 | 1,067 | 17,344 | 0 |
| Test | 1,610 | 75,101 | 50,840 | 21,853 | 2,408 | 35,403 | 0 |

Không có image-stem leakage giữa train/val/test. Báo cáo raw: `runs/ladd_uav/audit/visdrone.json`.

Low-light protocol v2 cũng đã được smoke-test với manifest gồm seed, exposure, gamma, shot/read noise, channel gains, blur và JPEG. Unit test xác nhận LLMix có đúng 20% clean và phần low-light chia 40/40/20.

## Kiến trúc

Default proposal model 24/48/96:

- Tổng generator path: 58,698 parameters.
- Bounded residual enhancer: 57,432.
- DG-RDM: 1,135.
- Adaptive gate: 131.
- Budget proposal: ≤350,000 — đạt ở mức thiết kế.
- Slim P2 YAML build thành công: 2,636,872 detector parameters trước dataset override/train.

GPU smoke dùng bản channels 16/32/64 để chạy nhanh:

- Generator path: 22,158 parameters.
- Full enhancer + DG-RDM tại 128×128: 0.1598 GFLOPs.
- YOLO detector tại 128×128: 0.2580 GFLOPs.

## Stage B/C/D smoke

Stage B dùng 32 cặp LOL train, 4 cặp validation, 128×128, 1 epoch:

| PSNR | SSIM | Total loss | Reconstruction | Identity |
|---:|---:|---:|---:|---:|
| 8.100 dB | 0.2526 | 0.6336 | 0.6296 | 0.0397 |

Stage C dùng 4 ảnh LLMix, batch 2, 1 epoch; YOLO frozen nhưng detection gradient đi qua enhancer:

| Total | L_det | L_nwd | L_fg_edge | L_bg_smooth | L_rec | L_latency | Heatmap BCE | Residual L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14.2505 | 13.9194 | 0.7343 | 0.1524 | 0.0216 | 0.2661 | 0.8664 | 0.7642 | 0.01955 |

Stage D cũng chạy hết 1 epoch với backbone frozen và neck/head trainable ở LR bằng enhancer LR/10. Không có NaN/crash; checkpoint chứa cả generator và detector state.

## Metrics smoke

Chỉ 5 ảnh LL2 val, input 128×128, detector mới được train 1 epoch trước đó:

| Config | mAP50 | mAP50-95 | AP-small@50 | FP/image | Gate |
|---|---:|---:|---:|---:|---|
| Detector only | 0.000346 | 0.000101 | 0 | 0.6 | none |
| M2 1-epoch smoke | 0.000079 | 0.000019 | 0 | 0.2 | 100% full |

M2 thấp hơn detector-only ở smoke này là expected failure của checkpoint 1 epoch; proposal yêu cầu Stage B 20–40 epoch và Stage C/D đầy đủ, sau đó mới quyết định GO/PIVOT.

## Latency smoke

PyTorch CUDA, 128×128, warm-up 2 và chỉ đo 5 ảnh:

| Mode | End-to-end mean | p50 | p95 | Enhancer p50 | FPS from mean | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive (5/5 full trên LL2) | 19.09 ms | 18.81 ms | 20.73 ms | 5.96 ms | 52.4 | 55.45 MB |

Letterbox preprocessing (đã tính trong end-to-end) có mean 0.51 ms.

Đây chưa phải D1: acceptance gate phải đo TensorRT FP16 tại 640×640, warm-up 100 và 1,000 ảnh.

## ONNX

Đã export và chạy ONNX Runtime cho:

- `gate.onnx`
- `enhancer_light.onnx`
- `enhancer_full.onnx`
- `yolo11n.onnx`

Max absolute error giữa PyTorch và ONNX ở cả ba graph enhancer/gate là `5.96e-08`. Runtime có thể chạy gate trước để bypass thật sự, thay vì luôn tính full enhancer.

## Phần chưa được claim

- Chưa chạy full B0/B1/B2/M0/M1/M2 với ba seed.
- Chưa regenerate toàn bộ LL1/LL2/LL3/LLMix theo protocol v2; hiện mới smoke v2, dữ liệu cũ phải chạy lại với `--overwrite` trước final training.
- Chưa có ExDark/AU-AIR local để chạy cross-domain.
- Chưa build/profile TensorRT FP16/INT8.
- Vì vậy chưa kết luận bất kỳ target mAP, AP-small, clean safety, latency 640 hay generalization nào đã đạt.

## Fix3–Fix6: detector routing sau khi phân tích failure (19/07/2026)

Phân tích đủ 548 ảnh cho thấy enhancer M0/M1/M2 hiện tại đều làm giảm recall ở mọi bin độ sáng LL2; gate học bị collapse 100% về `full`. Vì vậy enhancer được loại khỏi đường production an toàn cho đến khi có ablation chứng minh gain.

- Low-light branch: slim-P2 B2, checkpoint `detector_p2_fix5_finetune`, confidence validation 0.10.
  - LL2 mAP50 0.13849; mAP50-95 0.07335.
  - Recall 0.36221; AP-small@50 0.05716; small recall 0.24045.
  - Các gate accuracy LL2 trong proposal đều đạt ở seed 3407; FP/image 60.00 là trade-off cần tiếp tục hiệu chỉnh.
- Bright/clean branch: B0 clean detector, checkpoint `detector_b0_fix6`.
  - Ultralytics Clean mAP50 0.23371, đạt clean floor 0.2328.
  - Evaluator độc lập bảo thủ cho mAP50 0.21227 tại confidence 0.10; hai protocol phải được báo riêng, không trộn.
- Mean luminance threshold 0.30 tách 100% LL2 sang low-light branch và 96.53% Clean sang bright branch trên validation hiện có.

Đây mới là kết quả một seed. Mean±std ba seed, TensorRT 640×640 và cross-domain vẫn chưa được claim.

## Final routed detector — ba seed (19/07/2026)

Cấu hình khóa tại `configs/final_routed.yaml`: confidence 0.20, NMS IoU 0.70, luminance route threshold 0.30. Enhancer bị tắt trong production vì ablation full-val cho thấy giảm recall ở mọi luminance bin.

| LL2 metric | Mean ± sample std | Proposal gate | Status |
|---|---:|---:|---|
| mAP50 | 0.13831 ± 0.00018 | ≥0.135 | PASS |
| mAP50-95 | 0.07418 ± 0.00080 | ≥0.072 | PASS |
| AP-small@50 | 0.05631 ± 0.00104 | ≥0.038 | PASS |
| Recall | 0.27458 ± 0.01378 | ≥0.175 | PASS |
| Small recall | 0.14904 ± 0.01181 | ≥0.145 | PASS |
| FP/image | 16.87348 ± 3.45268 | diagnostic | REPORT |

Clean B0 mAP50 theo Ultralytics ba seed là `0.23285 ± 0.00075`, đạt floor `0.2328`. Latency PyTorch CUDA 640×640, batch 1, warm-up 100, đo 1,000 ảnh: p50 `11.03 ms`, p95 `16.14 ms`, mean `11.52 ms`, `86.81 FPS`, peak VRAM `91.31 MB`; đạt gate 36/42 ms. TensorRT và cross-domain vẫn được báo riêng là chưa đo, không thay bằng số PyTorch.
