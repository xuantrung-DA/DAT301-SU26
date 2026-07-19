# LADD-UAV

Latency-Aware Detection-Guided Dynamic Enhancement for small-object detection in low-light UAV imagery. Project này triển khai theo `DAT_Proposal.xlsx`; main path là:

```text
RGB UAV frame
  -> Adaptive Gate {bypass, light, full}
  -> ultra-light bounded residual enhancer (DSConv 24/48/96)
  -> DG-RDM (GT heatmap khi train, objectness heatmap khi inference)
  -> YOLO11n (+ slim P2/NWD theo ablation)
  -> boxes/classes/scores + gate/latency/memory logs
```

U-Net 64→512, six ResBlocks, CBAM và GAN đã được loại khỏi main path. GAN chỉ còn là legacy/optional ablation.

## Cấu trúc chính

```text
configs/project.yaml                    Cấu hình Stage B/C/D và loss mặc định
configs/detector_adaptation.yaml        Baseline B0/B1/B2, ba seed đăng ký
configs/experiment_matrix.yaml          Ma trận B0–B4, M0–M2, A1–A3, D1–D2
configs/yolo11n_p2.yaml                 Slim P2 head cho A1
src/models/enhancement.py               Bounded residual enhancer + pipeline LADD
src/models/gate.py                      Gate bypass/light/full
src/models/dgrdm.py                     DG-RDM + Gaussian GT heatmap
src/training/pretrain_enhancer.py       Stage B
src/training/joint_train.py             Stage C/D
src/evaluation/evaluate_detector.py     mAP, AP-small, small recall, FP/image
src/evaluation/profile_latency.py       p50/p95/FPS/memory, warm-up 100/đo 1000
src/evaluation/statistics.py            mean±std và paired bootstrap 95% CI
src/deployment/export_onnx.py           ONNX gate/light/full + parity report
docs/PROTOCOL_V1.md                     Protocol đánh giá đã khóa
```

## 1. Môi trường

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
```

Kiểm tra:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
```

## 2. Dữ liệu và audit

Convert VisDrone sang YOLO, sau đó tạo audit JSON có corrupt-label check, phân bố small/medium/large và kiểm tra trùng split:

```powershell
python src/preprocess/visdrone2yolo.py --raw-root raw/VisDrone --out-root datasets/VisDrone
python src/preprocess/check_yolo_dataset.py `
  --data datasets/VisDrone/visdrone_clean.yaml `
  --output runs/audit/visdrone.json
```

Sinh low-light theo đúng công thức proposal. Mỗi ảnh lưu exposure, gamma, shot/read noise, channel gain, blur, JPEG và seed trong `manifest.csv`.

```powershell
python src/preprocess/make_lowlight_visdrone.py --input-root datasets/VisDrone --output-root datasets/VisDrone-LL/LL1 --level LL1 --seed 3407 --overwrite
python src/preprocess/make_lowlight_visdrone.py --input-root datasets/VisDrone --output-root datasets/VisDrone-LL/LL2 --level LL2 --seed 3407 --overwrite
python src/preprocess/make_lowlight_visdrone.py --input-root datasets/VisDrone --output-root datasets/VisDrone-LL/LL3 --level LL3 --seed 3407 --overwrite
python src/preprocess/make_lowlight_visdrone.py --input-root datasets/VisDrone --output-root datasets/VisDrone-LL/LLMix --level LLMix --seed 3407 --overwrite
```

`LLMix` dùng 20% clean; 80% low-light còn lại chia 40% LL1, 40% LL2, 20% LL3. Nếu output cũ không có schema v2, script buộc dùng `--overwrite` hoặc thư mục mới để tránh manifest sai.

## 3. Stage A — detector adaptation

Chạy B0/B1/B2 với ba seed `3407, 2025, 301`:

```powershell
python -m training.train_detector --baseline B0 --seed 3407 --epochs 100
python -m training.train_detector --baseline B1 --seed 3407 --epochs 100
python -m training.train_detector --baseline B2 --seed 3407 --epochs 100
```

Lặp lại cho hai seed còn lại. A1 slim-P2:

```powershell
python -m training.train_detector --baseline B2 --seed 3407 `
  --model-config configs/yolo11n_p2.yaml
```

## 4. Stage B/C/D — LADD-UAV

Stage B warm-up enhancer trên LOL-v1:

```powershell
python -m training.pretrain_enhancer --config configs/project.yaml
```

Stage C freeze toàn bộ YOLO, ramp detection loss 0→1 trong 5 epoch:

```powershell
python -m training.joint_train --config configs/project.yaml `
  --checkpoint runs/ladd_uav/checkpoints/seed_3407/stage_b_best.pt --stage C
```

Stage D giữ backbone frozen, chỉ fine-tune neck/head với LR bằng `enhancer LR / 10`:

```powershell
python -m training.joint_train --config configs/project.yaml `
  --checkpoint runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt --stage D
```

Loss main: `L_det + 0.20 L_nwd + 0.15 L_fg_edge + 0.05 L_bg_smooth + 0.10 L_identity + 0.25 L_rec + 0.02 L_latency`; thêm heatmap BCE để supervision objectness của DG-RDM.

## 5. Evaluation và ablation

Chạy B3/M0/M1/M2 cùng threshold/protocol:

```powershell
python -m evaluation.run_ablation `
  --images datasets/VisDrone-LL/LL2/images/val `
  --labels datasets/VisDrone-LL/LL2/labels/val `
  --yolo runs/detector_adaptation/b2_seed_3407/weights/best.pt `
  --generator runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt `
  --joint-checkpoint runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt `
  --seed 3407
```

Output có mAP50, mAP50-95, AP-small, precision, recall, F1, class-wise recall, small recall, tiny-side recall, FP/image, gate rate và per-image CSV.

Sau khi đủ ba seed:

```powershell
python -m evaluation.statistics --input runs/ablation --baseline none --candidate m2
python -m evaluation.failure_analysis --per-image runs/ablation/m2_seed_3407_per_image.csv
```

## 6. Deployment và latency

Xuất ba graph riêng để runtime có thể bypass thật sự, không phải tính full enhancer rồi mới trộn:

```powershell
python -m deployment.export_onnx `
  --generator runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt `
  --joint-checkpoint runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt `
  --yolo runs/detector_adaptation/b2_seed_3407/weights/best.pt
```

`gate.onnx` chạy trước; bypass bỏ qua enhancer, light gọi `enhancer_light.onnx`, full gọi `enhancer_full.onnx`. `parity_report.json` lưu sai số số học ONNX/PyTorch.

Build bốn TensorRT FP16 engine bằng cùng timing cache và lưu manifest/log cho từng graph:

```powershell
python -m deployment.build_tensorrt --onnx-dir exports/ladd_uav `
  --output exports/ladd_uav/tensorrt_fp16 --precision fp16
```

Script fail-fast nếu `trtexec` chưa có trong `PATH` hoặc thiếu bất kỳ ONNX graph nào.

Profile đúng protocol batch 1, warm-up 100, đo 1000 ảnh:

```powershell
python -m evaluation.profile_latency `
  --images datasets/VisDrone-LL/LL2/images/val `
  --generator runs/ladd_uav/checkpoints/seed_3407/stage_c_last.pt `
  --yolo runs/detector_adaptation/b2_seed_3407/weights/best.pt `
  --warmup 100 --measurements 1000
```

TensorRT FP16 có thể build từ ba ONNX bằng `trtexec`; INT8 chỉ giữ khi mAP giảm không quá 0.005.

## 7. Tiêu chí GO/NO-GO

- LL2 mAP50 ≥ 0.135; mAP50-95 ≥ 0.072.
- LL2 small recall ≥ 0.145; recall tổng ≥ 0.175.
- Clean mAP50 ≥ 0.2328.
- p50 ≤ 36 ms, p95 ≤ 42 ms, enhancer overhead ≤ 6 ms.
- Gate skip rate trên Clean/LL1 ≥ 55%.
- Real-domain relative mAP gain ≥ 5% trên ít nhất một tập ExDark/AU-AIR.

Các con số trên là acceptance gate của proposal, không phải kết quả đã đạt. Xem [RUN_RESULTS.md](RUN_RESULTS.md) cho smoke-test thực tế và [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) cho trạng thái từng deliverable.
