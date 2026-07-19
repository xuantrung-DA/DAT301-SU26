# Proposal implementation status

| Proposal item | Implementation | Status |
|---|---|---|
| M1 adaptive gate | `src/models/gate.py` | Implemented + unit tested |
| M2 bounded residual enhancer | `src/models/enhancement.py` | Implemented; hard ≤0.35M test |
| M3 DG-RDM | `src/models/dgrdm.py` | Implemented; GT/objectness interface tested |
| M4 YOLO11n + optional P2 | `src/training/detector_bridge.py`, `configs/yolo11n_p2.yaml` | Implemented; P2 remains exploratory |
| M5 NWD | `src/training/losses.py` | Implemented as optional train-only loss |
| Stage A B0/B1/B2 | `src/training/train_detector.py` | Runnable for 3 registered seeds |
| Stage B | `src/training/pretrain_enhancer.py` | Implemented |
| Stage C/D | `src/training/joint_train.py` | Implemented with freeze/unfreeze and loss ramp |
| Data protocol v2 | `src/preprocess/lowlight.py`, `make_lowlight_visdrone.py` | Implemented; existing v1 data must be regenerated |
| AP-small/diagnostics | `src/evaluation/evaluate_detector.py` | Implemented |
| Mean±std/bootstrap | `src/evaluation/statistics.py` | Implemented |
| Failure taxonomy | `src/evaluation/failure_analysis.py` | Implemented |
| p50/p95/FPS/memory | `src/evaluation/profile_latency.py` | Implemented |
| ONNX branch export/parity | `src/deployment/export_onnx.py` | Implemented |
| TensorRT FP16 engine | `src/deployment/build_tensorrt_python.py`, `src/deployment/profile_tensorrt.py` | Built and profiled; latency gate passed |
| ExDark/AU-AIR full evaluation | `scripts/data/make_visdrone_overlap_labels.py` | Full test splits evaluated; ≥5% gain gate failed |
| Three-seed final results | `runs/evaluation_p2_final/statistics_3seed.json` | LL2 mean gates and clean-safety gate passed |

“Implemented” nghĩa là code path, config, logging và test đã tồn tại; không đồng nghĩa acceptance gate nghiên cứu đã đạt. Kết quả smoke và kết quả final phải được tách riêng.
