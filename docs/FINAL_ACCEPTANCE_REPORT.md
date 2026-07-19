# Final acceptance report — 19/07/2026

## TensorRT FP16

TensorRT 11.1.0.106, strongly typed FP16, batch 1, 640×640, warm-up 100 and 1,000 measured iterations on RTX 4060 Laptop:

| Engine | Mean | p50 | p95 | FPS | Gate |
|---|---:|---:|---:|---:|---|
| lowlight P2 | 2.43 ms | 2.40 ms | 2.75 ms | 412.00 | PASS |
| clean B0 | 2.15 ms | 2.03 ms | 2.85 ms | 465.49 | PASS |

Parity against ONNX Runtime FP16: score p99 absolute error is `1.67e-6` (P2) and `1.31e-6` (B0); box p99 absolute error is `1.0 px` and `0.5 px`, respectively. Engines and manifest are in `exports/final_routed_seed3407/tensorrt`.

## Cross-domain

Evaluation uses every image in each test split and only overlap classes remapped to VisDrone. AU-AIR is split by whole sequences.

| Dataset/config | Images | mAP50 | mAP50-95 | Relative to B0 |
|---|---:|---:|---:|---:|
| ExDark B0 | 2,563 | 0.02322 | 0.01140 | baseline |
| ExDark P2 | 2,563 | 0.01122 | 0.00509 | -51.68% |
| ExDark routed | 2,563 | 0.01151 | 0.00532 | -50.44% |
| ExDark B0 + gamma | 2,563 | 0.02040 | 0.00977 | -12.13% |
| ExDark B0 + CLAHE | 2,563 | 0.02228 | 0.01104 | -4.03% |
| AU-AIR B0 | 6,679 | 0.07223 | 0.02539 | baseline |
| AU-AIR P2 | 6,679 | 0.05563 | 0.01883 | -22.98% |

## Decision

LL2 mean gates, clean-safety and latency pass. Zero-shot real-domain relative mAP gain ≥5% does **not** pass, so no zero-shot generalization gain is claimed. The supervised adaptation result below passes the proposal's real-domain improvement target when domain-specific training is allowed.

## Supervised real-domain adaptation update

An additional B0 branch was fine-tuned only on the ExDark overlap `train` split and selected only with the `val` split. The untouched 2,563-image `test` split reaches mAP50 `0.69321` and mAP50-95 `0.44460`, versus B0 mAP50 `0.02322`. This passes the ≥5% real-domain improvement target when supervised domain adaptation is permitted.

The adapted checkpoint is not a universal replacement: it scores mAP50 `0.00849` on LL2 and `0.02036` on clean VisDrone. Production therefore retains the existing P2 low-light and B0 clean branches and adds the adapted branch only under explicit ExDark domain metadata, as documented in `configs/final_multidomain.yaml`. This is a supervised adaptation result, not a zero-shot generalization claim.

## Learned-router update

The explicit metadata requirement has been removed. A 5,755-parameter learned router obtains 91.90% balanced routing accuracy with p95 latency `0.523 ms`. End-to-end routed mAP50 is `0.13916` on LL2 and `0.60492` on ExDark. Clean mAP50 is `0.21079` in the independent evaluator versus B0's `0.21227` under the same protocol, a `0.70%` safety delta. Full novelty and ablation framing is in `docs/NOVELTY_REPORT.md`.
