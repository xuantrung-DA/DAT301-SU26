# Evaluation Protocol v1

Protocol này được khóa trước khi chạy so sánh chính.

## Data và preprocessing

- Giữ official VisDrone train/val/test split.
- Ảnh detector dùng letterbox cố định 640×640; không stretch.
- Core seeds: 3407, 2025, 301.
- LLMix: 20% clean; 80% low-light chia LL1/LL2/LL3 theo tỷ lệ 40/40/20.
- Cùng confidence/NMS threshold cho mọi cấu hình; chỉ sweep trên validation.
- Không tune trên test hoặc real-domain test.

## Object sizes

- Primary small: diện tích GT trong tọa độ ảnh gốc `< 32²` pixel.
- Medium: `32² ≤ area < 96²`; large: `area ≥ 96²`.
- Secondary tiny: width `<16` hoặc height `<16` pixel trong ảnh gốc.

## Accuracy outputs

- mAP50, mAP50-95, AP-small@50, AP-small@50:95.
- Precision, recall, F1, class-wise recall.
- Small recall, tiny-side recall, FP/image.
- Raw per-image CSV phải được giữ để bootstrap và failure analysis.

## Latency outputs

- Same hardware and versions for every compared configuration.
- Batch size 1, CUDA synchronize before/after each measured section.
- Warm-up 100, measure 1000 images.
- Report p50, p95, mean, std, FPS, peak GPU memory, gate-mode rates.
- Report enhancer-only and end-to-end separately.

## Statistics

- Core configurations B0/B1/B2/M0/M1/M2/D1: three registered seeds.
- Report mean±sample std.
- Final paired comparison: paired bootstrap 95% CI over matching validation images, exact `n`, and paired effect size.
- Exploratory A1/A2/A3/D2 may use one seed and must be labeled exploratory.

## Acceptance gates

| Dimension | Gate |
|---|---:|
| LL2 mAP50 | ≥0.135 |
| LL2 mAP50-95 | ≥0.072 |
| LL2 recall | ≥0.175 |
| LL2 AP-small | ≥0.038 |
| LL2 small recall | ≥0.145 |
| Clean mAP50 | ≥0.2328 |
| p50 / p95 | ≤36 / ≤42 ms |
| Enhancer p50 overhead | ≤6 ms |
| Gate skip Clean/LL1 | ≥55% |
| Real-domain relative mAP gain | ≥5% |
