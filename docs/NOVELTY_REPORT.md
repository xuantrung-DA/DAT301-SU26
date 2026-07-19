# Learned conditional routing novelty report

## Contribution

The final system replaces manual domain metadata with a learned 5,755-parameter router. It predicts `clean`, `synthetic_lowlight`, or `real_lowlight` and executes exactly one specialist detector. This is conditional computation rather than a three-model ensemble.

The router is trained only on training splits, selected on validation splits, and evaluated on 3,659 held-out/validation images. No detection labels or test-domain metadata are used at routing time.

## Router evidence

| Metric | Result |
|---|---:|
| Overall routing accuracy | 84.996% |
| Balanced routing accuracy | 91.904% |
| Clean recall | 98.723% |
| Synthetic-low-light recall | 97.628% |
| Real-low-light recall | 79.360% |
| Parameters | 5,755 |
| Router latency p95 | 0.523 ms |

## End-to-end evidence

| Domain | Learned-router mAP50 | Relevant gate/baseline | Status |
|---|---:|---:|---|
| LL2 | 0.13916 | ≥0.135 | PASS |
| ExDark | 0.60492 | B0 0.02322; required gain ≥5% | PASS |
| Clean, independent evaluator | 0.21079 | B0 same evaluator 0.21227 | -0.70% safety delta |

The learned router sends 97.81% of LL2 images to P2, 98.91% of clean images to B0, and 79.13% of ExDark images to the adapted detector. ExDark performance remains far above the baseline despite routing errors.

## Defensible novelty claim

The defensible contribution is a lightweight learned conditional router for multi-domain UAV detection that preserves synthetic low-light performance and clean safety while enabling supervised real-low-light adaptation, with sub-millisecond routing overhead and one-detector-per-image execution. It is not a claim of zero-shot domain generalization.
