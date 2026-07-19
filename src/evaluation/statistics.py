"""Mean±std and paired bootstrap confidence intervals from raw evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SUMMARY_METRICS = [
    "map50", "map50_95", "ap_small_50", "ap_small_50_95", "precision",
    "recall", "f1", "small_recall", "tiny_side_recall", "fp_per_image",
]


def bootstrap_difference(left: np.ndarray, right: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    if len(left) != len(right) or not len(left):
        raise ValueError("Paired bootstrap requires non-empty arrays of equal length")
    rng = np.random.default_rng(seed)
    differences = left - right
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    boot = differences[indices].mean(axis=1)
    std = differences.std(ddof=1) if len(differences) > 1 else 0.0
    return {
        "n": int(len(left)),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "paired_effect_size_dz": float(differences.mean() / std) if std > 0 else 0.0,
    }


def load_rows(path: Path) -> dict[str, dict[str, float]]:
    output = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            tp, fp, fn = float(row["tp"]), float(row["fp"]), float(row["fn"])
            small_tp, small_fn = float(row["small_tp"]), float(row["small_fn"])
            output[Path(row["image"]).name] = {
                "recall": tp / max(tp + fn, 1.0),
                "precision": tp / max(tp + fp, 1.0),
                "small_recall": small_tp / max(small_tp + small_fn, 1.0),
                "fp_per_image": fp,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/statistics.json"))
    args = parser.parse_args()

    summaries: dict[str, list[dict]] = {}
    for path in args.input.rglob("*_summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        domain = value.get("domain") or path.parent.name or "unspecified"
        key = f"{domain}/{value['configuration']}"
        summaries.setdefault(key, []).append(value)
    aggregate = {}
    for configuration, values in summaries.items():
        aggregate[configuration] = {}
        for metric in SUMMARY_METRICS:
            observations = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            aggregate[configuration][metric] = {
                "mean": float(observations.mean()),
                "std": float(observations.std(ddof=1)) if len(observations) > 1 else 0.0,
                "seeds": len(observations),
                "values": observations.tolist(),
            }

    paired_by_domain: dict[str, dict[str, list[list[float]]]] = {}
    baseline_paths = {
        (str(path.parent.relative_to(args.input)), path.name.replace(f"{args.baseline}_", "")): path
        for path in args.input.rglob(f"{args.baseline}_seed_*_per_image.csv")
    }
    candidate_paths = {
        (str(path.parent.relative_to(args.input)), path.name.replace(f"{args.candidate}_", "")): path
        for path in args.input.rglob(f"{args.candidate}_seed_*_per_image.csv")
    }
    common_runs = sorted(set(baseline_paths) & set(candidate_paths))
    for run_key in common_runs:
        relative_parent, _run_name = run_key
        domain = Path(relative_parent).name if relative_parent not in ("", ".") else "unspecified"
        paired = paired_by_domain.setdefault(
            domain,
            {metric: [[], []] for metric in ("recall", "precision", "small_recall", "fp_per_image")},
        )
        baseline_rows = load_rows(baseline_paths[run_key])
        candidate_rows = load_rows(candidate_paths[run_key])
        for image in sorted(set(baseline_rows) & set(candidate_rows)):
            for metric in paired:
                paired[metric][0].append(candidate_rows[image][metric])
                paired[metric][1].append(baseline_rows[image][metric])
    confidence_intervals = {
        domain: {
            metric: bootstrap_difference(np.asarray(values[0]), np.asarray(values[1]), args.bootstrap_samples, args.seed)
            for metric, values in paired.items() if values[0]
        }
        for domain, paired in paired_by_domain.items()
    }
    report = {
        "aggregate_mean_std": aggregate,
        "comparison": {
            "candidate": args.candidate,
            "baseline": args.baseline,
            "paired_bootstrap_by_domain": confidence_intervals,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
