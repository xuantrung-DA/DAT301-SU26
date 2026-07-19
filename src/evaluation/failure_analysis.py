"""Create a reproducible failure taxonomy from per-image evaluation rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--minimum-cases", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/failure_taxonomy.csv"))
    args = parser.parse_args()
    with args.per_image.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = []
    for row in rows:
        illumination = float(row["illumination_mean"])
        residual = float(row["residual_l1"])
        if int(row["small_fn"]) > 0:
            candidates.append((int(row["small_fn"]), "tiny_object_false_negative", row))
        if int(row["fp"]) > 0 and illumination < 0.30:
            candidates.append((int(row["fp"]), "dark_background_false_positive", row))
        if residual > 0.10:
            candidates.append((residual, "large_enhancement_residual", row))
        if illumination > 0.55 and row["gate_mode"] == "full":
            candidates.append((illumination, "clean_or_bright_over_enhancement", row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected, seen = [], set()
    for score, category, row in candidates:
        key = (row["image"], category)
        if key in seen:
            continue
        seen.add(key)
        selected.append({"category": category, "severity_score": score, **row})
        if len(selected) >= args.minimum_cases:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=selected[0].keys())
            writer.writeheader(); writer.writerows(selected)
    report = {
        "source": str(args.per_image),
        "selected_cases": len(selected),
        "minimum_requested": args.minimum_cases,
        "categories": sorted({row["category"] for row in selected}),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
