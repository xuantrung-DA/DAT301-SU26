#!/usr/bin/env python3
"""CLI: deterministically generate LL1/LL2/LL3/LLMix paired datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ladd_uav.data.lowlight import PROTOCOL_SEEDS, synthesize_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic paired low-light VisDrone variants. Every image's "
            "seed, alpha, gamma, noise, RGB gains, and blur are saved to JSONL."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Converted YOLO root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Low-light output root.")
    parser.add_argument("--splits", nargs="+", help="Splits to synthesize (default: discover all).")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("LL1", "LL2", "LL3", "LLMix"),
        help="Any of LL1 LL2 LL3 LLMix (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(PROTOCOL_SEEDS[0],),
        help=(
            "Synthesis seeds. Default: 3407. For the complete protocol pass "
            "--seeds 3407 2025 301; multiple seeds use seed-<n> subfolders."
        ),
    )
    parser.add_argument(
        "--image-backend",
        choices=("auto", "pillow", "opencv"),
        default="auto",
        help="Image codec; OpenCV is optional and auto prefers Pillow.",
    )
    parser.add_argument(
        "--blur-backend",
        choices=("portable", "opencv"),
        default="portable",
        help="Portable NumPy is reproducible across installs; OpenCV is optional.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds contains duplicates")
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("--seeds must be non-negative")
    summaries: dict[str, object] = {}
    multiple = len(args.seeds) > 1
    for seed in args.seeds:
        destination = args.output_root / f"seed-{seed}" if multiple else args.output_root
        result = synthesize_dataset(
            args.dataset_root,
            destination,
            splits=args.splits,
            variants=args.variants,
            base_seed=seed,
            overwrite=args.overwrite,
            image_backend=args.image_backend,
            blur_backend=args.blur_backend,
        )
        summaries[str(seed)] = {
            variant: {split: stats.to_dict() for split, stats in split_stats.items()}
            for variant, split_stats in result.items()
        }
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
