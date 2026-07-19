#!/usr/bin/env python3
"""CLI: convert AU-AIR JSON to YOLO with whole-video splitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ladd_uav.data.auair import convert_auair_dataset  # noqa: E402


def _ratios(value: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must be comma-separated numbers") from exc
    if len(values) != 3 or any(item < 0 for item in values) or abs(sum(values) - 1.0) >= 1e-9:
        raise argparse.ArgumentTypeError("ratios must be train,val,test and sum to 1")
    return values  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert AU-AIR annotations.json; whole video sequences stay in one split."
    )
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-ratios", type=_ratios, default=(0.7, 0.1, 0.2))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--class-id-base", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--image-backend", choices=("auto", "pillow", "opencv"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = convert_auair_dataset(
        args.images_root,
        args.annotations_json,
        args.output_root,
        split_ratios=args.split_ratios,
        seed=args.seed,
        class_id_base=args.class_id_base,
        transfer=args.transfer,
        overwrite=args.overwrite,
        image_backend=args.image_backend,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
