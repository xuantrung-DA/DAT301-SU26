#!/usr/bin/env python3
"""CLI: convert extracted official VisDrone2019-DET splits to YOLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ladd_uav.data.visdrone import (  # noqa: E402
    OFFICIAL_SPLIT_ORDER,
    convert_visdrone_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official VisDrone2019-DET annotation text files to normalized "
            "10-class YOLO labels while retaining train/val/test split folders."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Folder containing extracted VisDrone2019-DET-<split> directories.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Destination YOLO root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=OFFICIAL_SPLIT_ORDER,
        help="Official splits to convert (default: every discovered split).",
    )
    parser.add_argument(
        "--transfer",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How images are placed in the output tree (default: copy).",
    )
    parser.add_argument(
        "--image-backend",
        choices=("auto", "pillow", "opencv"),
        default="auto",
        help="Image decoder used to validate dimensions; OpenCV is optional.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Record and skip invalid rows/images instead of failing immediately.",
    )
    parser.add_argument(
        "--require-test-annotations",
        action="store_true",
        help="Fail if a test image has no official annotation (normally empty labels are created).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = convert_visdrone_dataset(
        args.source_root,
        args.output_root,
        splits=args.splits,
        transfer=args.transfer,
        overwrite=args.overwrite,
        strict=not args.skip_invalid,
        allow_missing_test_annotations=not args.require_test_annotations,
        image_backend=args.image_backend,
    )
    print(json.dumps({name: value.to_dict() for name, value in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
