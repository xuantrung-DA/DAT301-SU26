#!/usr/bin/env python3
"""CLI: convert official UAVDT detection data to sequence-safe YOLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ladd_uav.data.uavdt import convert_uavdt_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert UAVDT *_gt_whole.txt while retaining M_attr train/test sequences."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--image-backend", choices=("auto", "pillow", "opencv"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = convert_uavdt_dataset(
        args.source_root,
        args.output_root,
        transfer=args.transfer,
        overwrite=args.overwrite,
        image_backend=args.image_backend,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
