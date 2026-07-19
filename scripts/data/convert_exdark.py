#!/usr/bin/env python3
"""CLI: convert official ExDark bbGt annotations and splits to YOLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ladd_uav.data.exdark import convert_exdark_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ExDark using Groundtruth bbGt files and imageclasslist.txt splits."
    )
    parser.add_argument("--images-root", type=Path, required=True, help="Root of 12 image folders.")
    parser.add_argument(
        "--annotations-root", type=Path, required=True, help="Root of 12 Groundtruth folders."
    )
    parser.add_argument("--split-file", type=Path, required=True, help="Official imageclasslist.txt.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--image-backend", choices=("auto", "pillow", "opencv"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = convert_exdark_dataset(
        args.images_root,
        args.annotations_root,
        args.split_file,
        args.output_root,
        transfer=args.transfer,
        overwrite=args.overwrite,
        image_backend=args.image_backend,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
