#!/usr/bin/env python3
"""CLI: audit a YOLO dataset and emit source-pixel object-size statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ladd_uav.data.audit import audit_yolo_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit image/label pairing, corrupt YOLO rows, cross-split byte leakage, "
            "COCO S/M/L sizes, and width/height <16 px counts."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO dataset root.")
    parser.add_argument("--splits", nargs="+", help="Splits to audit (default: discover all).")
    parser.add_argument(
        "--class-count",
        type=int,
        default=10,
        help="Allowed classes are 0..N-1 (default: 10; use 0 for no upper bound).",
    )
    parser.add_argument("--report", type=Path, help="JSON report path (default: dataset/audit.json).")
    parser.add_argument("--max-issues", type=int, default=1000)
    parser.add_argument(
        "--no-content-leakage-check",
        action="store_true",
        help="Skip SHA-256 comparison of images across splits.",
    )
    parser.add_argument(
        "--image-backend", choices=("auto", "pillow", "opencv"), default="auto"
    )
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="Return exit status 0 even when integrity issues are found.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_yolo_dataset(
        args.dataset_root,
        splits=args.splits,
        class_count=None if args.class_count == 0 else args.class_count,
        check_content_leakage=not args.no_content_leakage_check,
        max_issues=args.max_issues,
        image_backend=args.image_backend,
    )
    report_path = args.report or args.dataset_root / "audit.json"
    report.write_json(report_path)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok or args.allow_issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
