"""Test low-light degradation and cheap enhancement baselines on one image.

Example:
python src/preprocess/test_one_image_lowlight.py \
  --image datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg \
  --level LL2 \
  --outdir debug_one_image
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.lowlight import (
    clahe_enhance_bgr,
    degrade_image_bgr,
    gamma_enhance_bgr,
    make_comparison_grid,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--level", default="LL2", choices=["LL1", "LL2", "LL3"])
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--outdir", type=Path, default=Path("debug_one_image"))
    parser.add_argument("--gamma", type=float, default=0.60)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    low = degrade_image_bgr(img, level=args.level, seed=args.seed)
    gamma = gamma_enhance_bgr(low, gamma=args.gamma)
    clahe = clahe_enhance_bgr(low, clip_limit=args.clahe_clip)

    stem = args.image.stem
    cv2.imwrite(str(args.outdir / f"{stem}_{args.level}.jpg"), low)
    cv2.imwrite(str(args.outdir / f"{stem}_{args.level}_gamma.jpg"), gamma)
    cv2.imwrite(str(args.outdir / f"{stem}_{args.level}_clahe.jpg"), clahe)

    grid = make_comparison_grid(
        [img, low, gamma, clahe],
        ["clean", args.level, f"{args.level}+gamma", f"{args.level}+CLAHE"],
    )
    grid_path = args.outdir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"[DONE] Saved outputs to {args.outdir}")
    print(f"[VIEW] {grid_path}")


if __name__ == "__main__":
    main()
