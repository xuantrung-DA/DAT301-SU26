"""Apply cheap enhancement baselines to a folder or YOLO dataset split.

Use this for baseline experiments:
  LL2 -> gamma -> YOLO
  LL2 -> CLAHE -> YOLO

Example folder mode:
python src/preprocess/enhancement_baselines.py \
  --input datasets/VisDrone-LL/LL2/images/val \
  --output datasets/VisDrone-LL/LL2_gamma/images/val \
  --method gamma
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.lowlight import clahe_enhance_bgr, gamma_enhance_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input image folder")
    parser.add_argument("--output", type=Path, required=True, help="Output image folder")
    parser.add_argument("--method", choices=["gamma", "clahe"], required=True)
    parser.add_argument("--gamma", type=float, default=0.60)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in args.input.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for p in tqdm(images, desc=args.method):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read {p}")
            continue
        if args.method == "gamma":
            out = gamma_enhance_bgr(img, gamma=args.gamma)
        else:
            out = clahe_enhance_bgr(img, clip_limit=args.clahe_clip, tile_grid_size=args.tile_grid_size)
        cv2.imwrite(str(args.output / p.name), out)
    print(f"[DONE] Saved {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
