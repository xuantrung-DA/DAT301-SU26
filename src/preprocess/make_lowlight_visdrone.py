"""Create LowLight-VisDrone from a YOLO-formatted VisDrone dataset.

Input:
  datasets/VisDrone/images/{train,val,test}
  datasets/VisDrone/labels/{train,val,test}

Output:
  datasets/VisDrone-LL/LL2/images/{train,val,test}
  datasets/VisDrone-LL/LL2/labels/{train,val,test}

Labels are copied unchanged because only image illumination is degraded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.lowlight import LEVEL_CONFIGS, degrade_image_bgr


def stable_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (int(base_seed) + offset) % (2**32 - 1)


def copy_labels(input_root: Path, output_root: Path, split: str):
    src = input_root / "labels" / split
    dst = output_root / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"[WARN] Missing label folder: {src}")
        return
    for lbl in src.glob("*.txt"):
        shutil.copy2(lbl, dst / lbl.name)


def process_split(input_root: Path, output_root: Path, split: str, level: str, seed: int, ext: str = ".jpg"):
    img_dir = input_root / "images" / split
    out_img_dir = output_root / "images" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if not img_dir.exists():
        print(f"[WARN] Missing image folder: {img_dir}")
        return []

    rows = []
    image_paths = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img_path in tqdm(image_paths, desc=f"{level}-{split}"):
        rel = img_path.relative_to(input_root).as_posix()
        img_seed = stable_seed(seed, rel)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read {img_path}")
            continue
        out = degrade_image_bgr(img, level=level, seed=img_seed)
        out_name = img_path.with_suffix(ext).name
        out_path = out_img_dir / out_name
        cv2.imwrite(str(out_path), out)
        rows.append({"split": split, "source": rel, "output": out_path.relative_to(output_root).as_posix(), "seed": img_seed, "level": level})
    copy_labels(input_root, output_root, split)
    return rows


def write_yaml(output_root: Path, level: str):
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    text = f"""# Auto-generated LowLight-VisDrone YOLO config: {level}
path: {output_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""
    (output_root / f"visdrone_{level.lower()}.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="YOLO VisDrone root, e.g. datasets/VisDrone")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root, e.g. datasets/VisDrone-LL/LL2")
    parser.add_argument("--level", default="LL2", choices=["LL1", "LL2", "LL3"])
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in args.splits:
        all_rows.extend(process_split(args.input_root, args.output_root, split, args.level, args.seed))

    cfg = LEVEL_CONFIGS[args.level]
    config = {
        "level": args.level,
        "base_seed": args.seed,
        "beta_range": cfg.beta_range,
        "gamma_range": cfg.gamma_range,
        "noise_sigma_range": cfg.noise_sigma_range,
        "jpeg_quality_range": cfg.jpeg_quality_range,
        "blur_prob": cfg.blur_prob,
        "note": "Labels are copied unchanged from clean VisDrone."
    }
    (args.output_root / "degradation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with (args.output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "output", "seed", "level"])
        writer.writeheader()
        writer.writerows(all_rows)

    write_yaml(args.output_root, args.level)
    print(f"[DONE] Created {args.level} at {args.output_root}")


if __name__ == "__main__":
    main()
