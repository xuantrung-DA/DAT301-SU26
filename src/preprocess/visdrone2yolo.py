"""Convert VisDrone2019-DET annotations to YOLO detection format.
Expected raw structure after unzip:
raw_visdrone/
  VisDrone2019-DET-train/
    images/*.jpg
    annotations/*.txt
  VisDrone2019-DET-val/
    images/*.jpg
    annotations/*.txt
  VisDrone2019-DET-test-dev/
    images/*.jpg
    annotations/*.txt

Output structure:
datasets/VisDrone/
  images/train/*.jpg
  labels/train/*.txt
  images/val/*.jpg
  labels/val/*.txt
  images/test/*.jpg
  labels/test/*.txt

VisDrone annotation columns:
  bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
Object category: 0 ignored, 1 pedestrian, ..., 10 motor.
YOLO class index: 0 pedestrian, ..., 9 motor.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def clip_bbox_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    x1 = max(0.0, min(float(x), img_w - 1.0))
    y1 = max(0.0, min(float(y), img_h - 1.0))
    x2 = max(0.0, min(float(x + w), img_w - 1.0))
    y2 = max(0.0, min(float(y + h), img_h - 1.0))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return x1, y1, bw, bh


def convert_one_annotation(ann_path: Path, img_path: Path) -> list[str]:
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    lines_out: list[str] = []
    raw = ann_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return lines_out

    for line_no, line in enumerate(raw.splitlines(), start=1):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            category = int(float(parts[5]))
        except ValueError:
            print(f"[WARN] Cannot parse {ann_path}:{line_no}: {line}")
            continue

        # In VisDrone, score==0 and/or category==0 usually means ignored region.
        if score == 0 or category == 0:
            continue
        if not (1 <= category <= 10):
            continue

        x, y, w, h = clip_bbox_xywh(x, y, w, h, img_w, img_h)
        if w <= 1 or h <= 1:
            continue

        cls = category - 1
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        vals = [x_center, y_center, w_norm, h_norm]
        if any(v < 0 or v > 1 for v in vals):
            continue
        lines_out.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    return lines_out


def convert_split(raw_root: Path, out_root: Path, split: str, copy_images: bool = True):
    folder = SPLITS[split]
    src_dir = raw_root / folder
    src_img_dir = src_dir / "images"
    src_ann_dir = src_dir / "annotations"
    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists() or not src_ann_dir.exists():
        raise FileNotFoundError(f"Missing images/annotations in {src_dir}")

    image_paths = sorted(src_img_dir.glob("*.jpg"))
    print(f"[INFO] Converting {split}: {len(image_paths)} images")

    for img_path in tqdm(image_paths, desc=f"{split}"):
        ann_path = src_ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            print(f"[WARN] Missing annotation for {img_path.name}")
            continue
        if copy_images:
            shutil.copy2(img_path, out_img_dir / img_path.name)
        lines = convert_one_annotation(ann_path, img_path)
        (out_lbl_dir / f"{img_path.stem}.txt").write_text("".join(lines), encoding="utf-8")


def write_yaml(out_root: Path):
    names_block = "\n".join([f"  {k}: {v}" for k, v in VISDRONE_NAMES.items()])
    yaml_text = f"""# Auto-generated VisDrone YOLO config
path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""
    yaml_path = out_root / "visdrone_clean.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[INFO] Wrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder containing VisDrone2019-DET-train/val/test-dev")
    parser.add_argument("--out-root", type=Path, required=True, help="Output YOLO dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLITS))
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels; do not copy images")
    args = parser.parse_args()

    for split in args.splits:
        convert_split(args.raw_root, args.out_root, split, copy_images=not args.no_copy_images)
    write_yaml(args.out_root)
    print("[DONE] VisDrone conversion completed.")


if __name__ == "__main__":
    main()
