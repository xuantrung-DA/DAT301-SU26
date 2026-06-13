"""Draw YOLO labels on one image for quick verification.

Example:
python src/preprocess/draw_yolo_labels.py \
  --image datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg \
  --label datasets/VisDrone/labels/val/0000001_02999_d_0000005.txt \
  --out debug_labels.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def auto_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("debug_labels.jpg"))
    parser.add_argument("--conf", type=float, default=None, help="Reserved for prediction files; ignored for GT labels")
    args = parser.parse_args()

    label_path = args.label or auto_label_path(args.image)
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    h, w = img.shape[:2]

    if not label_path.exists():
        raise FileNotFoundError(f"Cannot find label: {label_path}")

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = NAMES[cls] if 0 <= cls < len(NAMES) else str(cls)
        cv2.putText(img, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"[DONE] Saved {args.out}")


if __name__ == "__main__":
    main()
