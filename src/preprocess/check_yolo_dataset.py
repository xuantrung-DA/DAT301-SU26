"""Check YOLO detection dataset labels and summarize class/small-object distribution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_root(data_yaml: Path, cfg: dict) -> Path:
    root = Path(cfg["path"])
    if not root.is_absolute():
        # First try relative to current working directory; then relative to YAML parent.
        if root.exists():
            return root
        return (data_yaml.parent / root).resolve()
    return root


def check_split(root: Path, split_name: str, rel_img_dir: str, n_classes: int):
    img_dir = root / rel_img_dir
    label_dir = root / "labels" / split_name
    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else []
    class_counter = Counter()
    small_counter = Counter()
    medium_counter = Counter()
    large_counter = Counter()
    tiny_side_counter = Counter()
    errors = []
    n_boxes = 0

    for img_path in tqdm(images, desc=f"check-{split_name}"):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            errors.append(f"missing label: {lbl_path}")
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w_img, h_img = im.size
        except Exception as exc:
            errors.append(f"corrupt image {img_path}: {exc}")
            continue
        for line_no, line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"bad format {lbl_path}:{line_no}: {line}")
                continue
            try:
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])
            except ValueError:
                errors.append(f"parse error {lbl_path}:{line_no}: {line}")
                continue
            if cls < 0 or cls >= n_classes:
                errors.append(f"class out of range {lbl_path}:{line_no}: {cls}")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                errors.append(f"bbox out of range {lbl_path}:{line_no}: {line}")
                continue
            n_boxes += 1
            class_counter[cls] += 1
            box_width, box_height = bw * w_img, bh * h_img
            box_area = box_width * box_height
            if box_area < 32 * 32:
                small_counter[cls] += 1
            elif box_area < 96 * 96:
                medium_counter[cls] += 1
            else:
                large_counter[cls] += 1
            if box_width < 16 or box_height < 16:
                tiny_side_counter[cls] += 1

    return {
        "split": split_name,
        "images": len(images),
        "boxes": n_boxes,
        "class_counter": dict(class_counter),
        "small_counter": dict(small_counter),
        "medium_counter": dict(medium_counter),
        "large_counter": dict(large_counter),
        "tiny_side_counter": dict(tiny_side_counter),
        "image_stems": [path.stem for path in images],
        "errors": errors[:50],
        "n_errors_total": len(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="YOLO data YAML")
    parser.add_argument("--output", type=Path, help="Optional JSON audit report")
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    root = resolve_root(args.data, cfg)
    names = cfg.get("names", {})
    n_classes = len(names)
    print(f"[INFO] root={root}")
    print(f"[INFO] n_classes={n_classes}")

    split_map = {"train": cfg.get("train"), "val": cfg.get("val"), "test": cfg.get("test")}
    all_results = []
    for split, rel_img_dir in split_map.items():
        if not rel_img_dir:
            continue
        result = check_split(root, split, rel_img_dir, n_classes)
        all_results.append(result)

    leakage = {}
    for index, first in enumerate(all_results):
        for second in all_results[index + 1:]:
            overlap = sorted(set(first["image_stems"]) & set(second["image_stems"]))
            leakage[f"{first['split']}__{second['split']}"] = {"count": len(overlap), "examples": overlap[:20]}

    for r in all_results:
        print("\n" + "=" * 80)
        print(f"Split: {r['split']}")
        print(f"Images: {r['images']} | Boxes: {r['boxes']} | Errors: {r['n_errors_total']}")
        print("Class distribution:")
        for cls, count in sorted(r["class_counter"].items()):
            name = names.get(cls, names.get(str(cls), str(cls)))
            small = r["small_counter"].get(cls, 0)
            medium = r["medium_counter"].get(cls, 0)
            large = r["large_counter"].get(cls, 0)
            tiny = r["tiny_side_counter"].get(cls, 0)
            print(f"  {cls:2d} {name:16s}: {count:8d} | small={small:8d} medium={medium:8d} large={large:8d} tiny-side={tiny:8d}")
        if r["errors"]:
            print("First errors:")
            for e in r["errors"][:10]:
                print("  -", e)
    print("\nLeakage by image stem:", leakage)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = [{key: value for key, value in result.items() if key != "image_stems"} for result in all_results]
        args.output.write_text(json.dumps({"data": str(args.data), "root": str(root), "splits": serializable, "leakage": leakage}, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote {args.output}")


if __name__ == "__main__":
    main()
