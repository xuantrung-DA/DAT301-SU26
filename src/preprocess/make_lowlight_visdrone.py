"""Create deterministic LL1/LL2/LL3/LLMix variants of VisDrone."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.lowlight import LEVEL_CONFIGS, degrade_image_bgr

LEVEL_CHOICES = ["LL1", "LL2", "LL3", "LLMix"]
MANIFEST_FIELDS = [
    "split", "source", "output", "seed", "level", "is_clean", "exposure", "gamma",
    "read_noise_sigma", "shot_peak", "channel_gain_b", "channel_gain_g", "channel_gain_r",
    "blur_applied", "blur_type", "jpeg_quality",
]


def stable_seed(base_seed: int, relative_path: str) -> int:
    offset = int(hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + offset) % (2**32 - 1)


def choose_mix_level(seed: int) -> str:
    """20% clean; remaining low-light is 40/40/20 LL1/LL2/LL3."""
    bucket = int(seed) % 100
    if bucket < 20:
        return "CLEAN"
    lowlight_bucket = bucket - 20
    if lowlight_bucket < 32:
        return "LL1"
    if lowlight_bucket < 64:
        return "LL2"
    return "LL3"


def copy_labels(input_root: Path, output_root: Path, split: str) -> None:
    source = input_root / "labels" / split
    destination = output_root / "labels" / split
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Missing label folder: {source}")
    for label in source.glob("*.txt"):
        shutil.copy2(label, destination / label.name)


def process_split(
    input_root: Path,
    output_root: Path,
    split: str,
    level: str,
    seed: int,
    extension: str = ".jpg",
    workers: int = 1,
    overwrite: bool = False,
    max_images: int | None = None,
) -> list[dict]:
    image_dir = input_root / "images" / split
    output_image_dir = output_root / "images" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")
    image_paths = sorted(path for path in image_dir.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        image_paths = image_paths[:max_images]

    def process_one(image_path: Path) -> dict | None:
        relative = image_path.relative_to(input_root).as_posix()
        image_seed = stable_seed(seed, relative)
        applied_level = choose_mix_level(image_seed) if level == "LLMix" else level
        output_path = output_image_dir / image_path.with_suffix(extension).name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        if applied_level == "CLEAN":
            result = image
            metadata = {
                "seed": image_seed, "level": "CLEAN", "exposure": 1.0, "gamma": 1.0,
                "read_noise_sigma": 0.0, "shot_peak": 0.0, "channel_gain_b": 1.0,
                "channel_gain_g": 1.0, "channel_gain_r": 1.0, "blur_applied": False,
                "blur_type": "none", "jpeg_quality": 100,
            }
        else:
            result, metadata = degrade_image_bgr(image, applied_level, image_seed, return_metadata=True)
        if overwrite or not output_path.exists():
            if not cv2.imwrite(str(output_path), result):
                return None
        return {
            "split": split,
            "source": relative,
            "output": output_path.relative_to(output_root).as_posix(),
            "is_clean": applied_level == "CLEAN",
            **{key: metadata[key] for key in MANIFEST_FIELDS if key in metadata},
        }

    if workers > 1:
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = [row for row in tqdm(executor.map(process_one, image_paths), total=len(image_paths), desc=f"{level}-{split}") if row]
    else:
        rows = [row for row in (process_one(path) for path in tqdm(image_paths, desc=f"{level}-{split}")) if row]
    copy_labels(input_root, output_root, split)
    return rows


def write_yaml(output_root: Path, level: str) -> None:
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    text = f"""# Auto-generated LowLight-VisDrone YOLO config: {level}
path: {output_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""
    (output_root / f"visdrone_{level.lower()}.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--level", default="LL2", choices=LEVEL_CHOICES)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing images using this protocol")
    parser.add_argument("--max-images", type=int, help="Debug/smoke limit per split")
    args = parser.parse_args()

    old_config_path = args.output_root / "degradation_config.json"
    if old_config_path.exists() and not args.overwrite:
        old_config = json.loads(old_config_path.read_text(encoding="utf-8"))
        if old_config.get("schema_version") != 2:
            raise RuntimeError("Output was generated by an older protocol. Use --overwrite or choose a new output directory.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for split in args.splits:
        rows.extend(process_split(args.input_root, args.output_root, split, args.level, args.seed, workers=args.workers, overwrite=args.overwrite, max_images=args.max_images))

    config = {
        "schema_version": 2,
        "level": args.level,
        "base_seed": args.seed,
        "level_configs": {name: vars(config) for name, config in LEVEL_CONFIGS.items()},
        "llmix_distribution": {"clean": 0.20, "within_lowlight": {"LL1": 0.40, "LL2": 0.40, "LL3": 0.20}},
        "formula": "clip(a * I**gamma * channel_gain + shot_noise + read_noise, 0, 1)",
        "labels": "Copied unchanged from clean VisDrone.",
    }
    old_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (args.output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_yaml(args.output_root, args.level)
    print(f"[DONE] Created {args.level}: {len(rows)} images at {args.output_root}")


if __name__ == "__main__":
    main()
