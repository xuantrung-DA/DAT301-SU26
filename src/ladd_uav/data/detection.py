"""Generic helpers for external detection dataset converters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def xywh_to_normalized_yolo(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float, bool]:
    """Clip absolute ``left,top,width,height`` and return normalized YOLO xywh."""

    values = (left, top, width, height)
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError("bounding box contains a non-finite value")
    if width <= 0 or height <= 0:
        raise ValueError("bounding box width and height must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    raw = (left, top, left + width, top + height)
    x1 = min(float(image_width), max(0.0, raw[0]))
    y1 = min(float(image_height), max(0.0, raw[1]))
    x2 = min(float(image_width), max(0.0, raw[2]))
    y2 = min(float(image_height), max(0.0, raw[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box lies outside the image")
    clipped = (x1, y1, x2, y2) != raw
    return (
        (x1 + x2) / (2.0 * image_width),
        (y1 + y2) / (2.0 * image_height),
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
        clipped,
    )


def format_yolo_row(class_id: int, box: Sequence[float]) -> str:
    if len(box) != 4:
        raise ValueError("YOLO box must contain x_center, y_center, width, height")
    return f"{class_id} " + " ".join(f"{float(value):.8f}" for value in box)


def write_dataset_yaml(output_root: Path, class_names: Sequence[str], splits: Iterable[str]) -> None:
    split_names = list(splits)
    lines = ["path: ."]
    for key in ("train", "val", "test"):
        if key in split_names:
            lines.append(f"{key}: images/{key}")
    lines.extend(["names:", *(f"  {index}: {name}" for index, name in enumerate(class_names))])
    (Path(output_root) / "dataset.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
