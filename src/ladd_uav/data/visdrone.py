"""Convert official VisDrone2019-DET splits to a YOLO directory tree."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from .common import canonical_json, iter_images
from .imaging import IOBackend, image_size

# VisDrone category 0 is an ignored region and category 11 is "others".  The
# standard 10-class detection protocol maps official categories 1..10 to YOLO
# class indices 0..9.
VISDRONE_CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
VISDRONE_TO_YOLO = {official_id: official_id - 1 for official_id in range(1, 11)}
OFFICIAL_SPLIT_ORDER = ("train", "val", "test-dev", "test-challenge")


@dataclass(frozen=True)
class VisDroneSplit:
    name: str
    root: Path
    images: Path
    annotations: Path | None


@dataclass(frozen=True)
class VisDroneAnnotation:
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int
    truncation: int
    occlusion: int


@dataclass
class SplitConversionStats:
    split: str
    images: int = 0
    labels: int = 0
    objects: int = 0
    ignored_objects: int = 0
    clipped_objects: int = 0
    invalid_annotations: int = 0
    missing_annotations: int = 0
    unreadable_images: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _infer_split_name(name: str) -> str | None:
    lowered = name.casefold().replace("_", "-")
    for split in ("test-challenge", "test-dev", "train", "val"):
        if re.search(rf"(?:^|[-.]){re.escape(split)}(?:$|[-.])", lowered):
            return split
    # Also permit bare canonical names.
    return lowered if lowered in OFFICIAL_SPLIT_ORDER else None


def _make_split(root: Path, name: str | None = None) -> VisDroneSplit | None:
    image_dir = root / "images"
    if not image_dir.is_dir():
        return None
    split_name = name or _infer_split_name(root.name)
    if split_name is None:
        return None
    annotation_dir = root / "annotations"
    return VisDroneSplit(
        name=split_name,
        root=root,
        images=image_dir,
        annotations=annotation_dir if annotation_dir.is_dir() else None,
    )


def discover_visdrone_splits(source_root: Path) -> dict[str, VisDroneSplit]:
    """Discover official extracted split folders without merging their data.

    Supported raw layouts are::

        raw/VisDrone2019-DET-train/{images,annotations}
        raw/VisDrone2019-DET-val/{images,annotations}

    a single extracted split passed directly, or the alternate layout::

        raw/images/{train,val,test-dev}/
        raw/annotations/{train,val,test-dev}/
    """

    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"VisDrone source root does not exist: {source_root}")

    found: dict[str, VisDroneSplit] = {}
    direct = _make_split(source_root)
    if direct is not None:
        found[direct.name] = direct

    for child in sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        split = _make_split(child)
        if split is not None:
            if split.name in found and found[split.name].root != split.root:
                raise ValueError(
                    f"multiple raw folders discovered for split {split.name!r}: "
                    f"{found[split.name].root} and {split.root}"
                )
            found[split.name] = split

    images_root = source_root / "images"
    annotations_root = source_root / "annotations"
    if images_root.is_dir():
        for split_name in OFFICIAL_SPLIT_ORDER:
            split_images = images_root / split_name
            if not split_images.is_dir():
                continue
            split_annotations = annotations_root / split_name
            alternate = VisDroneSplit(
                name=split_name,
                root=source_root,
                images=split_images,
                annotations=split_annotations if split_annotations.is_dir() else None,
            )
            if split_name in found and found[split_name].images != alternate.images:
                raise ValueError(f"multiple raw folders discovered for split {split_name!r}")
            found[split_name] = alternate

    if not found:
        raise FileNotFoundError(
            f"no VisDrone splits found under {source_root}; expected an extracted "
            "VisDrone2019-DET-<split>/images directory"
        )
    return {name: found[name] for name in OFFICIAL_SPLIT_ORDER if name in found}


def parse_visdrone_annotation(line: str, *, source: str = "annotation") -> VisDroneAnnotation:
    values = [value.strip() for value in line.strip().split(",")]
    while values and values[-1] == "":
        values.pop()
    if len(values) != 8:
        raise ValueError(f"{source}: expected 8 comma-separated fields, got {len(values)}")
    try:
        left, top, width, height, score = (float(value) for value in values[:5])
        category, truncation, occlusion = (int(float(value)) for value in values[5:8])
    except ValueError as exc:
        raise ValueError(f"{source}: contains a non-numeric field") from exc
    numbers = (left, top, width, height, score)
    if not all(value == value and abs(value) != float("inf") for value in numbers):
        raise ValueError(f"{source}: contains a non-finite value")
    return VisDroneAnnotation(
        left=left,
        top=top,
        width=width,
        height=height,
        score=score,
        category=category,
        truncation=truncation,
        occlusion=occlusion,
    )


def annotation_to_yolo(
    annotation: VisDroneAnnotation,
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float, bool] | None:
    """Map and clip one annotation; return ``None`` for ignored categories."""

    class_id = VISDRONE_TO_YOLO.get(annotation.category)
    if class_id is None:
        return None
    if annotation.width <= 0 or annotation.height <= 0:
        raise ValueError("bounding box width and height must be positive")

    raw_x1 = annotation.left
    raw_y1 = annotation.top
    raw_x2 = annotation.left + annotation.width
    raw_y2 = annotation.top + annotation.height
    x1 = min(float(image_width), max(0.0, raw_x1))
    y1 = min(float(image_height), max(0.0, raw_y1))
    x2 = min(float(image_width), max(0.0, raw_x2))
    y2 = min(float(image_height), max(0.0, raw_y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box falls completely outside the image")
    clipped = (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2)
    box_width, box_height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (
        class_id,
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
        clipped,
    )


def _transfer_file(source: Path, destination: Path, mode: Literal["copy", "hardlink", "symlink"]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:  # pragma: no cover - guarded by public validation
        raise ValueError(f"unsupported transfer mode: {mode}")


def _convert_split(
    raw: VisDroneSplit,
    output_root: Path,
    *,
    transfer: Literal["copy", "hardlink", "symlink"],
    overwrite: bool,
    strict: bool,
    allow_missing_annotations: bool,
    image_backend: IOBackend,
) -> SplitConversionStats:
    stats = SplitConversionStats(split=raw.name)
    output_images = output_root / "images" / raw.name
    output_labels = output_root / "labels" / raw.name
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    images = list(iter_images(raw.images))
    if not images:
        raise FileNotFoundError(f"no images found in VisDrone split: {raw.images}")

    seen_stems: set[str] = set()
    index_lines: list[str] = []
    for source_image in images:
        if source_image.stem.casefold() in seen_stems:
            raise ValueError(f"duplicate image stem in split {raw.name}: {source_image.stem}")
        seen_stems.add(source_image.stem.casefold())
        try:
            width, height = image_size(source_image, backend=image_backend)
        except ValueError:
            stats.unreadable_images += 1
            if strict:
                raise
            continue

        source_annotation = raw.annotations / f"{source_image.stem}.txt" if raw.annotations else None
        yolo_rows: list[str] = []
        if source_annotation is None or not source_annotation.is_file():
            stats.missing_annotations += 1
            if not allow_missing_annotations:
                raise FileNotFoundError(
                    f"missing annotation for {source_image.name} in split {raw.name}"
                )
        else:
            for line_number, raw_line in enumerate(
                source_annotation.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                if not raw_line.strip():
                    continue
                source = f"{source_annotation}:{line_number}"
                try:
                    annotation = parse_visdrone_annotation(raw_line, source=source)
                    converted = annotation_to_yolo(annotation, width, height)
                except ValueError:
                    stats.invalid_annotations += 1
                    if strict:
                        raise
                    continue
                if converted is None:
                    stats.ignored_objects += 1
                    continue
                class_id, center_x, center_y, box_width, box_height, clipped = converted
                stats.clipped_objects += int(clipped)
                stats.objects += 1
                yolo_rows.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} "
                    f"{box_width:.8f} {box_height:.8f}"
                )

        destination_image = output_images / source_image.name
        destination_label = output_labels / f"{source_image.stem}.txt"
        existing = [path for path in (destination_image, destination_label) if path.exists() or path.is_symlink()]
        if existing and not overwrite:
            raise FileExistsError(f"destination already exists (use overwrite=True): {existing[0]}")
        if overwrite:
            for path in existing:
                path.unlink()
        _transfer_file(source_image, destination_image, transfer)
        destination_label.write_text(
            "\n".join(yolo_rows) + ("\n" if yolo_rows else ""), encoding="utf-8", newline="\n"
        )
        stats.images += 1
        stats.labels += 1
        index_lines.append(f"images/{raw.name}/{source_image.name}")

    (output_root / f"{raw.name}.txt").write_text(
        "\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8", newline="\n"
    )
    return stats


def _write_dataset_yaml(output_root: Path, split_names: Iterable[str]) -> None:
    names = list(split_names)
    lines = ["path: ."]
    if "train" in names:
        lines.append("train: images/train")
    if "val" in names:
        lines.append("val: images/val")
    tests = [name for name in ("test-dev", "test-challenge") if name in names]
    if len(tests) == 1:
        lines.append(f"test: images/{tests[0]}")
    elif tests:
        lines.append("test:")
        lines.extend(f"  - images/{name}" for name in tests)
    lines.extend(["names:", *(f"  {index}: {name}" for index, name in enumerate(VISDRONE_CLASS_NAMES))])
    (output_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def convert_visdrone_dataset(
    source_root: Path,
    output_root: Path,
    *,
    splits: Iterable[str] | None = None,
    transfer: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
    strict: bool = True,
    allow_missing_test_annotations: bool = True,
    image_backend: IOBackend = "auto",
) -> dict[str, SplitConversionStats]:
    """Convert selected official splits and return per-split counters.

    Train/validation annotations are mandatory.  Test annotations may be absent
    because official challenge test labels are not distributed; empty YOLO label
    files are then created so downstream loaders keep a one-to-one image/label
    layout.
    """

    if transfer not in {"copy", "hardlink", "symlink"}:
        raise ValueError("transfer must be one of: copy, hardlink, symlink")
    discovered = discover_visdrone_splits(Path(source_root))
    selected = list(discovered) if splits is None else list(dict.fromkeys(splits))
    unknown = [name for name in selected if name not in OFFICIAL_SPLIT_ORDER]
    if unknown:
        raise ValueError(f"unknown VisDrone splits: {unknown}; expected {OFFICIAL_SPLIT_ORDER}")
    missing = [name for name in selected if name not in discovered]
    if missing:
        raise FileNotFoundError(
            f"requested splits were not discovered: {missing}; found {list(discovered)}"
        )
    if not selected:
        raise ValueError("at least one split must be selected")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, SplitConversionStats] = {}
    for name in selected:
        results[name] = _convert_split(
            discovered[name],
            output_root,
            transfer=transfer,
            overwrite=overwrite,
            strict=strict,
            allow_missing_annotations=allow_missing_test_annotations and name.startswith("test"),
            image_backend=image_backend,
        )
    _write_dataset_yaml(output_root, selected)
    summary = {
        "class_names": list(VISDRONE_CLASS_NAMES),
        "source_root": str(Path(source_root).resolve()),
        "splits": {name: result.to_dict() for name, result in results.items()},
        "transfer": transfer,
    }
    (output_root / "conversion_summary.json").write_text(
        canonical_json(summary) + "\n", encoding="utf-8", newline="\n"
    )
    return results
