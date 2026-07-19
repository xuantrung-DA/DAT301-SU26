"""Convert official ExDark bbGt annotations and paper splits to YOLO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .common import canonical_json, iter_images, transfer_file
from .detection import format_yolo_row, write_dataset_yaml, xywh_to_normalized_yolo
from .imaging import IOBackend, image_size

EXDARK_CLASS_NAMES = (
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
)
_CLASS_LOOKUP = {name.casefold(): index for index, name in enumerate(EXDARK_CLASS_NAMES)}
_SPLIT_IDS = {1: "train", 2: "val", 3: "test"}

# The generic People class is not identical to VisDrone's posture-specific
# pedestrian/people distinction, so that mapping is explicitly marked ambiguous.
EXDARK_VISDRONE_OVERLAP = {
    "Bicycle": {"visdrone_class": "bicycle", "visdrone_id": 2, "ambiguous": False},
    "Bus": {"visdrone_class": "bus", "visdrone_id": 8, "ambiguous": False},
    "Car": {"visdrone_class": "car", "visdrone_id": 3, "ambiguous": False},
    "Motorbike": {"visdrone_class": "motor", "visdrone_id": 9, "ambiguous": False},
    "People": {
        "visdrone_class": "pedestrian",
        "visdrone_id": 0,
        "ambiguous": True,
        "note": "ExDark does not separate standing/walking from sitting persons.",
    },
}


def _index_images(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for image in iter_images(root, recursive=True):
        for key in {image.name.casefold(), image.stem.casefold()}:
            if key in result and result[key] != image:
                raise ValueError(f"ambiguous ExDark image key {key!r}: {result[key]} and {image}")
            result[key] = image
    return result


def _index_annotations(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for annotation in sorted(root.rglob("*.txt"), key=lambda path: path.as_posix().casefold()):
        keys = {annotation.stem.casefold()}
        nested_stem = Path(annotation.stem).stem.casefold()  # Handles image.jpg.txt.
        keys.add(nested_stem)
        for key in keys:
            if key in result and result[key] != annotation:
                raise ValueError(
                    f"ambiguous ExDark annotation key {key!r}: {result[key]} and {annotation}"
                )
            result[key] = annotation
    return result


def _parse_split_file(path: Path) -> list[tuple[str, str, dict[str, int]]]:
    records: list[tuple[str, str, dict[str, int]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if fields[0].casefold() == "name" or "train/val/test" in line.casefold():
            continue
        if len(fields) < 5:
            raise ValueError(f"{path}:{line_number}: expected at least 5 columns")
        try:
            class_label, lighting, location, split_id = map(int, fields[1:5])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-integer metadata column") from exc
        if split_id not in _SPLIT_IDS:
            raise ValueError(f"{path}:{line_number}: invalid experiment split {split_id}")
        records.append(
            (
                fields[0],
                _SPLIT_IDS[split_id],
                {"image_class": class_label, "lighting": lighting, "location": location},
            )
        )
    if not records:
        raise FileNotFoundError(f"no ExDark split rows found in {path}")
    return records


def _parse_annotation(path: Path) -> list[tuple[int, float, float, float, float]]:
    objects: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        fields = stripped.split()
        if len(fields) < 5:
            raise ValueError(f"{path}:{line_number}: expected class and [left top width height]")
        class_id = _CLASS_LOOKUP.get(fields[0].casefold())
        if class_id is None:
            raise ValueError(f"{path}:{line_number}: unknown ExDark class {fields[0]!r}")
        try:
            left, top, width, height = map(float, fields[1:5])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid bounding box") from exc
        objects.append((class_id, left, top, width, height))
    return objects


def convert_exdark_dataset(
    images_root: Path,
    annotations_root: Path,
    split_file: Path,
    output_root: Path,
    *,
    transfer: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
    image_backend: IOBackend = "auto",
) -> dict[str, int]:
    """Convert all 12 classes using the official imageclasslist split column."""

    images_root, annotations_root, split_file, output_root = map(
        Path, (images_root, annotations_root, split_file, output_root)
    )
    if not images_root.is_dir() or not annotations_root.is_dir() or not split_file.is_file():
        raise FileNotFoundError(
            "ExDark requires existing --images-root, --annotations-root, and --split-file"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    image_index, annotation_index = _index_images(images_root), _index_annotations(annotations_root)
    counters = {"train": 0, "val": 0, "test": 0, "objects": 0, "clipped": 0}
    seen_images: set[Path] = set()
    manifests: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for listed_name, split, metadata in _parse_split_file(split_file):
        keys = (listed_name.casefold(), Path(listed_name).stem.casefold())
        source_image = next((image_index[key] for key in keys if key in image_index), None)
        if source_image is None:
            raise FileNotFoundError(f"ExDark split references missing image: {listed_name}")
        if source_image in seen_images:
            raise ValueError(f"ExDark image occurs more than once in split file: {source_image}")
        seen_images.add(source_image)
        source_annotation = next(
            (annotation_index[key] for key in keys if key in annotation_index), None
        )
        if source_annotation is None:
            raise FileNotFoundError(f"missing ExDark annotation for {source_image}")
        width, height = image_size(source_image, backend=image_backend)
        rows: list[str] = []
        for class_id, left, top, box_width, box_height in _parse_annotation(source_annotation):
            x, y, w, h, clipped = xywh_to_normalized_yolo(
                left, top, box_width, box_height, width, height
            )
            rows.append(format_yolo_row(class_id, (x, y, w, h)))
            counters["objects"] += 1
            counters["clipped"] += int(clipped)

        destination_image = output_root / "images" / split / source_image.name
        destination_label = output_root / "labels" / split / f"{source_image.stem}.txt"
        transfer_file(source_image, destination_image, mode=transfer, overwrite=overwrite)
        if destination_label.exists() and not overwrite:
            raise FileExistsError(destination_label)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n"
        )
        counters[split] += 1
        manifests[split].append(
            canonical_json(
                {
                    "image": destination_image.relative_to(output_root).as_posix(),
                    "label": destination_label.relative_to(output_root).as_posix(),
                    "metadata": metadata,
                    "schema_version": 1,
                    "source_annotation": source_annotation.relative_to(annotations_root).as_posix(),
                    "source_image": source_image.relative_to(images_root).as_posix(),
                    "split": split,
                }
            )
        )

    for split, rows in manifests.items():
        path = output_root / "manifests" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")
    write_dataset_yaml(output_root, EXDARK_CLASS_NAMES, ("train", "val", "test"))
    (output_root / "overlap_mapping.json").write_text(
        json.dumps(
            {
                "full_set_names": list(EXDARK_CLASS_NAMES),
                "visdrone_overlap": EXDARK_VISDRONE_OVERLAP,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "conversion_summary.json").write_text(
        canonical_json(counters) + "\n", encoding="utf-8", newline="\n"
    )
    return counters
