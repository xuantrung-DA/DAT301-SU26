"""Convert official UAVDT detection ground truth to split-preserving YOLO."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping

from .common import canonical_json, iter_images, transfer_file
from .detection import format_yolo_row, write_dataset_yaml, xywh_to_normalized_yolo
from .imaging import IOBackend, image_size

UAVDT_CLASS_NAMES = ("car", "truck", "bus")
UAVDT_VISDRONE_OVERLAP = {
    "car": {"visdrone_class": "car", "visdrone_id": 3},
    "truck": {"visdrone_class": "truck", "visdrone_id": 5},
    "bus": {"visdrone_class": "bus", "visdrone_id": 8},
}
_FIELD_COUNT = 9


def _is_supervisely_layout(root: Path) -> bool:
    """Return whether ``root`` is the DatasetNinja/Kaggle export we downloaded."""

    return all((root / split / kind).is_dir() for split in ("train", "test") for kind in ("img", "ann"))


def _supervisely_sequence(image_name: str, payload: Mapping[str, Any]) -> str:
    for tag in payload.get("tags", []):
        if isinstance(tag, Mapping) and str(tag.get("name", "")).casefold() == "sequence":
            value = str(tag.get("value", "")).strip()
            if value:
                return value
    match = re.match(r"^([A-Z]\d{4})_img\d+", Path(image_name).stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot derive UAVDT sequence from {image_name}")
    return match.group(1).upper()


def _supervisely_box(obj: Mapping[str, Any]) -> tuple[int, float, float, float, float]:
    title = str(obj.get("classTitle", "")).strip().casefold()
    try:
        class_id = UAVDT_CLASS_NAMES.index(title)
    except ValueError as exc:
        raise ValueError(f"unknown UAVDT classTitle {title!r}") from exc
    points = obj.get("points")
    exterior = points.get("exterior") if isinstance(points, Mapping) else None
    if not isinstance(exterior, list) or len(exterior) != 2:
        raise ValueError("UAVDT rectangle must contain two exterior points")
    (x1, y1), (x2, y2) = exterior
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return class_id, left, top, right - left, bottom - top


def _convert_supervisely_uavdt(
    source_root: Path,
    output_root: Path,
    *,
    transfer: Literal["copy", "hardlink", "symlink"],
    overwrite: bool,
    image_backend: IOBackend,
) -> dict[str, int]:
    """Convert the actual Kaggle archive's per-image Supervisely JSON files."""

    output_root.mkdir(parents=True, exist_ok=True)
    counters = {
        "train": 0,
        "test": 0,
        "objects": 0,
        "clipped": 0,
        "sequences": 0,
        "skipped_non_detection_frames": 0,
        "skipped_non_detection_objects": 0,
        "skipped_duplicate_objects": 0,
    }
    manifests: dict[str, list[str]] = {"train": [], "test": []}
    all_sequences: set[str] = set()

    for split in ("train", "test"):
        images_root = source_root / split / "img"
        annotations_root = source_root / split / "ann"
        for source_image in iter_images(images_root):
            annotation = annotations_root / f"{source_image.name}.json"
            if not annotation.is_file():
                raise FileNotFoundError(f"missing UAVDT annotation for {source_image.name}")
            payload = json.loads(annotation.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"UAVDT annotation root must be an object: {annotation}")
            sequence = _supervisely_sequence(source_image.name, payload)
            objects = payload.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError(f"UAVDT objects must be a list: {annotation}")
            # DatasetNinja's archive combines UAVDT-DET M* sequences with the
            # UAVDT-SOT S* subset.  S* has only a generic ``vehicle`` class and
            # is not part of the three-class detection benchmark.
            if not sequence.upper().startswith("M"):
                counters["skipped_non_detection_frames"] += 1
                counters["skipped_non_detection_objects"] += len(objects)
                continue
            all_sequences.add(sequence)
            width, height = image_size(source_image, backend=image_backend)
            declared = payload.get("size")
            if isinstance(declared, Mapping):
                if int(declared.get("width", width)) != width or int(declared.get("height", height)) != height:
                    raise ValueError(f"UAVDT JSON/image size mismatch for {source_image.name}")

            rows: list[str] = []
            seen_rows: set[str] = set()
            object_metadata: list[dict[str, Any]] = []
            for obj in objects:
                if not isinstance(obj, Mapping):
                    raise ValueError(f"UAVDT object must be a mapping: {annotation}")
                class_id, left, top, box_width, box_height = _supervisely_box(obj)
                x, y, w, h, clipped = xywh_to_normalized_yolo(
                    left, top, box_width, box_height, width, height
                )
                row = format_yolo_row(class_id, (x, y, w, h))
                if row in seen_rows:
                    counters["skipped_duplicate_objects"] += 1
                    continue
                seen_rows.add(row)
                rows.append(row)
                counters["objects"] += 1
                counters["clipped"] += int(clipped)
                object_metadata.append({"class_id": class_id, "source_id": obj.get("id")})

            destination_image = output_root / "images" / split / sequence / source_image.name
            destination_label = output_root / "labels" / split / sequence / f"{source_image.stem}.txt"
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
                        "objects": object_metadata,
                        "schema_version": 1,
                        "sequence": sequence,
                        "source_annotation": annotation.relative_to(source_root).as_posix(),
                        "source_image": source_image.relative_to(source_root).as_posix(),
                        "split": split,
                    }
                )
            )

    counters["sequences"] = len(all_sequences)
    for split, rows in manifests.items():
        path = output_root / "manifests" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    write_dataset_yaml(output_root, UAVDT_CLASS_NAMES, ("train", "test"))
    (output_root / "overlap_mapping.json").write_text(
        json.dumps(UAVDT_VISDRONE_OVERLAP, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "conversion_summary.json").write_text(
        canonical_json(counters) + "\n", encoding="utf-8", newline="\n"
    )
    return counters


def _find_images_root(root: Path) -> Path:
    candidates = (
        root / "UAV-benchmark-M",
        root / "UAVDT_Benchmark_M",
        root / "UAV-benchmark-MOTD_v1.0" / "UAV-benchmark-M",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot find UAVDT frame root; tried: {', '.join(str(path) for path in candidates)}"
    )


def _find_gt_root(root: Path) -> Path:
    candidates = (
        root / "UAV-benchmark-MOTD_v1.0" / "GT",
        root / "GT",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot find UAVDT GT root; tried: {', '.join(str(path) for path in candidates)}"
    )


def _official_sequences(root: Path) -> dict[str, list[str]]:
    attr_root = root / "M_attr"
    result: dict[str, list[str]] = {}
    for split in ("train", "test"):
        directory = attr_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"missing official UAVDT split attributes: {directory}")
        sequences = sorted(
            {path.name.split("_attr", 1)[0] for path in directory.glob("*_attr.txt")},
            key=str.casefold,
        )
        if not sequences:
            raise FileNotFoundError(f"no *_attr.txt files in {directory}")
        result[split] = sequences
    overlap = set(result["train"]) & set(result["test"])
    if overlap:
        raise ValueError(f"UAVDT sequences occur in train and test: {sorted(overlap)}")
    return result


def _parse_gt(path: Path) -> dict[int, list[tuple[int, float, float, float, float, int, int]]]:
    by_frame: dict[int, list[tuple[int, float, float, float, float, int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), start=1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) != _FIELD_COUNT:
                raise ValueError(f"{path}:{line_number}: expected {_FIELD_COUNT} CSV fields")
            try:
                frame_id = int(float(fields[0]))
                left, top, width, height = map(float, fields[2:6])
                out_of_view, occlusion, category = (int(float(value)) for value in fields[6:9])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
            if category not in (1, 2, 3):
                raise ValueError(f"{path}:{line_number}: invalid object category {category}")
            by_frame[frame_id].append(
                (category - 1, left, top, width, height, out_of_view, occlusion)
            )
    return by_frame


def _frame_id(path: Path) -> int:
    match = re.fullmatch(r"img(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unexpected UAVDT frame filename: {path.name}")
    return int(match.group(1))


def convert_uavdt_dataset(
    source_root: Path,
    output_root: Path,
    *,
    transfer: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
    image_backend: IOBackend = "auto",
) -> dict[str, int]:
    """Convert DET ``*_gt_whole.txt`` using M_attr's official train/test groups."""

    source_root, output_root = Path(source_root), Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if _is_supervisely_layout(source_root):
        return _convert_supervisely_uavdt(
            source_root,
            output_root,
            transfer=transfer,
            overwrite=overwrite,
            image_backend=image_backend,
        )
    images_root, gt_root = _find_images_root(source_root), _find_gt_root(source_root)
    official = _official_sequences(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counters = {"train": 0, "test": 0, "objects": 0, "clipped": 0, "sequences": 0}
    manifests: dict[str, list[str]] = {"train": [], "test": []}

    for split, sequences in official.items():
        for sequence in sequences:
            sequence_dir = images_root / sequence
            gt_path = gt_root / f"{sequence}_gt_whole.txt"
            if not sequence_dir.is_dir() or not gt_path.is_file():
                raise FileNotFoundError(
                    f"UAVDT sequence {sequence} requires {sequence_dir} and {gt_path}"
                )
            ground_truth = _parse_gt(gt_path)
            images = list(iter_images(sequence_dir))
            if not images:
                raise FileNotFoundError(f"no UAVDT frames in {sequence_dir}")
            available_frames = {_frame_id(image) for image in images}
            extra_gt = sorted(set(ground_truth) - available_frames)
            if extra_gt:
                raise ValueError(
                    f"UAVDT GT references missing frames in {sequence}: {extra_gt[:10]}"
                )
            counters["sequences"] += 1
            for source_image in images:
                frame_id = _frame_id(source_image)
                width, height = image_size(source_image, backend=image_backend)
                rows: list[str] = []
                object_metadata: list[dict[str, int]] = []
                for class_id, left, top, box_width, box_height, out_of_view, occlusion in ground_truth.get(frame_id, []):
                    x, y, w, h, clipped = xywh_to_normalized_yolo(
                        left, top, box_width, box_height, width, height
                    )
                    rows.append(format_yolo_row(class_id, (x, y, w, h)))
                    object_metadata.append(
                        {"class_id": class_id, "occlusion": occlusion, "out_of_view": out_of_view}
                    )
                    counters["objects"] += 1
                    counters["clipped"] += int(clipped)
                destination_image = output_root / "images" / split / sequence / source_image.name
                destination_label = (
                    output_root / "labels" / split / sequence / f"{source_image.stem}.txt"
                )
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
                            "frame_id": frame_id,
                            "image": destination_image.relative_to(output_root).as_posix(),
                            "label": destination_label.relative_to(output_root).as_posix(),
                            "objects": object_metadata,
                            "schema_version": 1,
                            "sequence": sequence,
                            "split": split,
                        }
                    )
                )

    for split, rows in manifests.items():
        path = output_root / "manifests" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    write_dataset_yaml(output_root, UAVDT_CLASS_NAMES, ("train", "test"))
    (output_root / "overlap_mapping.json").write_text(
        json.dumps(UAVDT_VISDRONE_OVERLAP, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "conversion_summary.json").write_text(
        canonical_json(counters) + "\n", encoding="utf-8", newline="\n"
    )
    return counters
