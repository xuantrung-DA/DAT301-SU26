"""Convert AU-AIR JSON annotations to YOLO with sequence-safe splits."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .common import canonical_json, iter_images, stable_seed, transfer_file
from .detection import format_yolo_row, write_dataset_yaml, xywh_to_normalized_yolo
from .imaging import IOBackend, image_size

DEFAULT_AUAIR_CLASSES = (
    "human",
    "car",
    "van",
    "truck",
    "bike",
    "motorbike",
    "bus",
    "trailer",
)
_NAME_ALIASES = {
    "person": "human",
    "people": "human",
    "bicycle": "bike",
    "motorcycle": "motorbike",
    "trailar": "trailer",  # Spelling used by some AU-AIR releases/tools.
}
_VISDRONE_NAMES = {
    "human": ("pedestrian", 0, True),
    "car": ("car", 3, False),
    "van": ("van", 4, False),
    "truck": ("truck", 5, False),
    "bike": ("bicycle", 2, False),
    "motorbike": ("motor", 9, False),
    "bus": ("bus", 8, False),
}


def _normalize_name(name: object) -> str:
    value = str(name).strip().casefold().replace("_", "-")
    return _NAME_ALIASES.get(value, value)


def _categories(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("categories", DEFAULT_AUAIR_CLASSES)
    if isinstance(raw, Mapping):
        try:
            ordered = [raw[key] for key in sorted(raw, key=lambda key: int(key))]
        except (TypeError, ValueError):
            ordered = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        ordered = list(raw)
    else:
        raise ValueError("AU-AIR categories must be a list or mapping")
    names: list[str] = []
    for entry in ordered:
        if isinstance(entry, Mapping):
            entry = entry.get("name", entry.get("label"))
        if entry is None:
            raise ValueError("AU-AIR category entry has no name")
        names.append(_normalize_name(entry))
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate AU-AIR categories after normalization: {names}")
    return tuple(names)


def _records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("annotations", payload.get("images"))
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, Mapping):
        records = []
        for image_name, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"AU-AIR record {image_name!r} is not an object")
            record = dict(value)
            record.setdefault("image_name", image_name)
            records.append(record)
    else:
        raise ValueError("AU-AIR JSON must contain an annotations list/mapping")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("every AU-AIR annotation record must be an object")
    return [dict(record) for record in records]


def _image_name(record: Mapping[str, Any]) -> str:
    value = record.get("image_name", record.get("file_name", record.get("filename")))
    if not value:
        raise ValueError("AU-AIR record is missing image_name")
    return Path(str(value)).name


def _boxes(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = record.get("bbox", record.get("boxes", record.get("objects")))
    if value is None:
        # Some converted schemas put object annotations under this field while
        # top-level records live under "images".
        value = record.get("annotations", [])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"AU-AIR {_image_name(record)} bbox field must be a list of objects")
    return value  # type: ignore[return-value]


def _raw_class(box: Mapping[str, Any]) -> object:
    for key in ("class", "category", "category_id", "class_id", "label"):
        if key in box:
            return box[key]
    raise ValueError("AU-AIR box is missing a class/category")


def _class_base(records: Sequence[Mapping[str, Any]], class_count: int, requested: str) -> int:
    if requested in {"0", "1"}:
        return int(requested)
    numeric: list[int] = []
    for record in records:
        for box in _boxes(record):
            value = _raw_class(box)
            if isinstance(value, bool):
                continue
            try:
                numeric.append(int(value))
            except (TypeError, ValueError):
                pass
    if 0 in numeric:
        return 0
    if class_count in numeric and numeric and min(numeric) >= 1:
        return 1
    # Official AU-AIR JSON uses zero-based indices.  An explicit CLI override is
    # available for third-party exports whose observed subset is ambiguous.
    return 0


def _class_id(value: object, classes: Sequence[str], base: int) -> int:
    if isinstance(value, str) and not re.fullmatch(r"[-+]?\d+", value.strip()):
        normalized = _normalize_name(value)
        try:
            return classes.index(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown AU-AIR class name {value!r}") from exc
    try:
        result = int(value) - base
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid AU-AIR class id {value!r}") from exc
    if result < 0 or result >= len(classes):
        raise ValueError(f"AU-AIR class id {value!r} maps outside 0..{len(classes)-1}")
    return result


def _xywh(box: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw_bbox = box.get("bbox")
    if isinstance(raw_bbox, Sequence) and not isinstance(raw_bbox, (str, bytes)):
        if len(raw_bbox) != 4:
            raise ValueError("AU-AIR bbox array must have four values")
        return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]
    aliases = (
        ("left", "top", "width", "height"),
        ("x", "y", "w", "h"),
        ("xmin", "ymin", "width", "height"),
    )
    for keys in aliases:
        if all(key in box for key in keys):
            return tuple(float(box[key]) for key in keys)  # type: ignore[return-value]
    if all(key in box for key in ("xmin", "ymin", "xmax", "ymax")):
        left, top = float(box["xmin"]), float(box["ymin"])
        return left, top, float(box["xmax"]) - left, float(box["ymax"]) - top
    raise ValueError("AU-AIR box has no recognized xywh/xyxy coordinates")


def sequence_id(image_name: str, record: Mapping[str, Any] | None = None) -> str:
    """Return an explicit video id or remove the final frame-number suffix."""

    if record:
        for key in ("sequence", "sequence_id", "video", "video_id"):
            if record.get(key) not in (None, ""):
                return str(record[key])
    stem = Path(image_name).stem
    match = re.match(r"^(.+?)[_-](\d+)$", stem)
    return match.group(1) if match else stem


def _assign_sequences(
    sequence_counts: Mapping[str, int],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    if any(value < 0 for value in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError("split ratios must be non-negative and sum to one")
    split_names = ("train", "val", "test")
    total = sum(sequence_counts.values())
    targets = {name: total * ratio for name, ratio in zip(split_names, ratios)}
    assigned = Counter({name: 0 for name in split_names})
    result: dict[str, str] = {}
    groups = sorted(
        sequence_counts,
        key=lambda name: (-sequence_counts[name], stable_seed(seed, "auair-sequence", name), name),
    )
    for group in groups:
        # Fill the split with the greatest remaining target.  Whole videos are
        # indivisible, guaranteeing that adjacent frames cannot leak.
        selected = max(
            split_names,
            key=lambda name: (targets[name] - assigned[name], -split_names.index(name)),
        )
        result[group] = selected
        assigned[selected] += sequence_counts[group]
    return result


def convert_auair_dataset(
    images_root: Path,
    annotations_json: Path,
    output_root: Path,
    *,
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 3407,
    class_id_base: Literal["auto", "0", "1"] = "auto",
    transfer: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
    image_backend: IOBackend = "auto",
) -> dict[str, int]:
    """Convert AU-AIR and allocate whole video sequences, never adjacent frames."""

    images_root, annotations_json, output_root = map(
        Path, (images_root, annotations_json, output_root)
    )
    if not images_root.is_dir() or not annotations_json.is_file():
        raise FileNotFoundError("AU-AIR requires an images directory and annotations.json")
    payload = json.loads(annotations_json.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("AU-AIR annotation JSON root must be an object")
    classes, records = _categories(payload), _records(payload)
    base = _class_base(records, len(classes), class_id_base)
    image_index: dict[str, Path] = {}
    for image in iter_images(images_root, recursive=True):
        key = image.name.casefold()
        if key in image_index:
            raise ValueError(f"duplicate AU-AIR image filename: {image.name}")
        image_index[key] = image

    normalized_records: list[tuple[dict[str, Any], Path, str]] = []
    sequence_counts: Counter[str] = Counter()
    seen: set[Path] = set()
    for record in records:
        name = _image_name(record)
        image = image_index.get(name.casefold())
        if image is None:
            raise FileNotFoundError(f"AU-AIR annotations reference missing image: {name}")
        if image in seen:
            raise ValueError(f"AU-AIR image has duplicate annotation records: {image}")
        seen.add(image)
        group = sequence_id(name, record)
        normalized_records.append((record, image, group))
        sequence_counts[group] += 1
    if not normalized_records:
        raise FileNotFoundError("AU-AIR annotation file contains no images")
    assignments = _assign_sequences(sequence_counts, split_ratios, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    counters = {
        "train": 0,
        "val": 0,
        "test": 0,
        "objects": 0,
        "clipped": 0,
        "invalid_boxes": 0,
    }
    manifests: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for record, source_image, group in sorted(normalized_records, key=lambda item: item[1].name.casefold()):
        split = assignments[group]
        width, height = image_size(source_image, backend=image_backend)
        json_width = record.get("image_width", record.get("width"))
        json_height = record.get("image_height", record.get("height"))
        if json_width is not None and int(json_width) != width:
            raise ValueError(f"AU-AIR JSON/image width mismatch for {source_image}")
        if json_height is not None and int(json_height) != height:
            raise ValueError(f"AU-AIR JSON/image height mismatch for {source_image}")
        rows: list[str] = []
        for box in _boxes(record):
            class_id = _class_id(_raw_class(box), classes, base)
            left, top, box_width, box_height = _xywh(box)
            if box_width <= 0 or box_height <= 0:
                counters["invalid_boxes"] += 1
                continue
            x, y, w, h, clipped = xywh_to_normalized_yolo(
                left, top, box_width, box_height, width, height
            )
            rows.append(format_yolo_row(class_id, (x, y, w, h)))
            counters["objects"] += 1
            counters["clipped"] += int(clipped)
        destination_image = output_root / "images" / split / group / source_image.name
        destination_label = output_root / "labels" / split / group / f"{source_image.stem}.txt"
        transfer_file(source_image, destination_image, mode=transfer, overwrite=overwrite)
        if destination_label.exists() and not overwrite:
            raise FileExistsError(destination_label)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n"
        )
        counters[split] += 1
        metadata = {
            key: record[key]
            for key in ("altitude", "time", "timestamp", "weather")
            if key in record
        }
        manifests[split].append(
            canonical_json(
                {
                    "image": destination_image.relative_to(output_root).as_posix(),
                    "label": destination_label.relative_to(output_root).as_posix(),
                    "metadata": metadata,
                    "schema_version": 1,
                    "sequence": group,
                    "split": split,
                    "source_image": source_image.relative_to(images_root).as_posix(),
                }
            )
        )

    for split, rows in manifests.items():
        path = output_root / "manifests" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")
    write_dataset_yaml(output_root, classes, ("train", "val", "test"))
    overlap = {
        str(index): {
            "auair_class": name,
            "visdrone_class": _VISDRONE_NAMES[name][0],
            "visdrone_id": _VISDRONE_NAMES[name][1],
            "ambiguous": _VISDRONE_NAMES[name][2],
        }
        for index, name in enumerate(classes)
        if name in _VISDRONE_NAMES
    }
    (output_root / "overlap_mapping.json").write_text(
        json.dumps({"full_set_names": list(classes), "visdrone_overlap": overlap}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "class_id_base": base,
        "classes": list(classes),
        "counters": counters,
        "seed": seed,
        "sequence_assignments": assignments,
        "split_ratios": list(split_ratios),
    }
    (output_root / "conversion_summary.json").write_text(
        canonical_json(summary) + "\n", encoding="utf-8", newline="\n"
    )
    return counters
