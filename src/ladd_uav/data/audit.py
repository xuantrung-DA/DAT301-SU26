"""Integrity and small-object audit for split-preserving YOLO datasets."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .common import canonical_json, iter_images, sha256_file
from .imaging import IOBackend, image_size


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    path: str
    message: str
    split: str | None = None
    line: int | None = None


@dataclass
class _ObjectAccumulator:
    total: int = 0
    coco_small: int = 0
    coco_medium: int = 0
    coco_large: int = 0
    width_lt_16: int = 0
    height_lt_16: int = 0
    either_side_lt_16: int = 0
    both_sides_lt_16: int = 0
    width_sum: float = 0.0
    height_sum: float = 0.0
    area_sum: float = 0.0
    min_width: float = math.inf
    max_width: float = 0.0
    min_height: float = math.inf
    max_height: float = 0.0
    min_area: float = math.inf
    max_area: float = 0.0
    area_bins: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    def add(self, width: float, height: float) -> None:
        area = width * height
        self.total += 1
        self.coco_small += int(area < 32.0**2)
        self.coco_medium += int(32.0**2 <= area < 96.0**2)
        self.coco_large += int(area >= 96.0**2)
        self.width_lt_16 += int(width < 16.0)
        self.height_lt_16 += int(height < 16.0)
        self.either_side_lt_16 += int(width < 16.0 or height < 16.0)
        self.both_sides_lt_16 += int(width < 16.0 and height < 16.0)
        self.width_sum += width
        self.height_sum += height
        self.area_sum += area
        self.min_width = min(self.min_width, width)
        self.max_width = max(self.max_width, width)
        self.min_height = min(self.min_height, height)
        self.max_height = max(self.max_height, height)
        self.min_area = min(self.min_area, area)
        self.max_area = max(self.max_area, area)
        # Descriptive bins complement the required COCO S/M/L histogram.
        if area < 16.0**2:
            index = 0
        elif area < 32.0**2:
            index = 1
        elif area < 64.0**2:
            index = 2
        elif area < 96.0**2:
            index = 3
        else:
            index = 4
        self.area_bins[index] += 1

    def to_dict(self) -> dict[str, object]:
        total = self.total

        def mean(value: float) -> float | None:
            return value / total if total else None

        def finite_or_none(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "area_histogram_px2": {
                "[0,256)": self.area_bins[0],
                "[256,1024)": self.area_bins[1],
                "[1024,4096)": self.area_bins[2],
                "[4096,9216)": self.area_bins[3],
                "[9216,inf)": self.area_bins[4],
            },
            "coco_size": {
                "large_area_ge_96sq": self.coco_large,
                "medium_32sq_to_96sq": self.coco_medium,
                "small_area_lt_32sq": self.coco_small,
            },
            "dimensions_px": {
                "area": {
                    "max": self.max_area if total else None,
                    "mean": mean(self.area_sum),
                    "min": finite_or_none(self.min_area),
                },
                "height": {
                    "max": self.max_height if total else None,
                    "mean": mean(self.height_sum),
                    "min": finite_or_none(self.min_height),
                },
                "width": {
                    "max": self.max_width if total else None,
                    "mean": mean(self.width_sum),
                    "min": finite_or_none(self.min_width),
                },
            },
            "secondary_small_object": {
                "both_width_and_height_lt_16": self.both_sides_lt_16,
                "either_width_or_height_lt_16": self.either_side_lt_16,
                "height_lt_16": self.height_lt_16,
                "width_lt_16": self.width_lt_16,
            },
            "total": total,
        }


@dataclass
class AuditReport:
    dataset_root: str
    splits: list[str]
    class_count: int | None
    images: int = 0
    label_files: int = 0
    valid_objects: int = 0
    corrupt_label_files: int = 0
    corrupt_label_rows: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    unreadable_images: int = 0
    cross_split_duplicates: int = 0
    issues: list[AuditIssue] = field(default_factory=list)
    issues_truncated: int = 0
    object_sizes: dict[str, object] = field(default_factory=dict)
    per_class: dict[str, dict[str, object]] = field(default_factory=dict)
    per_split: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.corrupt_label_files,
                self.missing_labels,
                self.orphan_labels,
                self.unreadable_images,
                self.cross_split_duplicates,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "class_count": self.class_count,
            "counters": {
                "corrupt_label_files": self.corrupt_label_files,
                "corrupt_label_rows": self.corrupt_label_rows,
                "cross_split_duplicates": self.cross_split_duplicates,
                "images": self.images,
                "label_files": self.label_files,
                "missing_labels": self.missing_labels,
                "orphan_labels": self.orphan_labels,
                "unreadable_images": self.unreadable_images,
                "valid_objects": self.valid_objects,
            },
            "dataset_root": self.dataset_root,
            "issues": [asdict(issue) for issue in self.issues],
            "issues_truncated": self.issues_truncated,
            "object_sizes": self.object_sizes,
            "ok": self.ok,
            "per_class": self.per_class,
            "per_split": self.per_split,
            "splits": self.splits,
        }

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(self.to_dict()) + "\n", encoding="utf-8", newline="\n")


def _discover_splits(root: Path, requested: Iterable[str] | None) -> dict[str, tuple[Path, Path]]:
    images_root, labels_root = root / "images", root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"expected {images_root} and {labels_root}")
    if requested is None:
        names = sorted(
            [directory.name for directory in images_root.iterdir() if directory.is_dir()],
            key=str.casefold,
        )
        if not names and list(iter_images(images_root)):
            names = ["all"]
    else:
        names = list(dict.fromkeys(requested))
    if not names:
        raise FileNotFoundError(f"no image splits found below {images_root}")
    result: dict[str, tuple[Path, Path]] = {}
    for name in names:
        if name == "all" and not (images_root / name).is_dir():
            image_dir, label_dir = images_root, labels_root
        else:
            image_dir, label_dir = images_root / name, labels_root / name
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing image/label directory for split {name!r}")
        result[name] = (image_dir, label_dir)
    return result


def _parse_yolo_row(
    row: str,
    *,
    class_count: int | None,
    boundary_tolerance: float,
) -> tuple[int, float, float, float, float]:
    values = row.split()
    if len(values) != 5:
        raise ValueError(f"expected 5 whitespace-separated fields, got {len(values)}")
    try:
        raw_class, center_x, center_y, width, height = (float(value) for value in values)
    except ValueError as exc:
        raise ValueError("contains a non-numeric field") from exc
    if not all(math.isfinite(value) for value in (raw_class, center_x, center_y, width, height)):
        raise ValueError("contains a non-finite value")
    if not raw_class.is_integer():
        raise ValueError("class id must be an integer")
    class_id = int(raw_class)
    if class_id < 0 or (class_count is not None and class_id >= class_count):
        expected = f"0..{class_count - 1}" if class_count is not None else "non-negative"
        raise ValueError(f"class id {class_id} is outside {expected}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("normalized width and height must be positive")
    if width > 1.0 + boundary_tolerance or height > 1.0 + boundary_tolerance:
        raise ValueError("normalized width or height exceeds 1")
    if not (-boundary_tolerance <= center_x <= 1.0 + boundary_tolerance):
        raise ValueError("normalized x center is outside [0,1]")
    if not (-boundary_tolerance <= center_y <= 1.0 + boundary_tolerance):
        raise ValueError("normalized y center is outside [0,1]")
    x1, x2 = center_x - width / 2.0, center_x + width / 2.0
    y1, y2 = center_y - height / 2.0, center_y + height / 2.0
    if x1 < -boundary_tolerance or y1 < -boundary_tolerance:
        raise ValueError("box begins outside the normalized image")
    if x2 > 1.0 + boundary_tolerance or y2 > 1.0 + boundary_tolerance:
        raise ValueError("box ends outside the normalized image")
    return class_id, center_x, center_y, width, height


def audit_yolo_dataset(
    dataset_root: Path,
    *,
    splits: Iterable[str] | None = None,
    class_count: int | None = 10,
    check_content_leakage: bool = True,
    max_issues: int = 1000,
    boundary_tolerance: float = 1e-6,
    image_backend: IOBackend = "auto",
) -> AuditReport:
    """Audit pairing/labels/splits and calculate object sizes in source pixels."""

    if class_count is not None and class_count <= 0:
        raise ValueError("class_count must be positive or None")
    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")
    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be non-negative")
    dataset_root = Path(dataset_root)
    split_dirs = _discover_splits(dataset_root, splits)
    report = AuditReport(
        dataset_root=str(dataset_root.resolve()),
        splits=list(split_dirs),
        class_count=class_count,
    )
    overall_sizes = _ObjectAccumulator()
    class_sizes: dict[int, _ObjectAccumulator] = {}
    content_seen: dict[str, tuple[str, Path]] = {}

    def issue(kind: str, path: Path, message: str, split: str, line: int | None = None) -> None:
        item = AuditIssue(kind=kind, path=str(path), message=message, split=split, line=line)
        if len(report.issues) < max_issues:
            report.issues.append(item)
        else:
            report.issues_truncated += 1

    for split, (image_dir, label_dir) in split_dirs.items():
        images = list(iter_images(image_dir, recursive=True))
        labels = sorted(label_dir.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
        expected_label_relatives = {
            image.relative_to(image_dir).with_suffix(".txt").as_posix().casefold() for image in images
        }
        split_sizes = _ObjectAccumulator()
        split_corrupt_files = 0
        split_valid_objects = 0
        for orphan in labels:
            relative = orphan.relative_to(label_dir).as_posix().casefold()
            if relative not in expected_label_relatives:
                report.orphan_labels += 1
                issue("orphan_label", orphan, "label has no matching image", split)

        for image in images:
            report.images += 1
            relative = image.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file():
                report.missing_labels += 1
                issue("missing_label", label, f"no label for image {image.name}", split)
                continue
            report.label_files += 1
            try:
                pixel_width, pixel_height = image_size(image, backend=image_backend)
            except ValueError as exc:
                report.unreadable_images += 1
                issue("unreadable_image", image, str(exc), split)
                continue

            if check_content_leakage:
                digest = sha256_file(image)
                previous = content_seen.get(digest)
                if previous is not None and previous[0] != split:
                    report.cross_split_duplicates += 1
                    issue(
                        "cross_split_duplicate",
                        image,
                        f"byte-identical to {previous[1]} in split {previous[0]}",
                        split,
                    )
                else:
                    content_seen[digest] = (split, image)

            corrupt_file = False
            seen_rows: set[tuple[int, float, float, float, float]] = set()
            try:
                rows = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                report.corrupt_label_rows += 1
                corrupt_file = True
                issue("unreadable_label", label, str(exc), split)
                rows = []
            for line_number, row in enumerate(rows, start=1):
                if not row.strip():
                    continue
                try:
                    parsed = _parse_yolo_row(
                        row,
                        class_count=class_count,
                        boundary_tolerance=boundary_tolerance,
                    )
                except ValueError as exc:
                    report.corrupt_label_rows += 1
                    corrupt_file = True
                    issue("corrupt_label_row", label, str(exc), split, line_number)
                    continue
                if parsed in seen_rows:
                    report.corrupt_label_rows += 1
                    corrupt_file = True
                    issue("duplicate_label_row", label, "duplicate object row", split, line_number)
                    continue
                seen_rows.add(parsed)
                class_id, _, _, normalized_width, normalized_height = parsed
                object_width = normalized_width * pixel_width
                object_height = normalized_height * pixel_height
                overall_sizes.add(object_width, object_height)
                split_sizes.add(object_width, object_height)
                class_sizes.setdefault(class_id, _ObjectAccumulator()).add(
                    object_width, object_height
                )
                report.valid_objects += 1
                split_valid_objects += 1
            if corrupt_file:
                report.corrupt_label_files += 1
                split_corrupt_files += 1

        report.per_split[split] = {
            "corrupt_label_files": split_corrupt_files,
            "images": len(images),
            "label_files": len(labels),
            "object_sizes": split_sizes.to_dict(),
            "valid_objects": split_valid_objects,
        }

    report.object_sizes = overall_sizes.to_dict()
    report.per_class = {
        str(class_id): accumulator.to_dict()
        for class_id, accumulator in sorted(class_sizes.items())
    }
    return report
