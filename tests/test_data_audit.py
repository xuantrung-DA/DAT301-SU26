from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladd_uav.data.audit import audit_yolo_dataset


class AuditTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 100), (10, 20, 30)).save(root / "images" / "train" / "small.png")
        (root / "labels" / "train" / "small.txt").write_text(
            "0 0.05000000 0.05000000 0.10000000 0.10000000\n", encoding="utf-8"
        )
        Image.new("RGB", (100, 100), (30, 20, 10)).save(root / "images" / "val" / "medium.png")
        (root / "labels" / "val" / "medium.txt").write_text(
            "1 0.50000000 0.50000000 0.50000000 0.50000000\n", encoding="utf-8"
        )

    def test_valid_sizes_use_original_image_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            report = audit_yolo_dataset(root)
            self.assertTrue(report.ok)
            self.assertEqual(report.valid_objects, 2)
            self.assertEqual(report.object_sizes["coco_size"]["small_area_lt_32sq"], 1)
            self.assertEqual(report.object_sizes["coco_size"]["medium_32sq_to_96sq"], 1)
            self.assertEqual(
                report.object_sizes["secondary_small_object"]["either_width_or_height_lt_16"], 1
            )

    def test_corrupt_missing_and_orphan_labels_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            (root / "labels" / "train" / "small.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n0 nan 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            Image.new("RGB", (10, 10)).save(root / "images" / "train" / "missing.png")
            (root / "labels" / "train" / "orphan.txt").write_text("", encoding="utf-8")
            report = audit_yolo_dataset(root, check_content_leakage=False)
            self.assertFalse(report.ok)
            self.assertEqual(report.corrupt_label_files, 1)
            self.assertEqual(report.corrupt_label_rows, 2)
            self.assertEqual(report.missing_labels, 1)
            self.assertEqual(report.orphan_labels, 1)
            self.assertEqual({issue.kind for issue in report.issues}, {
                "corrupt_label_row", "duplicate_label_row", "missing_label", "orphan_label"
            })


if __name__ == "__main__":
    unittest.main()
