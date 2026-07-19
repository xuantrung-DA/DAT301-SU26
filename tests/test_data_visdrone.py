from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladd_uav.data.visdrone import (
    annotation_to_yolo,
    convert_visdrone_dataset,
    discover_visdrone_splits,
    parse_visdrone_annotation,
)


class VisDroneConversionTests(unittest.TestCase):
    def test_official_split_conversion_clips_and_maps_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw" / "VisDrone2019-DET-train"
            (raw / "images").mkdir(parents=True)
            (raw / "annotations").mkdir()
            Image.new("RGB", (100, 50), (120, 130, 140)).save(raw / "images" / "frame.jpg")
            (raw / "annotations" / "frame.txt").write_text(
                "10,10,20,20,1,4,0,0\n"  # car -> class 3
                "-5,-5,10,10,1,1,0,0\n"  # clipped pedestrian
                "0,0,10,10,1,0,0,0\n"  # ignored region
                "0,0,10,10,1,11,0,0\n",  # others
                encoding="utf-8",
            )

            discovered = discover_visdrone_splits(root / "raw")
            self.assertEqual(list(discovered), ["train"])
            output = root / "yolo"
            stats = convert_visdrone_dataset(root / "raw", output)
            self.assertEqual(stats["train"].images, 1)
            self.assertEqual(stats["train"].objects, 2)
            self.assertEqual(stats["train"].ignored_objects, 2)
            self.assertEqual(stats["train"].clipped_objects, 1)
            rows = (output / "labels" / "train" / "frame.txt").read_text().splitlines()
            self.assertEqual(rows[0], "3 0.20000000 0.40000000 0.20000000 0.40000000")
            self.assertEqual(rows[1], "0 0.02500000 0.05000000 0.05000000 0.10000000")
            self.assertTrue((output / "images" / "train" / "frame.jpg").is_file())
            self.assertIn("train: images/train", (output / "dataset.yaml").read_text())

    def test_annotation_parser_rejects_bad_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 8"):
            parse_visdrone_annotation("1,2,3")
        annotation = parse_visdrone_annotation("0,0,10,10,1,1,0,0,")
        converted = annotation_to_yolo(annotation, 10, 10)
        self.assertIsNotNone(converted)

    def test_missing_train_annotation_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "VisDrone2019-DET-train"
            (raw / "images").mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(raw / "images" / "missing.png")
            with self.assertRaises(FileNotFoundError):
                convert_visdrone_dataset(raw, Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
