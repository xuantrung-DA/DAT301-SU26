from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladd_uav.data.auair import convert_auair_dataset
from ladd_uav.data.exdark import convert_exdark_dataset
from ladd_uav.data.lol import prepare_lol_v1
from ladd_uav.data.uavdt import convert_uavdt_dataset


def make_image(path: Path, color: tuple[int, int, int] = (100, 100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)


class ExternalDatasetTests(unittest.TestCase):
    def test_lol_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for official, color in (("our485", 20), ("eval15", 40)):
                make_image(root / "raw" / official / "low" / "a.png", (color, color, color))
                make_image(root / "raw" / official / "high" / "a.png", (color + 50,) * 3)
            counters = prepare_lol_v1(root / "raw", root / "out")
            self.assertEqual(counters, {"train": 1, "test": 1})
            self.assertTrue((root / "out" / "train" / "low" / "a.png").is_file())

    def test_auair_sequences_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for sequence_index in range(3):
                for frame in range(2):
                    name = f"video{sequence_index}_{frame:06}.jpg"
                    make_image(root / "images" / name, (20 * sequence_index, frame, 50))
                    records.append(
                        {
                            "image_name": name,
                            "image_width": 40,
                            "image_height": 30,
                            "bbox": [{"class": 1, "left": 4, "top": 3, "width": 10, "height": 8}],
                        }
                    )
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps({"categories": list(("human", "car", "van", "truck", "bike", "motorbike", "bus", "trailer")), "annotations": records}),
                encoding="utf-8",
            )
            convert_auair_dataset(root / "images", annotations, root / "out")
            summary = json.loads((root / "out" / "conversion_summary.json").read_text())
            self.assertEqual(set(summary["sequence_assignments"]), {"video0", "video1", "video2"})
            for sequence, split in summary["sequence_assignments"].items():
                self.assertEqual(len(list((root / "out" / "images" / split / sequence).glob("*.jpg"))), 2)

    def test_exdark_official_split_and_bbgt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_image(root / "images" / "Bicycle" / "bike.jpg")
            annotation = root / "groundtruth" / "Bicycle" / "bike.jpg.txt"
            annotation.parent.mkdir(parents=True)
            annotation.write_text("% bbGt version=3\nBicycle 4 3 10 8 0 0 0 0 0 0 0\n", encoding="utf-8")
            split_file = root / "imageclasslist.txt"
            split_file.write_text("bike.jpg 1 1 2 1\n", encoding="utf-8")
            result = convert_exdark_dataset(
                root / "images", root / "groundtruth", split_file, root / "out"
            )
            self.assertEqual(result["train"], 1)
            self.assertTrue((root / "out" / "labels" / "train" / "bike.txt").read_text().startswith("0 "))

    def test_uavdt_official_sequence_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            for split, sequence in (("train", "M0101"), ("test", "M0201")):
                attr = root / "M_attr" / split / f"{sequence}_attr.txt"
                attr.parent.mkdir(parents=True)
                attr.write_text("0,0,0,0,0,0,0,0,0,0\n", encoding="utf-8")
                make_image(root / "UAV-benchmark-M" / sequence / "img000001.jpg")
                gt = root / "UAV-benchmark-MOTD_v1.0" / "GT" / f"{sequence}_gt_whole.txt"
                gt.parent.mkdir(parents=True, exist_ok=True)
                gt.write_text("1,1,4,3,10,8,1,1,1\n", encoding="utf-8")
            result = convert_uavdt_dataset(root, root.parent / "out")
            self.assertEqual(result["train"], 1)
            self.assertEqual(result["test"], 1)
            self.assertTrue(
                (root.parent / "out" / "labels" / "train" / "M0101" / "img000001.txt").is_file()
            )

    def test_uavdt_supervisely_kaggle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            for split, sequence in (("train", "M0101"), ("test", "M0201")):
                name = f"{sequence}_img000001.jpg"
                make_image(root / split / "img" / name)
                annotation = root / split / "ann" / f"{name}.json"
                annotation.parent.mkdir(parents=True)
                annotation.write_text(
                    json.dumps(
                        {
                            "tags": [{"name": "sequence", "value": sequence}],
                            "size": {"width": 40, "height": 30},
                            "objects": [
                                {
                                    "id": 1,
                                    "classTitle": "car",
                                    "points": {"exterior": [[4, 3], [14, 11]]},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            result = convert_uavdt_dataset(root, root.parent / "out")
            self.assertEqual(result["train"], 1)
            self.assertEqual(result["test"], 1)
            self.assertEqual(result["sequences"], 2)
            label = root.parent / "out" / "labels" / "train" / "M0101" / "M0101_img000001.txt"
            self.assertTrue(label.read_text().startswith("0 "))


if __name__ == "__main__":
    unittest.main()
