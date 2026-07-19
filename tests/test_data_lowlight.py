from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladd_uav.data.imaging import ImageDependencyError, has_opencv
from ladd_uav.data.lowlight import (
    LEVEL_SPECS,
    LowLightParameters,
    assign_llmix_levels,
    sample_parameters,
    synthesize_dataset,
    synthesize_image,
)


class LowLightTests(unittest.TestCase):
    def test_parameter_ranges_and_darkening(self) -> None:
        image = np.full((24, 32, 3), 200, dtype=np.uint8)
        for level, spec in LEVEL_SPECS.items():
            params = sample_parameters(level, 12345)
            self.assertTrue(spec.alpha[0] <= params.alpha <= spec.alpha[1])
            self.assertTrue(spec.gamma[0] <= params.gamma <= spec.gamma[1])
            self.assertTrue(spec.read_sigma[0] <= params.read_sigma <= spec.read_sigma[1])
            self.assertTrue(all(1 - spec.color_jitter <= gain <= 1 + spec.color_jitter for gain in params.color_gains))
            output = synthesize_image(image, params)
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, np.uint8)
            self.assertLess(float(output.mean()), float(image.mean()))

    def test_llmix_is_exactly_twenty_percent_clean_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            image_dir, label_dir = dataset / "images" / "train", dataset / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index in range(10):
                Image.new("RGB", (20, 16), (80 + index, 100, 120)).save(image_dir / f"{index:02}.png")
                (label_dir / f"{index:02}.txt").write_text(
                    "0 0.50000000 0.50000000 0.20000000 0.25000000\n", encoding="utf-8"
                )

            first, second = root / "first", root / "second"
            stats = synthesize_dataset(dataset, first, variants=("LLMix",), base_seed=3407)
            synthesize_dataset(dataset, second, variants=("llmix",), base_seed=3407)
            self.assertEqual(stats["LLMix"]["train"].clean_images, 2)
            manifest_one = first / "LLMix" / "manifests" / "train.jsonl"
            manifest_two = second / "LLMix" / "manifests" / "train.jsonl"
            self.assertEqual(manifest_one.read_bytes(), manifest_two.read_bytes())
            rows = [json.loads(line) for line in manifest_one.read_text().splitlines()]
            self.assertEqual(sum(row["is_clean"] for row in rows), 2)
            for row in rows:
                self.assertIn("alpha", row)
                self.assertIn("gamma", row)
                self.assertIn("noise", row)
                self.assertIn("color", row)
                self.assertIn("blur", row)
                output_relative = Path(row["output_image"]).relative_to("LLMix")
                first_bytes = (first / "LLMix" / output_relative).read_bytes()
                second_bytes = (second / "LLMix" / output_relative).read_bytes()
                self.assertEqual(first_bytes, second_bytes)
                label_relative = Path("labels/train") / (Path(row["source_image"]).stem + ".txt")
                self.assertEqual(
                    (first / "LLMix" / label_relative).read_bytes(),
                    (dataset / "labels" / "train" / label_relative.name).read_bytes(),
                )

    def test_assignment_is_stable_and_opencv_failure_is_explicit(self) -> None:
        keys = [f"train/{index}.png" for index in range(10)]
        first = assign_llmix_levels(keys, 301)
        second = assign_llmix_levels(list(reversed(keys)), 301)
        self.assertEqual(first, second)
        self.assertEqual(sum(level == "CLEAN" for level in first.values()), 2)
        if not has_opencv():
            params = LowLightParameters(
                level="LL1",
                seed=1,
                alpha=0.6,
                gamma=1.6,
                read_sigma=0.01,
                shot_photons=1000.0,
                color_gains=(1.0, 1.0, 1.0),
                blur_probability=1.0,
                blur_applied=True,
                blur_length=3,
                blur_angle_degrees=0.0,
            )
            with self.assertRaisesRegex(ImageDependencyError, "OpenCV"):
                synthesize_image(np.full((8, 8, 3), 128, np.uint8), params, blur_backend="opencv")


if __name__ == "__main__":
    unittest.main()
