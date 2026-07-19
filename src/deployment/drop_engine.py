from __future__ import annotations

import argparse
from pathlib import Path
from queue import Queue
from threading import Event
import time

import cv2
import torch

from inference import enhance_bgr_with_details, letterbox_bgr, load_detector, load_generator, save_detection_json, serialize_ultralytics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DropEngine:
    def __init__(self, input_dir: Path, output_dir: Path, generator_checkpoint: Path | None, yolo_weights: str, confidence: float, force_mode: str | None = None, image_size: int = 640, bright_yolo: str | None = None, route_threshold: float = 0.30):
        self.input_dir, self.output_dir = input_dir, output_dir; self.confidence = confidence
        self.input_dir.mkdir(parents=True, exist_ok=True); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = load_generator(generator_checkpoint, self.device) if generator_checkpoint else None; self.detector = load_detector(yolo_weights); self.bright_detector = load_detector(bright_yolo) if bright_yolo else None; self.route_threshold = route_threshold; self.queue, self.seen = Queue(), set(); self.force_mode = force_mode; self.image_size = image_size

    def scan(self):
        for path in sorted(self.input_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path not in self.seen:
                self.seen.add(path); self.queue.put(path)

    def process(self, path: Path):
        image = cv2.imread(str(path))
        if image is None: return
        image, letterbox = letterbox_bgr(image, self.image_size)
        if self.device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        illumination = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0)
        if self.generator is None:
            enhanced, diagnostics = image, {"gate_mode": "bypass", "residual_l1": 0.0, "illumination_mean": illumination}
        else:
            enhanced, diagnostics = enhance_bgr_with_details(self.generator, image, self.device, force_mode=self.force_mode)
        if self.device.type == "cuda": torch.cuda.synchronize()
        enhancement_ms = (time.perf_counter() - start) * 1000
        detector_start = time.perf_counter()
        active_detector = self.bright_detector if self.bright_detector is not None and illumination >= self.route_threshold else self.detector
        diagnostics["detector_route"] = "bright" if active_detector is self.bright_detector else "lowlight"
        prediction = active_detector.predict(enhanced, conf=self.confidence, verbose=False)[0]
        if self.device.type == "cuda": torch.cuda.synchronize()
        detector_ms = (time.perf_counter() - detector_start) * 1000
        elapsed = (time.perf_counter() - start) * 1000
        annotated = prediction.plot(); cv2.imwrite(str(self.output_dir/f"{path.stem}_detected.jpg"), annotated)
        cv2.imwrite(str(self.output_dir/f"{path.stem}_enhanced.jpg"), enhanced)
        diagnostics.update({"enhancer_ms": enhancement_ms, "detector_ms": detector_ms, "letterbox": letterbox})
        save_detection_json(self.output_dir/f"{path.stem}.json", path, serialize_ultralytics(prediction), elapsed, diagnostics)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[DONE] {path.name}: {elapsed:.1f} ms | gate={diagnostics['gate_mode']} | enhancer={enhancement_ms:.1f} ms")

    def run(self, interval: float, once: bool = False):
        print(f"Watching {self.input_dir.resolve()} (Ctrl+C to stop)")
        try:
            while True:
                self.scan()
                while not self.queue.empty(): self.process(self.queue.get())
                if once:
                    return
                time.sleep(interval)
        except KeyboardInterrupt: print("Stopped.")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=Path("input_drop")); parser.add_argument("--output", type=Path, default=Path("output_results")); parser.add_argument("--generator", type=Path); parser.add_argument("--yolo", default="yolo11n.pt"); parser.add_argument("--bright-yolo"); parser.add_argument("--route-threshold", type=float, default=0.30); parser.add_argument("--conf", type=float, default=0.20); parser.add_argument("--interval", type=float, default=0.5); parser.add_argument("--once", action="store_true", help="Process current files and exit"); parser.add_argument("--force-mode", choices=["bypass", "light", "full"]); parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(); DropEngine(args.input, args.output, args.generator, args.yolo, args.conf, args.force_mode, args.imgsz, args.bright_yolo, args.route_threshold).run(args.interval, args.once)


if __name__ == "__main__": main()
