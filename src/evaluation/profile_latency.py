"""Reproducible single-image p50/p95/FPS/GPU-memory latency harness."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from inference import enhance_bgr_with_details, letterbox_bgr, load_detector, load_generator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    return {
        "mean_ms": mean,
        "std_ms": float(array.std()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps_from_mean": float(1000.0 / mean) if mean > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--bright-yolo")
    parser.add_argument("--detector-route-threshold", type=float, default=0.30)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--mode", choices=["detector", "adaptive", "bypass", "light", "full"], default="detector")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("runs/profile"))
    args = parser.parse_args()
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode != "detector" and args.generator is None:
        raise ValueError("--generator is required unless --mode detector is selected")
    generator = load_generator(args.generator, device) if args.generator else None
    detector = load_detector(args.yolo, args.joint_checkpoint)
    bright_detector = load_detector(args.bright_yolo) if args.bright_yolo else None
    force_mode = None if args.mode == "adaptive" else args.mode

    def run_one(path: Path) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        synchronize()
        total_start = time.perf_counter()
        preprocess_start = total_start
        image, _ = letterbox_bgr(image, args.imgsz)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        if args.mode == "detector":
            enhanced = image
            enhancer_ms = 0.0
            illumination = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0)
            diagnostics = {"gate_mode": "bypass", "residual_l1": 0.0, "illumination_mean": illumination}
        else:
            enhance_start = time.perf_counter()
            enhanced, diagnostics = enhance_bgr_with_details(generator, image, device, force_mode=force_mode)
            synchronize()
            enhancer_ms = (time.perf_counter() - enhance_start) * 1000.0
        active_detector = (
            bright_detector
            if bright_detector is not None and diagnostics["illumination_mean"] >= args.detector_route_threshold
            else detector
        )
        detector_route = "bright" if active_detector is bright_detector else "lowlight"
        detector_start = time.perf_counter()
        active_detector.predict(enhanced, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        synchronize()
        detector_ms = (time.perf_counter() - detector_start) * 1000.0
        return {
            "image": str(path),
            "gate_mode": diagnostics["gate_mode"],
            "detector_route": detector_route,
            "preprocess_ms": preprocess_ms,
            "enhancer_ms": enhancer_ms,
            "detector_ms": detector_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
            "residual_l1": diagnostics["residual_l1"],
            "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        }

    for index in range(args.warmup):
        run_one(paths[index % len(paths)])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = [run_one(paths[index % len(paths)]) for index in range(args.measurements)]
    with (args.output / f"latency_{args.mode}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", *rows[0].keys()])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"iteration": index, **row})

    modes = {mode: sum(row["gate_mode"] == mode for row in rows) / len(rows) for mode in ("bypass", "light", "full")}
    summary = {
        "protocol": {"batch": 1, "warmup": args.warmup, "measurements": args.measurements, "image_size": args.imgsz},
        "mode": args.mode,
        "end_to_end": summarize([row["total_ms"] for row in rows]),
        "enhancer": summarize([row["enhancer_ms"] for row in rows]),
        "preprocess": summarize([row["preprocess_ms"] for row in rows]),
        "detector": summarize([row["detector_ms"] for row in rows]),
        "gate_rates": modes,
        "detector_route_rates": {
            route: sum(row["detector_route"] == route for row in rows) / len(rows)
            for route in ("bright", "lowlight")
        },
        "peak_gpu_memory_mb": max(row["gpu_memory_mb"] for row in rows),
        "hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }
    (args.output / f"latency_{args.mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
