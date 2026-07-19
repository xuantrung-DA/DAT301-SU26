"""Run the core B3/M0/M1/M2 image-front-end ablation with one protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--bright-yolo")
    parser.add_argument("--detector-route-threshold", type=float, default=0.30)
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--small-area", type=float, default=32.0**2)
    parser.add_argument("--tiny-side", type=float, default=16.0)
    parser.add_argument("--domain")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/ablation"))
    args = parser.parse_args()
    methods = ["none", "gamma", "clahe"] + (["m0", "m1", "m2"] if args.generator else [])
    for method in methods:
        command = [
            sys.executable, "-m", "evaluation.evaluate_detector",
            "--images", str(args.images), "--labels", str(args.labels),
            "--yolo", args.yolo, "--enhancement", method,
            "--seed", str(args.seed), "--imgsz", str(args.imgsz),
            "--confidence", str(args.confidence), "--nms-iou", str(args.nms_iou),
            "--small-area", str(args.small_area), "--tiny-side", str(args.tiny_side),
            "--output", str(args.output),
        ]
        if args.domain:
            command += ["--domain", args.domain]
        if args.bright_yolo:
            command += [
                "--bright-yolo", args.bright_yolo,
                "--detector-route-threshold", str(args.detector_route_threshold),
            ]
        if args.generator:
            command += ["--generator", str(args.generator)]
        if args.joint_checkpoint:
            command += ["--joint-checkpoint", str(args.joint_checkpoint)]
        if args.max_images:
            command += ["--max-images", str(args.max_images)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
