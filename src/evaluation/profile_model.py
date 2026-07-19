"""Report parameters, GFLOPs and checkpoint size for the Pareto table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from inference import load_detector, load_generator


class FullEnhancer(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, image):
        residual, features = self.model.enhancer(image)
        return self.model.dgrdm(image, residual, features)[0]


def profile(module: torch.nn.Module, image: torch.Tensor) -> dict[str, float]:
    from thop import profile as thop_profile

    macs, parameters = thop_profile(module.eval(), inputs=(image,), verbose=False)
    return {"parameters": int(parameters), "gflops": float(2.0 * macs / 1e9)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("runs/profile/model_profile.json"))
    args = parser.parse_args()
    generator = load_generator(args.generator, torch.device("cpu"))
    detector = load_detector(args.yolo, args.joint_checkpoint).model.eval()
    image = torch.zeros(1, 3, args.imgsz, args.imgsz)
    gate_parameters = sum(parameter.numel() for parameter in generator.gate.parameters())
    report = {
        "image_size": args.imgsz,
        "enhancer_full": profile(FullEnhancer(generator), image),
        "gate_parameters": gate_parameters,
        "generator_total_parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "detector": profile(detector, image),
        "checkpoint_mb": args.generator.stat().st_size / 1024**2,
    }
    report["full_pipeline_parameters"] = report["generator_total_parameters"] + report["detector"]["parameters"]
    report["full_pipeline_gflops_always_full"] = report["enhancer_full"]["gflops"] + report["detector"]["gflops"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
