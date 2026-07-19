"""Export branchable LADD-UAV ONNX graphs and validate numerical parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from inference import load_detector, load_generator


class GateExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gate = model.gate

    def forward(self, image, detector_confidence):
        probabilities, features, _ = self.gate(image, detector_confidence, hard=False)
        return probabilities, features


class LightEnhancerExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enhancer = model.enhancer
        self.light_alpha = model.light_alpha

    def forward(self, image):
        residual, _ = self.enhancer(image)
        return torch.clamp(image + self.light_alpha * residual, 0.0, 1.0)


class FullEnhancerExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enhancer = model.enhancer
        self.dgrdm = model.dgrdm

    def forward(self, image):
        residual, features = self.enhancer(image)
        enhanced, details = self.dgrdm(image, residual, features)
        return enhanced, details["predicted_heatmap"]


def export_module(module, inputs, path: Path, input_names: list[str], output_names: list[str], opset: int) -> None:
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    for name in input_names:
        if name == "image":
            dynamic_axes[name].update({2: "height", 3: "width"})
    for name in output_names:
        if name in {"enhanced", "objectness_heatmap", "gate_features_map"}:
            dynamic_axes[name].update({2: "height", 3: "width"})
    torch.onnx.export(
        module.eval(), inputs, path, input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False,
    )


def onnx_max_error(path: Path, inputs: dict[str, np.ndarray], expected: tuple[np.ndarray, ...]) -> list[float]:
    import onnxruntime as ort

    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, inputs)
    return [float(np.max(np.abs(left - right))) for left, right in zip(actual, expected)]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("exports/ladd_uav"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_generator(args.generator, torch.device("cpu"))
    image = torch.rand(1, 3, args.imgsz, args.imgsz)
    confidence = torch.zeros(1)

    modules = [
        ("gate.onnx", GateExport(model), (image, confidence), ["image", "detector_confidence"], ["gate_probabilities", "gate_features"]),
        ("enhancer_light.onnx", LightEnhancerExport(model), image, ["image"], ["enhanced"]),
        ("enhancer_full.onnx", FullEnhancerExport(model), image, ["image"], ["enhanced", "objectness_heatmap"]),
    ]
    parity: dict[str, list[float]] = {}
    for filename, module, inputs, input_names, output_names in modules:
        path = args.output / filename
        with torch.no_grad():
            output = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        expected_tensors = output if isinstance(output, tuple) else (output,)
        export_module(module, inputs, path, input_names, output_names, args.opset)
        runtime_inputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, inputs if isinstance(inputs, tuple) else (inputs,))
        }
        parity[filename] = onnx_max_error(
            path, runtime_inputs, tuple(tensor.detach().cpu().numpy() for tensor in expected_tensors)
        )

    detector = load_detector(args.yolo, args.joint_checkpoint)
    exported_detector = detector.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=True, simplify=True,
        project=str(args.output), name="yolo11n_ladd",
    )
    detector_output = args.output / "yolo11n.onnx"
    shutil.copy2(Path(exported_detector), detector_output)
    report = {
        "generator_checkpoint": str(args.generator),
        "detector_source": args.yolo,
        "detector_onnx": str(detector_output),
        "opset": args.opset,
        "image_size": args.imgsz,
        "max_absolute_error": parity,
        "branching": "Run gate first; bypass skips enhancer, light uses enhancer_light, full uses enhancer_full.",
    }
    (args.output / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
