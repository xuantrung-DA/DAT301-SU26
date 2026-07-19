from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models import DOMAIN_NAMES, LADDEnhancer, LearnedDomainRouter


def _model_from_config(config: dict) -> LADDEnhancer:
    model = config.get("model", {})
    return LADDEnhancer(
        channels=model.get("channels", [24, 48, 96]),
        fusion_blocks=model.get("fusion_blocks", 3),
        max_residual=model.get("max_residual", 0.35),
        foreground_alpha=model.get("foreground_alpha", 1.0),
        background_alpha=model.get("background_alpha", 0.25),
        light_alpha=model.get("light_alpha", 0.35),
    )


def load_generator(checkpoint: Path, device: torch.device) -> LADDEnhancer:
    state = torch.load(checkpoint, map_location="cpu")
    config = state.get("config", {}) if isinstance(state, dict) else {}
    model = _model_from_config(config).to(device)
    try:
        model.load_state_dict(state.get("generator", state))
    except RuntimeError as exc:
        raise RuntimeError("The checkpoint predates the LADD-UAV proposal architecture.") from exc
    return model.eval()


def load_detector(weights: str, joint_checkpoint: Path | None = None):
    from ultralytics import YOLO

    detector = YOLO(weights)
    if joint_checkpoint:
        state = torch.load(joint_checkpoint, map_location="cpu")
        if "detector" in state:
            detector.model.load_state_dict(state["detector"])
    return detector


def load_domain_router(checkpoint: Path, device: torch.device) -> tuple[LearnedDomainRouter, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if tuple(state.get("domains", ())) != DOMAIN_NAMES:
        raise ValueError("Domain router checkpoint has an incompatible class order")
    model = LearnedDomainRouter().to(device)
    model.load_state_dict(state["model"])
    return model.eval(), int(state.get("size", 160))


@torch.inference_mode()
def route_domain_bgr(model: LearnedDomainRouter, image_bgr: np.ndarray, device: torch.device, size: int = 160) -> dict[str, Any]:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    tensor = image_to_tensor(resized, device)
    route, probabilities = model.route(tensor)
    values = probabilities[0].cpu().tolist()
    return {"domain_route": DOMAIN_NAMES[int(route[0])], "domain_probabilities": dict(zip(DOMAIN_NAMES, values))}


def image_to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)


def letterbox_bgr(image_bgr: np.ndarray, image_size: int, fill: int = 114) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image_bgr.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (image_size - resized_width) // 2, (image_size - resized_height) // 2
    canvas = np.full((image_size, image_size, 3), fill, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "original_width": width, "original_height": height}


def unletterbox_xyxy(boxes: np.ndarray, metadata: dict[str, float]) -> np.ndarray:
    output = boxes.astype(np.float32, copy=True)
    output[:, [0, 2]] = (output[:, [0, 2]] - metadata["pad_x"]) / metadata["scale"]
    output[:, [1, 3]] = (output[:, [1, 3]] - metadata["pad_y"]) / metadata["scale"]
    output[:, [0, 2]] = np.clip(output[:, [0, 2]], 0, metadata["original_width"])
    output[:, [1, 3]] = np.clip(output[:, [1, 3]], 0, metadata["original_height"])
    return output


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    rgb = tensor[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.inference_mode()
def enhance_bgr_with_details(
    model: LADDEnhancer,
    image_bgr: np.ndarray,
    device: torch.device,
    detector_confidence: float | None = None,
    force_mode: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = image_to_tensor(image_bgr, device)
    confidence = None if detector_confidence is None else image.new_tensor([detector_confidence])
    output, details = model.forward_with_details(
        image,
        detector_confidence=confidence,
        hard_gate=True,
        force_mode=force_mode,
    )
    probabilities = details["gate_probabilities"][0].detach().cpu().tolist()
    metadata = {
        "gate_mode": ["bypass", "light", "full"][int(details["gate_mode"][0])],
        "gate_probabilities": {"bypass": probabilities[0], "light": probabilities[1], "full": probabilities[2]},
        "illumination_mean": float(details["gate_features"][0, 0].detach().cpu()),
        "dark_ratio": float(details["gate_features"][0, 1].detach().cpu()),
        "noise_proxy": float(details["gate_features"][0, 2].detach().cpu()),
        "residual_l1": float(details["applied_residual"].abs().mean().detach().cpu()),
    }
    return _tensor_to_bgr(output), metadata


@torch.inference_mode()
def enhance_bgr(model: LADDEnhancer, image_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    output, _ = enhance_bgr_with_details(model, image_bgr, device)
    return output


def serialize_ultralytics(result) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections
    names = result.names
    for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()):
        class_index = int(class_id.item())
        detections.append({
            "class_id": class_index,
            "class_name": names[class_index],
            "confidence": float(confidence),
            "xyxy": [float(value) for value in xyxy],
        })
    return detections


def save_detection_json(
    path: Path,
    source: Path,
    detections: list[dict],
    elapsed_ms: float,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "source": str(source),
        "inference_ms": elapsed_ms,
        "detections": detections,
        "diagnostics": diagnostics or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
