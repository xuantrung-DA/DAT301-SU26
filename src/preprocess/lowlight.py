"""Deterministic physics-inspired low-light synthesis for LowLight-VisDrone.

All image functions use OpenCV BGR uint8 images.  The returned manifest fields
record every sampled parameter required by the proposal's reproducibility
protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class LowLightConfig:
    exposure_range: tuple[float, float]
    gamma_range: tuple[float, float]
    read_noise_range: tuple[float, float]
    shot_peak_range: tuple[float, float]
    channel_jitter: float
    jpeg_quality_range: tuple[int, int]
    blur_prob: float


LEVEL_CONFIGS = {
    "LL1": LowLightConfig((0.55, 0.75), (1.4, 2.0), (0.005, 0.015), (80.0, 160.0), 0.05, (88, 98), 0.10),
    "LL2": LowLightConfig((0.30, 0.55), (2.0, 3.0), (0.010, 0.030), (35.0, 90.0), 0.10, (75, 94), 0.20),
    "LL3": LowLightConfig((0.10, 0.30), (3.0, 5.0), (0.020, 0.050), (15.0, 45.0), 0.15, (60, 88), 0.30),
}


def _illumination_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    small_h = max(4, height // 96)
    small_w = max(4, width // 96)
    coarse = rng.uniform(0.75, 1.05, size=(small_h, small_w)).astype(np.float32)
    mask = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    kernel = max(31, (min(height, width) // 12) | 1)
    return np.clip(cv2.GaussianBlur(mask, (kernel, kernel), 0), 0.70, 1.08)[..., None]


def _motion_blur(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    if rng.random() < 0.5:
        return cv2.GaussianBlur(image, (3, 3), 0), "gaussian_3x3"
    kernel = np.zeros((5, 5), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[2, :] = 0.2
        name = "horizontal_5"
    else:
        kernel[:, 2] = 0.2
        name = "vertical_5"
    return cv2.filter2D(image, -1, kernel), name


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def degrade_image_bgr(
    img_bgr: np.ndarray,
    level: str = "LL2",
    seed: int | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Apply ``clip(a * I**gamma * c + n_shot + n_read, 0, 1)``."""
    level = level.upper()
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"Unknown level '{level}'. Use one of {list(LEVEL_CONFIGS)}")
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a valid HxWx3 image")

    cfg = LEVEL_CONFIGS[level]
    rng = np.random.default_rng(seed)
    exposure = float(rng.uniform(*cfg.exposure_range))
    gamma = float(rng.uniform(*cfg.gamma_range))
    read_sigma = float(rng.uniform(*cfg.read_noise_range))
    shot_peak = float(rng.uniform(*cfg.shot_peak_range))
    channel_gain = rng.uniform(1.0 - cfg.channel_jitter, 1.0 + cfg.channel_jitter, size=3).astype(np.float32)
    jpeg_quality = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))
    blur_applied = bool(rng.random() < cfg.blur_prob)

    image = img_bgr.astype(np.float32) / 255.0
    illumination = _illumination_mask(image.shape[0], image.shape[1], rng)
    signal = exposure * np.power(np.clip(image, 0.0, 1.0), gamma)
    signal = np.clip(signal * channel_gain.reshape(1, 1, 3) * illumination, 0.0, 1.0)
    shot = rng.poisson(signal * shot_peak).astype(np.float32) / shot_peak - signal
    read = rng.normal(0.0, read_sigma, size=signal.shape).astype(np.float32)
    degraded = np.clip(signal + shot + read, 0.0, 1.0)

    blur_type = "none"
    if blur_applied:
        degraded, blur_type = _motion_blur(degraded, rng)
    output = _jpeg_compress(np.clip(degraded * 255.0, 0, 255).astype(np.uint8), jpeg_quality)
    metadata = {
        "level": level,
        "seed": int(seed) if seed is not None else None,
        "exposure": exposure,
        "gamma": gamma,
        "read_noise_sigma": read_sigma,
        "shot_peak": shot_peak,
        "channel_gain_b": float(channel_gain[0]),
        "channel_gain_g": float(channel_gain[1]),
        "channel_gain_r": float(channel_gain[2]),
        "blur_applied": blur_applied,
        "blur_type": blur_type,
        "jpeg_quality": jpeg_quality,
        "config": asdict(cfg),
    }
    return (output, metadata) if return_metadata else output


def gamma_enhance_bgr(img_bgr: np.ndarray, gamma: float = 0.60) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    image = img_bgr.astype(np.float32) / 255.0
    return np.clip(np.power(np.clip(image, 0, 1), gamma) * 255.0, 0, 255).astype(np.uint8)


def clahe_enhance_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return cv2.cvtColor(cv2.merge([clahe.apply(lightness), a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    output = img_bgr.copy()
    height, width = output.shape[:2]
    font_scale = max(0.5, min(height, width) / 900)
    thickness = max(1, int(round(font_scale * 2)))
    cv2.rectangle(output, (0, 0), (min(width, 280), int(34 * font_scale + 14)), (0, 0, 0), -1)
    cv2.putText(output, text, (8, int(28 * font_scale + 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def make_comparison_grid(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    base_height = min(image.shape[0] for image in images)
    prepared = []
    for image, label in zip(images, labels):
        new_width = int(round(image.shape[1] * base_height / image.shape[0]))
        prepared.append(_add_label(cv2.resize(image, (new_width, base_height), interpolation=cv2.INTER_AREA), label))
    return cv2.hconcat(prepared)
