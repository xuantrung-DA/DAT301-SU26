"""Low-light degradation and simple enhancement utilities.

All image functions use OpenCV BGR uint8 images.
The degradation is deterministic when a seed is passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LowLightConfig:
    beta_range: Tuple[float, float]
    gamma_range: Tuple[float, float]
    noise_sigma_range: Tuple[float, float]
    jpeg_quality_range: Tuple[int, int]
    blur_prob: float


LEVEL_CONFIGS = {
    "LL1": LowLightConfig((0.55, 0.75), (1.35, 1.80), (0.0, 5.0), (82, 96), 0.10),
    "LL2": LowLightConfig((0.35, 0.55), (1.80, 2.40), (5.0, 12.0), (70, 90), 0.20),
    "LL3": LowLightConfig((0.20, 0.35), (2.40, 3.20), (10.0, 20.0), (55, 80), 0.35),
}


def _illumination_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Create a smooth non-uniform illumination mask in [0.55, 1.05]."""
    small_h = max(4, height // 96)
    small_w = max(4, width // 96)
    coarse = rng.uniform(0.55, 1.05, size=(small_h, small_w)).astype(np.float32)
    mask = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    # Strong blur makes the illumination transition natural.
    k = max(31, (min(height, width) // 12) | 1)  # odd kernel size
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.50, 1.08)[..., None]


def _maybe_blur(img_float: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Tiny blur/no camera shake proxy. Keep mild to avoid unrealistic labels.
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img_float, (3, 3), 0)
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1, :] = 1.0 / 3.0
    return cv2.filter2D(img_float, -1, kernel)


def _jpeg_compress(img_uint8: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    ok, enc = cv2.imencode(".jpg", img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_uint8
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img_uint8


def degrade_image_bgr(img_bgr: np.ndarray, level: str = "LL2", seed: int | None = None) -> np.ndarray:
    """Generate a synthetic low-light version of one BGR image.

    Formula, simplified:
        I_low = clip((I * beta * M) ** gamma + noise)

    Args:
        img_bgr: OpenCV BGR uint8 image.
        level: LL1, LL2, or LL3.
        seed: deterministic seed.
    Returns:
        BGR uint8 low-light image.
    """
    level = level.upper()
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"Unknown level '{level}'. Use one of {list(LEVEL_CONFIGS)}")

    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("img_bgr must be a valid HxWx3 image")

    cfg = LEVEL_CONFIGS[level]
    rng = np.random.default_rng(seed)

    img = img_bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    beta = rng.uniform(*cfg.beta_range)
    gamma = rng.uniform(*cfg.gamma_range)
    sigma = rng.uniform(*cfg.noise_sigma_range) / 255.0
    quality = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))

    mask = _illumination_mask(h, w, rng)
    out = np.clip(img * beta * mask, 0.0, 1.0)
    out = np.power(out, gamma)

    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    if rng.random() < cfg.blur_prob:
        out = _maybe_blur(out, rng)

    out_uint8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out_uint8 = _jpeg_compress(out_uint8, quality)
    return out_uint8


def gamma_enhance_bgr(img_bgr: np.ndarray, gamma: float = 0.60) -> np.ndarray:
    """Cheap gamma brightening baseline. gamma < 1 brightens."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    img = img_bgr.astype(np.float32) / 255.0
    out = np.power(np.clip(img, 0, 1), gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def clahe_enhance_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE on L channel in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    font_scale = max(0.5, min(h, w) / 900)
    thickness = max(1, int(round(font_scale * 2)))
    cv2.rectangle(out, (0, 0), (min(w, 280), int(34 * font_scale + 14)), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(28 * font_scale + 5)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def make_comparison_grid(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Create a horizontal grid after resizing all images to same height."""
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    base_h = min(img.shape[0] for img in images)
    prepared = []
    for img, label in zip(images, labels):
        scale = base_h / img.shape[0]
        new_w = int(round(img.shape[1] * scale))
        resized = cv2.resize(img, (new_w, base_h), interpolation=cv2.INTER_AREA)
        prepared.append(_add_label(resized, label))
    return cv2.hconcat(prepared)
