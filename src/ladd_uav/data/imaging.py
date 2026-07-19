"""Image I/O and portable blur helpers with an explicitly optional OpenCV.

Importing this module never requires :mod:`cv2`.  The default I/O backend
prefers Pillow, and the default motion-blur implementation is pure NumPy so
installing OpenCV later cannot silently change generated pixels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

try:  # Pillow is the lightweight default image codec.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Image = None  # type: ignore[assignment]

try:  # OpenCV is an optional acceleration/backend choice.
    import cv2 as _cv2
except ImportError:  # pragma: no cover - environment dependent
    _cv2 = None

IOBackend = Literal["auto", "pillow", "opencv"]
BlurBackend = Literal["portable", "opencv"]


class ImageDependencyError(RuntimeError):
    """Raised when an explicitly selected image backend is unavailable."""


def has_opencv() -> bool:
    return _cv2 is not None


def _resolve_io_backend(backend: IOBackend) -> Literal["pillow", "opencv"]:
    if backend not in {"auto", "pillow", "opencv"}:
        raise ValueError(f"unsupported image backend: {backend!r}")
    if backend in {"auto", "pillow"} and Image is not None:
        return "pillow"
    if backend in {"auto", "opencv"} and _cv2 is not None:
        return "opencv"
    if backend == "opencv":
        raise ImageDependencyError(
            "OpenCV backend requested, but 'cv2' is not installed. Install "
            "opencv-python or use --image-backend pillow."
        )
    if backend == "pillow":
        raise ImageDependencyError(
            "Pillow backend requested, but 'PIL' is not installed. Install pillow."
        )
    raise ImageDependencyError("No image codec is available; install pillow or opencv-python.")


def read_rgb(path: Path, *, backend: IOBackend = "auto") -> np.ndarray:
    """Read an image as a contiguous ``uint8`` RGB array."""

    path = Path(path)
    selected = _resolve_io_backend(backend)
    if selected == "pillow":
        assert Image is not None
        try:
            with Image.open(path) as image:
                image.load()
                return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        except Exception as exc:
            raise ValueError(f"cannot decode image {path}: {exc}") from exc

    assert _cv2 is not None
    image = _cv2.imread(str(path), _cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image {path}")
    return np.ascontiguousarray(_cv2.cvtColor(image, _cv2.COLOR_BGR2RGB))


def image_size(path: Path, *, backend: IOBackend = "auto") -> tuple[int, int]:
    """Return ``(width, height)`` while validating that the image is readable."""

    path = Path(path)
    selected = _resolve_io_backend(backend)
    if selected == "pillow":
        assert Image is not None
        try:
            with Image.open(path) as image:
                image.verify()
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"cannot decode image {path}: {exc}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"image has invalid dimensions {width}x{height}: {path}")
        return int(width), int(height)

    array = read_rgb(path, backend="opencv")
    return int(array.shape[1]), int(array.shape[0])


def write_rgb(path: Path, array: np.ndarray, *, backend: IOBackend = "auto") -> None:
    """Write an RGB image with deterministic encoder options where supported."""

    path = Path(path)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB array, got {array.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = _resolve_io_backend(backend)
    suffix = path.suffix.lower()

    if selected == "pillow":
        assert Image is not None
        image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
        options: dict[str, object] = {}
        if suffix in {".jpg", ".jpeg"}:
            options.update(quality=95, subsampling=0, optimize=False, progressive=False)
        elif suffix == ".png":
            options.update(compress_level=6, optimize=False)
        try:
            image.save(path, **options)
        except Exception as exc:
            raise ValueError(f"cannot encode image {path}: {exc}") from exc
        return

    assert _cv2 is not None
    bgr = _cv2.cvtColor(np.ascontiguousarray(array), _cv2.COLOR_RGB2BGR)
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [_cv2.IMWRITE_JPEG_QUALITY, 95]
    elif suffix == ".png":
        params = [_cv2.IMWRITE_PNG_COMPRESSION, 6]
    if not _cv2.imwrite(str(path), bgr, params):
        raise ValueError(f"cannot encode image {path}")


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return points


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """Create a normalized odd square line kernel for deterministic motion blur."""

    if length < 1 or length % 2 == 0:
        raise ValueError("motion blur length must be a positive odd integer")
    center = length // 2
    radius = length // 2
    angle = math.radians(float(angle_degrees))
    dx = int(round(math.cos(angle) * radius))
    dy = int(round(math.sin(angle) * radius))
    points = _bresenham_line(center - dx, center - dy, center + dx, center + dy)
    kernel = np.zeros((length, length), dtype=np.float32)
    for x, y in points:
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total == 0.0:  # Defensive; length=1 still has its center point.
        kernel[center, center] = 1.0
        total = 1.0
    return kernel / total


def apply_motion_blur(
    array: np.ndarray,
    *,
    length: int,
    angle_degrees: float,
    backend: BlurBackend = "portable",
) -> np.ndarray:
    """Blur a float or uint8 RGB array, preserving its dtype and shape."""

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 array, got shape {array.shape}")
    kernel = motion_kernel(length, angle_degrees)
    if backend == "opencv":
        if _cv2 is None:
            raise ImageDependencyError(
                "OpenCV blur backend requested, but 'cv2' is not installed. "
                "Use --blur-backend portable or install opencv-python."
            )
        return _cv2.filter2D(array, ddepth=-1, kernel=kernel, borderType=_cv2.BORDER_REFLECT_101)
    if backend != "portable":
        raise ValueError(f"unsupported blur backend: {backend!r}")

    radius = length // 2
    padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    output = np.zeros_like(array, dtype=np.float32)
    height, width = array.shape[:2]
    for row, col in zip(*np.nonzero(kernel)):
        weight = float(kernel[row, col])
        output += padded[row : row + height, col : col + width].astype(np.float32) * weight
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return np.clip(np.rint(output), info.min, info.max).astype(array.dtype)
    return output.astype(array.dtype, copy=False)
