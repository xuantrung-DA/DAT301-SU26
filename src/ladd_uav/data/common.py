"""Small dependency-free helpers shared by data preparation modules."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def iter_images(directory: Path, *, recursive: bool = False) -> Iterator[Path]:
    """Yield supported image files in stable, case-insensitive path order."""

    directory = Path(directory)
    candidates: Iterable[Path] = directory.rglob("*") if recursive else directory.glob("*")
    images = (p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    yield from sorted(images, key=lambda p: p.as_posix().casefold())


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-independent 63-bit seed from a base seed and key parts.

    Python's built-in ``hash`` is deliberately salted per process and therefore
    must not be used for experiment manifests.
    """

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    message = "\0".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(message).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def canonical_json(data: Any) -> str:
    """Serialize a manifest/report row in a stable UTF-8 friendly form."""

    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def transfer_file(
    source: Path,
    destination: Path,
    *,
    mode: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
) -> None:
    """Copy/link one file with consistent collision handling."""

    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError("mode must be one of: copy, hardlink, symlink")
