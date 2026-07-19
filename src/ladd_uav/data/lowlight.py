"""Deterministic LL1/LL2/LL3/LLMix synthesis for paired UAV detection data."""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .common import canonical_json, iter_images, sha256_file, stable_seed
from .imaging import BlurBackend, IOBackend, apply_motion_blur, read_rgb, write_rgb

PROTOCOL_SEEDS = (3407, 2025, 301)


@dataclass(frozen=True)
class LowLightLevelSpec:
    alpha: tuple[float, float]
    gamma: tuple[float, float]
    read_sigma: tuple[float, float]
    shot_photons: tuple[float, float]
    color_jitter: float
    blur_probability: float
    blur_lengths: tuple[int, ...]


# Alpha/gamma/read-noise/color/blur values are taken directly from the project
# protocol.  ``shot_photons`` makes "mild / Poisson / stronger" operational:
# fewer effective photons produce stronger signal-dependent Poisson noise.
LEVEL_SPECS: Mapping[str, LowLightLevelSpec] = {
    "LL1": LowLightLevelSpec(
        alpha=(0.55, 0.75),
        gamma=(1.4, 2.0),
        read_sigma=(0.005, 0.015),
        shot_photons=(1000.0, 2000.0),
        color_jitter=0.05,
        blur_probability=0.10,
        blur_lengths=(3,),
    ),
    "LL2": LowLightLevelSpec(
        alpha=(0.30, 0.55),
        gamma=(2.0, 3.0),
        read_sigma=(0.01, 0.03),
        shot_photons=(400.0, 1000.0),
        color_jitter=0.10,
        blur_probability=0.20,
        blur_lengths=(3, 5),
    ),
    "LL3": LowLightLevelSpec(
        alpha=(0.10, 0.30),
        gamma=(3.0, 5.0),
        read_sigma=(0.02, 0.05),
        shot_photons=(100.0, 400.0),
        color_jitter=0.15,
        blur_probability=0.30,
        blur_lengths=(3, 5, 7),
    ),
}

# LLMix is 20% byte-identical clean data.  Its remaining 80% is distributed
# LL1:LL2:LL3 = 40:40:20, yielding unconditional weights 32:32:16.
LLMIX_WEIGHTS: Mapping[str, float] = {
    "CLEAN": 0.20,
    "LL1": 0.32,
    "LL2": 0.32,
    "LL3": 0.16,
}


@dataclass(frozen=True)
class LowLightParameters:
    level: str
    seed: int
    alpha: float
    gamma: float
    read_sigma: float
    shot_photons: float | None
    color_gains: tuple[float, float, float]
    blur_probability: float
    blur_applied: bool
    blur_length: int | None
    blur_angle_degrees: float | None

    @property
    def is_clean(self) -> bool:
        return self.level == "CLEAN"

    def manifest_fields(self, *, blur_backend: str) -> dict[str, object]:
        """Return the protocol-required per-image parameters."""

        return {
            "alpha": self.alpha,
            "blur": {
                "angle_degrees": self.blur_angle_degrees,
                "applied": self.blur_applied,
                "backend": blur_backend,
                "length": self.blur_length,
                "probability": self.blur_probability,
            },
            "color": {"channel_order": "RGB", "gains": list(self.color_gains)},
            "gamma": self.gamma,
            "noise": {
                "model": "none" if self.is_clean else "poisson_gaussian",
                "read_sigma": self.read_sigma,
                "shot_photons": self.shot_photons,
            },
            "seed": self.seed,
        }


@dataclass
class SynthesisStats:
    split: str
    variant: str
    images: int = 0
    clean_images: int = 0
    ll1_images: int = 0
    ll2_images: int = 0
    ll3_images: int = 0

    def record(self, level: str) -> None:
        self.images += 1
        if level == "CLEAN":
            self.clean_images += 1
        elif level == "LL1":
            self.ll1_images += 1
        elif level == "LL2":
            self.ll2_images += 1
        elif level == "LL3":
            self.ll3_images += 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_parameters(level: str, seed: int) -> LowLightParameters:
    """Sample all parameters for one image solely from its recorded seed."""

    level = level.upper()
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if level == "CLEAN":
        return LowLightParameters(
            level="CLEAN",
            seed=seed,
            alpha=1.0,
            gamma=1.0,
            read_sigma=0.0,
            shot_photons=None,
            color_gains=(1.0, 1.0, 1.0),
            blur_probability=0.0,
            blur_applied=False,
            blur_length=None,
            blur_angle_degrees=None,
        )
    if level not in LEVEL_SPECS:
        raise ValueError(f"unknown low-light level {level!r}; expected {tuple(LEVEL_SPECS)}")
    spec = LEVEL_SPECS[level]
    rng = np.random.default_rng(seed)
    alpha = float(rng.uniform(*spec.alpha))
    gamma = float(rng.uniform(*spec.gamma))
    read_sigma = float(rng.uniform(*spec.read_sigma))
    shot_photons = float(rng.uniform(*spec.shot_photons))
    gains = tuple(float(value) for value in rng.uniform(
        1.0 - spec.color_jitter, 1.0 + spec.color_jitter, size=3
    ))
    blur_applied = bool(rng.random() < spec.blur_probability)
    blur_length = int(rng.choice(spec.blur_lengths)) if blur_applied else None
    blur_angle = float(rng.uniform(-180.0, 180.0)) if blur_applied else None
    return LowLightParameters(
        level=level,
        seed=seed,
        alpha=alpha,
        gamma=gamma,
        read_sigma=read_sigma,
        shot_photons=shot_photons,
        color_gains=gains,  # type: ignore[arg-type]
        blur_probability=spec.blur_probability,
        blur_applied=blur_applied,
        blur_length=blur_length,
        blur_angle_degrees=blur_angle,
    )


def synthesize_image(
    image: np.ndarray,
    parameters: LowLightParameters,
    *,
    blur_backend: BlurBackend = "portable",
) -> np.ndarray:
    """Apply the documented Poisson-Gaussian low-light image formation model."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 RGB image, got {image.dtype} {image.shape}")
    if parameters.is_clean:
        return image.copy()
    if parameters.shot_photons is None or parameters.shot_photons <= 0:
        raise ValueError("low-light parameters require positive shot_photons")

    rng = np.random.default_rng(parameters.seed)
    # Consume the draws used by sample_parameters so noise starts at a stable,
    # non-overlapping point even when parameters were serialized and reloaded.
    spec = LEVEL_SPECS[parameters.level]
    rng.uniform(*spec.alpha)
    rng.uniform(*spec.gamma)
    rng.uniform(*spec.read_sigma)
    rng.uniform(*spec.shot_photons)
    rng.uniform(1.0 - spec.color_jitter, 1.0 + spec.color_jitter, size=3)
    blur_draw = rng.random()
    if blur_draw < spec.blur_probability:
        rng.choice(spec.blur_lengths)
        rng.uniform(-180.0, 180.0)

    normalized = image.astype(np.float32) / np.float32(255.0)
    gains = np.asarray(parameters.color_gains, dtype=np.float32).reshape(1, 1, 3)
    dark = parameters.alpha * np.power(normalized, parameters.gamma) * gains
    dark = np.clip(dark, 0.0, 1.0)
    photons = float(parameters.shot_photons)
    shot = rng.poisson(dark * photons).astype(np.float32) / np.float32(photons)
    read = rng.normal(0.0, parameters.read_sigma, size=dark.shape).astype(np.float32)
    output = np.clip(shot + read, 0.0, 1.0).astype(np.float32)
    if parameters.blur_applied:
        assert parameters.blur_length is not None and parameters.blur_angle_degrees is not None
        output = apply_motion_blur(
            output,
            length=parameters.blur_length,
            angle_degrees=parameters.blur_angle_degrees,
            backend=blur_backend,
        )
    return np.clip(np.rint(output * 255.0), 0.0, 255.0).astype(np.uint8)


def _largest_remainder_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("mixture weights must sum to one")
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order_index = {name: index for index, name in enumerate(weights)}
    by_remainder = sorted(
        weights,
        key=lambda name: (-(raw[name] - counts[name]), order_index[name]),
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1
    return counts


def assign_llmix_levels(keys: Sequence[str], base_seed: int) -> dict[str, str]:
    """Assign exact, deterministic LLMix counts using stable image hashes."""

    if len(set(keys)) != len(keys):
        raise ValueError("LLMix assignment keys must be unique")
    ordered = sorted(keys, key=lambda key: (stable_seed(base_seed, "llmix", key), key))
    counts = _largest_remainder_counts(len(ordered), LLMIX_WEIGHTS)
    assignments: dict[str, str] = {}
    offset = 0
    for level in LLMIX_WEIGHTS:
        for key in ordered[offset : offset + counts[level]]:
            assignments[key] = level
        offset += counts[level]
    return assignments


def _discover_splits(dataset_root: Path, requested: Iterable[str] | None) -> dict[str, tuple[Path, Path]]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            f"expected YOLO dataset directories {images_root} and {labels_root}"
        )
    if requested is not None:
        names = list(dict.fromkeys(requested))
    else:
        names = sorted(
            [path.name for path in images_root.iterdir() if path.is_dir()], key=str.casefold
        )
        if not names and list(iter_images(images_root)):
            names = ["all"]
    if not names:
        raise FileNotFoundError(f"no image splits found below {images_root}")

    splits: dict[str, tuple[Path, Path]] = {}
    for name in names:
        if name == "all" and not (images_root / name).is_dir():
            image_dir, label_dir = images_root, labels_root
        else:
            image_dir, label_dir = images_root / name, labels_root / name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image split does not exist: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"label split does not exist: {label_dir}")
        splits[name] = (image_dir, label_dir)
    return splits


def _prepare_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"destination already exists (use overwrite=True): {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_variant_yaml(source_root: Path, variant_root: Path, splits: Iterable[str]) -> None:
    source_yaml = source_root / "dataset.yaml"
    destination_yaml = variant_root / "dataset.yaml"
    if source_yaml.is_file():
        shutil.copyfile(source_yaml, destination_yaml)
        return
    names = list(splits)
    lines = ["path: ."]
    if "train" in names:
        lines.append("train: images/train")
    if "val" in names:
        lines.append("val: images/val")
    tests = [name for name in names if name.startswith("test")]
    if tests:
        lines.append(f"test: images/{tests[0]}")
    destination_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def synthesize_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    splits: Iterable[str] | None = None,
    variants: Iterable[str] = ("LL1", "LL2", "LL3", "LLMix"),
    base_seed: int = PROTOCOL_SEEDS[0],
    overwrite: bool = False,
    image_backend: IOBackend = "auto",
    blur_backend: BlurBackend = "portable",
) -> dict[str, dict[str, SynthesisStats]]:
    """Generate paired variants while retaining every source split and label.

    Outputs use ``<output>/<variant>/{images,labels}/<split>`` and one canonical
    JSONL manifest per variant/split.  Manifest paths are relative, and no clock
    timestamps are stored, so rerunning with the same inputs and seed produces
    byte-identical images and manifests.
    """

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    split_dirs = _discover_splits(dataset_root, splits)
    normalized_variants = [
        "LLMix" if value.upper() == "LLMIX" else value.upper() for value in variants
    ]
    normalized_variants = list(dict.fromkeys(normalized_variants))
    allowed = {*LEVEL_SPECS, "LLMix"}
    invalid = [name for name in normalized_variants if name not in allowed]
    if invalid:
        raise ValueError(f"unknown variants {invalid}; expected {sorted(allowed)}")
    if not normalized_variants:
        raise ValueError("at least one low-light variant must be requested")

    output_root.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, dict[str, SynthesisStats]] = {}
    for variant in normalized_variants:
        variant_root = output_root / variant
        (variant_root / "manifests").mkdir(parents=True, exist_ok=True)
        all_stats[variant] = {}
        for split, (image_dir, label_dir) in split_dirs.items():
            images = list(iter_images(image_dir, recursive=True))
            if not images:
                raise FileNotFoundError(f"no images found in split: {image_dir}")
            keys = [image.relative_to(image_dir).as_posix() for image in images]
            assignments = (
                assign_llmix_levels([f"{split}/{key}" for key in keys], base_seed)
                if variant == "LLMix"
                else {}
            )
            stats = SynthesisStats(split=split, variant=variant)
            manifest_rows: list[str] = []
            for source_image, relative_key in zip(images, keys):
                relative_path = Path(relative_key)
                source_label = label_dir / relative_path.with_suffix(".txt")
                if not source_label.is_file():
                    raise FileNotFoundError(f"missing YOLO label for {source_image}: {source_label}")
                selected_level = (
                    assignments[f"{split}/{relative_key}"] if variant == "LLMix" else variant
                )
                image_seed = stable_seed(base_seed, variant, split, relative_key)
                parameters = sample_parameters(selected_level, image_seed)
                destination_image = variant_root / "images" / split / relative_path
                destination_label = variant_root / "labels" / split / relative_path.with_suffix(".txt")
                _prepare_destination(destination_image, overwrite=overwrite)
                _prepare_destination(destination_label, overwrite=overwrite)
                if parameters.is_clean:
                    shutil.copyfile(source_image, destination_image)
                else:
                    image = read_rgb(source_image, backend=image_backend)
                    generated = synthesize_image(image, parameters, blur_backend=blur_backend)
                    write_rgb(destination_image, generated, backend=image_backend)
                shutil.copyfile(source_label, destination_label)
                stats.record(selected_level)
                row: dict[str, object] = {
                    "base_seed": base_seed,
                    "is_clean": parameters.is_clean,
                    "label_sha256": sha256_file(destination_label),
                    "output_image": destination_image.relative_to(output_root).as_posix(),
                    "output_sha256": sha256_file(destination_image),
                    "schema_version": 1,
                    "selected_level": selected_level,
                    "source_image": source_image.relative_to(dataset_root).as_posix(),
                    "source_sha256": sha256_file(source_image),
                    "split": split,
                    "variant": variant,
                }
                row.update(parameters.manifest_fields(blur_backend=blur_backend))
                manifest_rows.append(canonical_json(row))

            manifest_path = variant_root / "manifests" / f"{split}.jsonl"
            if manifest_path.exists() and not overwrite:
                raise FileExistsError(
                    f"manifest already exists (use overwrite=True): {manifest_path}"
                )
            manifest_path.write_text(
                "\n".join(manifest_rows) + "\n", encoding="utf-8", newline="\n"
            )
            all_stats[variant][split] = stats
        _write_variant_yaml(dataset_root, variant_root, split_dirs)

    summary = {
        "base_seed": base_seed,
        "protocol_seeds": list(PROTOCOL_SEEDS),
        "stats": {
            variant: {split: stats.to_dict() for split, stats in split_stats.items()}
            for variant, split_stats in all_stats.items()
        },
        "variants": normalized_variants,
    }
    summary_path = output_root / "synthesis_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"summary already exists (use overwrite=True): {summary_path}")
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8", newline="\n")
    return all_stats
