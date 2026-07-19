"""Validate and prepare official LOL-v1 paired enhancement splits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .common import canonical_json, iter_images, sha256_file, transfer_file
from .imaging import IOBackend, image_size

LOL_SPLITS = {"train": "our485", "test": "eval15"}


def _by_stem(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for image in iter_images(directory):
        key = image.stem.casefold()
        if key in result:
            raise ValueError(f"duplicate image stem in {directory}: {image.stem}")
        result[key] = image
    return result


def prepare_lol_v1(
    source_root: Path,
    output_root: Path,
    *,
    transfer: Literal["copy", "hardlink", "symlink"] = "copy",
    overwrite: bool = False,
    image_backend: IOBackend = "auto",
) -> dict[str, int]:
    """Prepare ``our485`` and ``eval15`` without changing pair/split identity.

    The output is ``<root>/<train|test>/<low|high>`` plus deterministic JSONL
    pair manifests.  Low/high images must have matching stems and dimensions.
    """

    source_root, output_root = Path(source_root), Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    seen_hashes: dict[str, tuple[str, Path]] = {}
    for split, official_folder in LOL_SPLITS.items():
        split_root = source_root / official_folder
        low_dir, high_dir = split_root / "low", split_root / "high"
        if not low_dir.is_dir() or not high_dir.is_dir():
            raise FileNotFoundError(
                f"LOL-v1 {split} expects {low_dir} and {high_dir}"
            )
        low, high = _by_stem(low_dir), _by_stem(high_dir)
        if low.keys() != high.keys():
            missing_high = sorted(low.keys() - high.keys())
            missing_low = sorted(high.keys() - low.keys())
            raise ValueError(
                f"unpaired LOL-v1 files in {split}: missing_high={missing_high[:10]}, "
                f"missing_low={missing_low[:10]}"
            )
        if not low:
            raise FileNotFoundError(f"no LOL-v1 image pairs found in {split_root}")

        rows: list[str] = []
        for key in sorted(low):
            low_image, high_image = low[key], high[key]
            low_size = image_size(low_image, backend=image_backend)
            high_size = image_size(high_image, backend=image_backend)
            if low_size != high_size:
                raise ValueError(
                    f"LOL-v1 pair has mismatched dimensions: {low_image}={low_size}, "
                    f"{high_image}={high_size}"
                )
            low_hash, high_hash = sha256_file(low_image), sha256_file(high_image)
            for digest, source in ((low_hash, low_image), (high_hash, high_image)):
                previous = seen_hashes.get(digest)
                if previous is not None and previous[0] != split:
                    raise ValueError(
                        f"LOL-v1 cross-split duplicate: {source} and {previous[1]}"
                    )
                seen_hashes[digest] = (split, source)
            destination_low = output_root / split / "low" / low_image.name
            destination_high = output_root / split / "high" / high_image.name
            transfer_file(low_image, destination_low, mode=transfer, overwrite=overwrite)
            transfer_file(high_image, destination_high, mode=transfer, overwrite=overwrite)
            rows.append(
                canonical_json(
                    {
                        "height": low_size[1],
                        "high": destination_high.relative_to(output_root).as_posix(),
                        "high_sha256": high_hash,
                        "low": destination_low.relative_to(output_root).as_posix(),
                        "low_sha256": low_hash,
                        "pair_id": low_image.stem,
                        "schema_version": 1,
                        "split": split,
                        "width": low_size[0],
                    }
                )
            )
        manifest = output_root / f"{split}_pairs.jsonl"
        if manifest.exists() and not overwrite:
            raise FileExistsError(manifest)
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        counters[split] = len(rows)

    summary = output_root / "preparation_summary.json"
    if summary.exists() and not overwrite:
        raise FileExistsError(summary)
    summary.write_text(
        canonical_json({"dataset": "LOL-v1", "pairs": counters, "transfer": transfer}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return counters
