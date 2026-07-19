#!/usr/bin/env python3
"""Remap converted cross-domain labels to the VisDrone detector taxonomy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-output", type=Path, help="Optional mirrored hard-link image tree")
    args = parser.parse_args()
    data = yaml.safe_load((args.dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    names = {int(key): value for key, value in data["names"].items()}
    mapping_doc = json.loads((args.dataset_root / "overlap_mapping.json").read_text(encoding="utf-8"))
    overlap = mapping_doc["visdrone_overlap"]
    source_to_target: dict[int, int] = {}
    for source_id, source_name in names.items():
        record = overlap.get(str(source_id), overlap.get(source_name))
        if record is not None:
            source_to_target[source_id] = int(record["visdrone_id"])
    source = args.dataset_root / "labels" / args.split
    source_images = args.dataset_root / "images" / args.split
    args.output.mkdir(parents=True, exist_ok=True)
    objects_in = objects_out = 0
    labels = sorted(source.rglob("*.txt"))
    for label in labels:
        output_rows = []
        for row in label.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if not fields:
                continue
            objects_in += 1
            source_id = int(fields[0])
            if source_id in source_to_target:
                output_rows.append(" ".join([str(source_to_target[source_id]), *fields[1:]]))
                objects_out += 1
        target = args.output / label.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
    if args.images_output:
        for image in sorted(path for path in source_images.rglob("*") if path.is_file()):
            target = args.images_output / image.relative_to(source_images)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(image, target)
    print(json.dumps({"files": len(labels), "objects_in": objects_in, "objects_overlap": objects_out, "mapping": source_to_target}, indent=2))


if __name__ == "__main__":
    main()
