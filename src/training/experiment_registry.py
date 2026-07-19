"""Build a machine-readable index of datasets, runs, results, and checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


JSON_GROUPS = {
    "data": (
        "datasets/**/audit.json",
        "datasets/**/conversion_summary.json",
        "datasets/**/degradation_config.json",
        "datasets/**/overlap_mapping.json",
    ),
    "configs": ("configs/**/*.json",),
    "results": ("runs/**/*result*.json", "runs/**/*summary*.json", "runs/**/*history*.json"),
    "checkpoints": ("runs/**/*.pt", "runs/**/*.onnx", "runs/**/*.engine"),
    "checkpoint_metadata": ("runs/**/weights/*.json", "runs/**/*stage_*.json", "runs/**/*joint*.json"),
    "deployment": ("runs/**/*parity*.json", "runs/**/*latency*.json", "runs/**/*profile*.json"),
}


def _record(root: Path, path: Path) -> dict:
    stat = path.stat()
    record = {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                record["keys"] = sorted(payload)
                for key in ("status", "stage", "seed", "epoch", "architecture", "configuration"):
                    if key in payload:
                        record[key] = payload[key]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            record["json_error"] = str(error)
    return record


def build_registry(root: Path) -> dict:
    root = root.resolve()
    groups: dict[str, list[dict]] = {}
    for name, patterns in JSON_GROUPS.items():
        paths = {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
        groups[name] = [_record(root, path) for path in sorted(paths)]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "groups": groups,
        "counts": {name: len(records) for name, records in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_registry.json"))
    args = parser.parse_args()
    registry = build_registry(args.root)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": registry["counts"]}, indent=2))


if __name__ == "__main__":
    main()
