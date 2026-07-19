"""Build branchable TensorRT engines from exported LADD-UAV ONNX graphs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


GRAPHS = {
    "gate": ("gate.onnx", "gate.engine"),
    "light": ("enhancer_light.onnx", "enhancer_light.engine"),
    "full": ("enhancer_full.onnx", "enhancer_full.engine"),
    "detector": ("yolo11n.onnx", "yolo11n.engine"),
}


def build_command(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str,
    workspace_mib: int,
    timing_cache: Path,
) -> list[str]:
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}MiB",
        f"--timingCacheFile={timing_cache}",
        "--builderOptimizationLevel=5",
        "--skipInference",
    ]
    if precision == "fp16":
        command.append("--fp16")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=Path("exports/ladd_uav"))
    parser.add_argument("--output", type=Path, default=Path("exports/ladd_uav/tensorrt_fp16"))
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    executable = shutil.which(args.trtexec)
    if executable is None:
        raise FileNotFoundError(
            f"TensorRT trtexec executable was not found: {args.trtexec}. "
            "Install TensorRT and add trtexec to PATH, or pass --trtexec."
        )
    missing = [args.onnx_dir / source for source, _target in GRAPHS.values() if not (args.onnx_dir / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing ONNX graph(s): " + ", ".join(str(path) for path in missing))

    args.output.mkdir(parents=True, exist_ok=True)
    timing_cache = args.output / "timing.cache"
    records = []
    for branch, (source_name, target_name) in GRAPHS.items():
        source = args.onnx_dir / source_name
        target = args.output / target_name
        command = build_command(
            executable,
            source.resolve(),
            target.resolve(),
            precision=args.precision,
            workspace_mib=args.workspace_mib,
            timing_cache=timing_cache.resolve(),
        )
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        log_path = args.output / f"build_{branch}.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not target.is_file():
            raise RuntimeError(f"TensorRT build failed for {branch}; inspect {log_path}")
        records.append({
            "branch": branch,
            "onnx": str(source.resolve()),
            "engine": str(target.resolve()),
            "engine_bytes": target.stat().st_size,
            "log": str(log_path.resolve()),
        })

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "precision": args.precision,
        "workspace_mib": args.workspace_mib,
        "trtexec": executable,
        "branching": "Run gate first; bypass skips enhancement, light uses enhancer_light, full uses enhancer_full; then run yolo11n.",
        "engines": records,
    }
    manifest_path = args.output / "engine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
