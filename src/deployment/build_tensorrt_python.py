"""Build fixed-shape TensorRT engines with the official Python bindings."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use NAME=PATH.onnx")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("model name must not be empty")
    return name.strip(), Path(raw_path)


def build_engine(onnx_path: Path, engine_path: Path, workspace_mib: int, fp16: bool) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 11 removed weak typing and the legacy FP16 builder flag.
    # Precision is carried by the ONNX tensors, so an FP16 build requires an
    # FP16 ONNX export and a strongly typed network.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_path}: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * 1024**2)
    input_dtypes = [network.get_input(i).dtype for i in range(network.num_inputs)]
    if fp16 and not all(dtype == trt.float16 for dtype in input_dtypes):
        raise ValueError(
            "--fp16 requires an FP16 ONNX export; got input dtypes "
            + ", ".join(str(dtype) for dtype in input_dtypes)
        )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build returned no bytes for {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return {
        "onnx": str(onnx_path.resolve()),
        "engine": str(engine_path.resolve()),
        "engine_bytes": engine_path.stat().st_size,
        "inputs": [
            {"name": network.get_input(i).name, "shape": list(network.get_input(i).shape), "dtype": str(network.get_input(i).dtype)}
            for i in range(network.num_inputs)
        ],
        "outputs": [
            {"name": network.get_output(i).name, "shape": list(network.get_output(i).shape), "dtype": str(network.get_output(i).dtype)}
            for i in range(network.num_outputs)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True, help="Repeat NAME=PATH.onnx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, onnx_path in args.model:
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        records.append({"name": name, **build_engine(onnx_path, args.output / f"{name}.engine", args.workspace_mib, args.fp16)})
    import tensorrt as trt

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorrt": trt.__version__,
        "precision": "fp16" if args.fp16 else "fp32",
        "workspace_mib": args.workspace_mib,
        "engines": records,
    }
    (args.output / "engine_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
