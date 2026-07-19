"""Benchmark a fixed-shape TensorRT engine using zero-copy PyTorch CUDA buffers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TRT_TO_TORCH = {
    "DataType.FLOAT": torch.float32,
    "DataType.HALF": torch.float16,
    "DataType.INT32": torch.int32,
    "DataType.INT64": torch.int64,
    "DataType.BOOL": torch.bool,
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(engine_path: Path, warmup: int, iterations: int) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {engine_path}")
    context = engine.create_execution_context()
    buffers: dict[str, torch.Tensor] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = TRT_TO_TORCH[str(engine.get_tensor_dtype(name))]
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
            tensor.uniform_(0, 1)
        else:
            outputs.append(name)

    # A dedicated non-default stream avoids TensorRT's implicit synchronization.
    stream = torch.cuda.Stream()
    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed during warmup")
    stream.synchronize()

    timings: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        end.record(stream)
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    return {
        "engine": str(engine_path.resolve()),
        "tensorrt": trt.__version__,
        "device": torch.cuda.get_device_name(),
        "warmup": warmup,
        "iterations": iterations,
        "input_tensors": {name: list(buffers[name].shape) for name in inputs},
        "output_tensors": {name: list(buffers[name].shape) for name in outputs},
        "mean_ms": float(np.mean(timings)),
        "p50_ms": percentile(timings, 50),
        "p95_ms": percentile(timings, 95),
        "fps": 1000.0 / float(np.mean(timings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    result = benchmark(args.engine, args.warmup, args.iterations)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
