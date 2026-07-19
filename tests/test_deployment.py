from pathlib import Path

from src.deployment.build_tensorrt import build_command
from src.deployment.build_tensorrt_python import parse_model


def test_tensorrt_fp16_command_is_reproducible() -> None:
    command = build_command(
        "trtexec",
        Path("gate.onnx"),
        Path("gate.engine"),
        precision="fp16",
        workspace_mib=1024,
        timing_cache=Path("timing.cache"),
    )
    assert command[0] == "trtexec"
    assert "--fp16" in command
    assert "--skipInference" in command
    assert "--memPoolSize=workspace:1024MiB" in command


def test_python_builder_model_argument() -> None:
    name, path = parse_model("lowlight=exports/model.onnx")
    assert name == "lowlight"
    assert path == Path("exports/model.onnx")
