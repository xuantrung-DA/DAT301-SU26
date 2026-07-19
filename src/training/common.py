from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "shape": list(value.shape),
        }
    return str(value)


def _state_dict_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    tensors = [item for item in value.values() if isinstance(item, torch.Tensor)]
    if len(tensors) != len(value):
        return None
    return {
        "keys": len(tensors),
        "parameters_and_buffers": sum(tensor.numel() for tensor in tensors),
    }


def save_checkpoint(path: Path, **state):
    """Save weights plus a human-readable JSON sidecar for every checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    metadata: dict[str, Any] = {
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_dicts: dict[str, Any] = {}
    for key, value in state.items():
        summary = _state_dict_summary(value)
        if summary is not None:
            state_dicts[key] = summary
        else:
            metadata[key] = _json_safe(value)
    if state_dicts:
        metadata["state_dicts"] = state_dicts
    write_json(path.with_suffix(".json"), metadata)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")
