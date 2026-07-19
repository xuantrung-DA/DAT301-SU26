"""Dataset preparation utilities for the LADD-UAV experiments.

The package intentionally keeps OpenCV optional.  Pillow and NumPy are enough
for the portable, deterministic data pipeline; callers may explicitly request
the OpenCV blur backend when it is installed.
"""

from .audit import AuditIssue, AuditReport, audit_yolo_dataset
from .auair import convert_auair_dataset
from .exdark import convert_exdark_dataset
from .lol import prepare_lol_v1
from .lowlight import (
    LLMIX_WEIGHTS,
    PROTOCOL_SEEDS,
    LEVEL_SPECS,
    LowLightParameters,
    synthesize_dataset,
    synthesize_image,
)
from .visdrone import (
    VISDRONE_CLASS_NAMES,
    VISDRONE_TO_YOLO,
    convert_visdrone_dataset,
    discover_visdrone_splits,
)
from .uavdt import convert_uavdt_dataset

__all__ = [
    "AuditIssue",
    "AuditReport",
    "LEVEL_SPECS",
    "LLMIX_WEIGHTS",
    "LowLightParameters",
    "PROTOCOL_SEEDS",
    "VISDRONE_CLASS_NAMES",
    "VISDRONE_TO_YOLO",
    "audit_yolo_dataset",
    "convert_auair_dataset",
    "convert_exdark_dataset",
    "convert_uavdt_dataset",
    "convert_visdrone_dataset",
    "discover_visdrone_splits",
    "synthesize_dataset",
    "synthesize_image",
    "prepare_lol_v1",
]
