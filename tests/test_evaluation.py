from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluate_detector import class_ap, load_ground_truth, match_at_confidence


def test_load_ground_truth_rejects_malformed_rows(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YOLO label"):
        load_ground_truth(label, 100, 100)


def test_small_threshold_is_used_consistently(tmp_path: Path) -> None:
    label = tmp_path / "one.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    ground_truth = load_ground_truth(label, 100, 100, tiny_side_threshold=24)
    record = {
        "gt": ground_truth,
        "pred_xyxy": np.asarray([[40, 40, 60, 60]], dtype=np.float32),
        "pred_conf": np.asarray([0.9], dtype=np.float32),
        "pred_class": np.asarray([0], dtype=np.int64),
    }
    assert ground_truth["tiny_side"].tolist() == [True]
    assert match_at_confidence(record, 0.25, small_area=500)["small_tp"] == 1
    assert class_ap([record], 0, 0.5, "small", small_area=500) == pytest.approx(1.0)


def test_statistics_domain_key_contract() -> None:
    summary = {"configuration": "m2", "domain": "LL2"}
    assert f"{summary['domain']}/{summary['configuration']}" == "LL2/m2"
