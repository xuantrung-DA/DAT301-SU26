"""Stage A detector adaptation for B0/B1/B2 and the three registered seeds."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from training.common import seed_everything, write_json


DATASETS = {
    "B0": Path("datasets/VisDrone/visdrone_clean.yaml"),
    "B1": Path("datasets/VisDrone-LL/LL2/visdrone_ll2.yaml"),
    "B2": Path("datasets/VisDrone-LL/LLMix/visdrone_llmix.yaml"),
}


def _read_epoch_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                key = key.strip()
                value = value.strip() if isinstance(value, str) else value
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=sorted(DATASETS), required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--model-config", type=Path, help="Optional custom model YAML, e.g. slim P2 ablation")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, choices=[3407, 2025, 301], default=3407)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=1.0, help="Training-data fraction; use <1 only for smoke tests")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--project", type=Path, default=Path("runs/detector_adaptation"))
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    seed_everything(args.seed)
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics before Stage A") from exc

    data = args.data or DATASETS[args.baseline]
    name = f"{args.baseline.lower()}_seed_{args.seed}"
    detector = YOLO(str(args.model_config)).load(args.weights) if args.model_config else YOLO(args.weights)
    result = detector.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, project=str(args.project), name=name,
        device=0 if torch.cuda.is_available() else "cpu", plots=True, save=True,
        workers=args.workers, fraction=args.fraction, patience=args.patience,
    )
    output = Path(result.save_dir)
    hardware = {
        "baseline": args.baseline,
        "seed": args.seed,
        "data": str(data),
        "weights": args.weights,
        "model_config": str(args.model_config) if args.model_config else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_size": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "fraction": args.fraction,
        "epochs": args.epochs,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run_manifest.json", hardware)
    epoch_history = _read_epoch_history(output / "results.csv")
    write_json(output / "epoch_history.json", epoch_history)
    metric_key = next(
        (key for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)") if epoch_history and key in epoch_history[0]),
        None,
    )
    best_index = (
        max(range(len(epoch_history)), key=lambda index: epoch_history[index].get(metric_key, float("-inf")))
        if epoch_history and metric_key
        else max(len(epoch_history) - 1, 0)
    )
    summary = {
        **hardware,
        "save_dir": str(output),
        "epochs_completed": len(epoch_history),
        "best_epoch_index": best_index,
        "best_metrics": epoch_history[best_index] if epoch_history else {},
        "last_metrics": epoch_history[-1] if epoch_history else {},
    }
    write_json(output / "results.json", summary)
    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint = output / "weights" / checkpoint_name
        if checkpoint.is_file():
            write_json(
                checkpoint.with_suffix(".json"),
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "baseline": args.baseline,
                    "seed": args.seed,
                    "metrics": summary["best_metrics"] if checkpoint_name == "best.pt" else summary["last_metrics"],
                    "run_results": str(output / "results.json"),
                },
            )
    print(json.dumps({"save_dir": str(output), **hardware}, indent=2))


if __name__ == "__main__":
    main()
