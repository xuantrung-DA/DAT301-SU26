"""Detection evaluation with AP-small, small recall, FP/image and raw rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from inference import enhance_bgr_with_details, image_to_tensor, letterbox_bgr, load_detector, load_domain_router, load_generator, route_domain_bgr, unletterbox_xyxy
from preprocess.lowlight import clahe_enhance_bgr, gamma_enhance_bgr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    intersection_min = np.maximum(left[:, None, :2], right[None, :, :2])
    intersection_max = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(intersection_max - intersection_min, 0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0, None).prod(axis=1)
    return intersection / np.clip(left_area[:, None] + right_area[None, :] - intersection, 1e-9, None)


def load_ground_truth(
    path: Path,
    width: int,
    height: int,
    *,
    tiny_side_threshold: float = 16.0,
) -> dict[str, np.ndarray]:
    rows = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) < 5:
                raise ValueError(f"Malformed YOLO label at {path}:{line_number}: expected 5 values")
            try:
                values = [float(value) for value in tokens[:5]]
            except ValueError as exc:
                raise ValueError(f"Malformed numeric YOLO label at {path}:{line_number}") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite YOLO label at {path}:{line_number}")
            rows.append(values)
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    if not len(array):
        return {"class": np.empty(0, dtype=np.int64), "xyxy": np.empty((0, 4), dtype=np.float32), "area": np.empty(0), "tiny_side": np.empty(0, dtype=bool)}
    xywh = array[:, 1:5] * np.array([width, height, width, height], dtype=np.float32)
    return {
        "class": array[:, 0].astype(np.int64),
        "xyxy": xywh_to_xyxy(xywh),
        "area": xywh[:, 2] * xywh[:, 3],
        "tiny_side": (xywh[:, 2] < tiny_side_threshold) | (xywh[:, 3] < tiny_side_threshold),
    }


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    points = np.linspace(0.0, 1.0, 101)
    values = [precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in points]
    return float(np.mean(values))


def class_ap(
    records: list[dict],
    class_id: int,
    iou_threshold: float,
    size: str = "all",
    small_area: float = 32.0**2,
) -> float | None:
    active_by_image: dict[int, np.ndarray] = {}
    ignored_by_image: dict[int, np.ndarray] = {}
    total_ground_truth = 0
    predictions = []
    for image_index, record in enumerate(records):
        gt_class = record["gt"]["class"] == class_id
        if size == "small":
            active_mask = gt_class & (record["gt"]["area"] < small_area)
            ignored_mask = gt_class & ~active_mask
        else:
            active_mask, ignored_mask = gt_class, np.zeros_like(gt_class)
        active_by_image[image_index] = record["gt"]["xyxy"][active_mask]
        ignored_by_image[image_index] = record["gt"]["xyxy"][ignored_mask]
        total_ground_truth += int(active_mask.sum())
        pred_mask = record["pred_class"] == class_id
        for box, confidence in zip(record["pred_xyxy"][pred_mask], record["pred_conf"][pred_mask]):
            predictions.append((float(confidence), image_index, box))
    if total_ground_truth == 0:
        return None
    predictions.sort(key=lambda item: item[0], reverse=True)
    matched = {index: np.zeros(len(boxes), dtype=bool) for index, boxes in active_by_image.items()}
    true_positive, false_positive = [], []
    for _confidence, image_index, box in predictions:
        active = active_by_image[image_index]
        active_iou = box_iou(box[None], active)[0]
        if len(active_iou):
            best = int(active_iou.argmax())
            if active_iou[best] >= iou_threshold and not matched[image_index][best]:
                matched[image_index][best] = True
                true_positive.append(1.0); false_positive.append(0.0)
                continue
        ignored_iou = box_iou(box[None], ignored_by_image[image_index])[0]
        if len(ignored_iou) and ignored_iou.max() >= iou_threshold:
            continue
        true_positive.append(0.0); false_positive.append(1.0)
    if not true_positive:
        return 0.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    return interpolated_ap(tp / total_ground_truth, tp / np.clip(tp + fp, 1e-9, None))


def match_at_confidence(
    record: dict,
    confidence: float,
    iou_threshold: float = 0.5,
    small_area: float = 32.0**2,
) -> dict[str, int]:
    keep = record["pred_conf"] >= confidence
    boxes, classes, confidences = record["pred_xyxy"][keep], record["pred_class"][keep], record["pred_conf"][keep]
    matched = np.zeros(len(record["gt"]["xyxy"]), dtype=bool)
    true_positive = 0
    for index in np.argsort(-confidences):
        box, class_id = boxes[index], classes[index]
        candidates = np.where((record["gt"]["class"] == class_id) & ~matched)[0]
        if not len(candidates):
            continue
        ious = box_iou(box[None], record["gt"]["xyxy"][candidates])[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold:
            matched[candidates[best]] = True
            true_positive += 1
    small = record["gt"]["area"] < small_area
    tiny_side = record["gt"]["tiny_side"]
    return {
        "tp": true_positive,
        "fp": int(len(boxes) - true_positive),
        "fn": int(len(matched) - matched.sum()),
        "small_tp": int((matched & small).sum()),
        "small_fn": int(small.sum() - (matched & small).sum()),
        "tiny_side_tp": int((matched & tiny_side).sum()),
        "tiny_side_fn": int(tiny_side.sum() - (matched & tiny_side).sum()),
    }


@torch.inference_mode()
def enhance_image(method: str, model, image: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict]:
    if method == "none":
        return image, {"gate_mode": "none", "residual_l1": 0.0}
    if method == "gamma":
        output = gamma_enhance_bgr(image)
        return output, {"gate_mode": "gamma", "residual_l1": float(np.mean(np.abs(output.astype(float) - image.astype(float))) / 255.0)}
    if method == "clahe":
        output = clahe_enhance_bgr(image)
        return output, {"gate_mode": "clahe", "residual_l1": float(np.mean(np.abs(output.astype(float) - image.astype(float))) / 255.0)}
    if method == "m0":
        tensor = image_to_tensor(image, device)
        residual, _ = model.enhancer(tensor)
        output = torch.clamp(tensor + residual, 0, 1)
        rgb = output[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), {"gate_mode": "always_full_no_dgrdm", "residual_l1": float(residual.abs().mean().cpu())}
    force_mode = "full" if method == "m1" else ("light" if method == "mlight" else None)
    return enhance_bgr_with_details(model, image, device, force_mode=force_mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--bright-yolo", help="Optional clean-domain detector selected by mean luminance")
    parser.add_argument("--real-yolo", help="Optional real-low-light detector selected by learned router")
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--detector-route-threshold", type=float, default=0.30)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--enhancement", choices=["none", "gamma", "clahe", "m0", "m1", "m2", "mlight"], default="none")
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--small-area", type=float, default=32.0**2)
    parser.add_argument("--tiny-side", type=float, default=16.0)
    parser.add_argument("--domain", help="Stable report key such as Clean, LL1, LL2, LL3, ExDark, or AU-AIR")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args()
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images}")
    if not args.labels.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {args.labels}")
    if args.small_area <= 0 or args.tiny_side <= 0:
        raise ValueError("--small-area and --tiny-side must be positive")
    if args.enhancement in {"m0", "m1", "m2", "mlight"} and not args.generator:
        raise ValueError("--generator is required for M0/M1/M2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(args.generator, device) if args.generator else None
    detector = load_detector(args.yolo, args.joint_checkpoint)
    bright_detector = load_detector(args.bright_yolo) if args.bright_yolo else None
    real_detector = load_detector(args.real_yolo) if args.real_yolo else None
    domain_router, router_size = load_domain_router(args.router_checkpoint, device) if args.router_checkpoint else (None, 160)
    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {args.images}")
    records, rows = [], []
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        model_input, letterbox = letterbox_bgr(image, args.imgsz)
        enhanced, diagnostics = enhance_image(args.enhancement, model, model_input, device)
        illumination_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0)
        if domain_router is not None:
            routing = route_domain_bgr(domain_router, image, device, router_size)
            routes = {"clean": bright_detector, "synthetic_lowlight": detector, "real_lowlight": real_detector}
            active_detector = routes[routing["domain_route"]]
            if active_detector is None: raise RuntimeError(f"No detector supplied for {routing['domain_route']}")
            detector_route = routing["domain_route"]
        else:
            active_detector = bright_detector if bright_detector is not None and illumination_mean >= args.detector_route_threshold else detector
            detector_route = "bright" if active_detector is bright_detector else "lowlight"
        result = active_detector.predict(enhanced, imgsz=args.imgsz, conf=0.001, iou=args.nms_iou, max_det=1000, verbose=False)[0]
        pred_xyxy = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        pred_xyxy = unletterbox_xyxy(pred_xyxy, letterbox)
        pred_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty(0)
        pred_class = result.boxes.cls.detach().cpu().numpy().astype(np.int64) if result.boxes is not None else np.empty(0, dtype=np.int64)
        record = {
            "image": str(image_path),
            "gt": load_ground_truth(
                args.labels / image_path.relative_to(args.images).with_suffix(".txt"),
                width,
                height,
                tiny_side_threshold=args.tiny_side,
            ),
            "pred_xyxy": pred_xyxy,
            "pred_conf": pred_conf,
            "pred_class": pred_class,
        }
        matched = match_at_confidence(record, args.confidence, small_area=args.small_area)
        records.append(record)
        rows.append({
            "image": str(image_path), **matched,
            "detections": len(pred_xyxy),
            "mean_confidence": float(pred_conf.mean()) if len(pred_conf) else 0.0,
            "illumination_mean": illumination_mean,
            "detector_route": detector_route,
            "gate_mode": diagnostics["gate_mode"],
            "residual_l1": diagnostics["residual_l1"],
        })

    if not records:
        raise ValueError(f"None of the {len(paths)} image files in {args.images} could be decoded")

    class_ids = sorted({int(value) for record in records for value in record["gt"]["class"]})
    thresholds = np.arange(0.5, 0.96, 0.05)
    aps = [value for class_id in class_ids for threshold in thresholds if (value := class_ap(records, class_id, float(threshold))) is not None]
    aps50 = [value for class_id in class_ids if (value := class_ap(records, class_id, 0.5)) is not None]
    aps_small = [value for class_id in class_ids for threshold in thresholds if (value := class_ap(records, class_id, float(threshold), "small", args.small_area)) is not None]
    aps50_small = [value for class_id in class_ids if (value := class_ap(records, class_id, 0.5, "small", args.small_area)) is not None]
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "small_tp", "small_fn", "tiny_side_tp", "tiny_side_fn")}
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    summary = {
        "configuration": args.enhancement,
        "domain": args.domain or args.images.parent.name,
        "images_root": str(args.images.resolve()),
        "labels_root": str(args.labels.resolve()),
        "seed": args.seed,
        "images": len(records),
        "protocol": {
            "image_size": args.imgsz,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "small_area": args.small_area,
            "tiny_side": args.tiny_side,
            "detector_route_threshold": args.detector_route_threshold if bright_detector is not None else None,
        },
        "map50": float(np.mean(aps50)) if aps50 else 0.0,
        "map50_95": float(np.mean(aps)) if aps else 0.0,
        "ap_small_50": float(np.mean(aps50_small)) if aps50_small else 0.0,
        "ap_small_50_95": float(np.mean(aps_small)) if aps_small else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "small_recall": totals["small_tp"] / max(totals["small_tp"] + totals["small_fn"], 1),
        "tiny_side_recall": totals["tiny_side_tp"] / max(totals["tiny_side_tp"] + totals["tiny_side_fn"], 1),
        "fp_per_image": totals["fp"] / max(len(records), 1),
        "class_wise_recall": {},
        "gate_rates": {mode: sum(row["gate_mode"] == mode for row in rows) / max(len(rows), 1) for mode in sorted({row["gate_mode"] for row in rows})},
        "detector_route_rates": {
            route: sum(row["detector_route"] == route for row in rows) / max(len(rows), 1)
            for route in sorted({row["detector_route"] for row in rows})
        },
    }
    for class_id in class_ids:
        class_tp = class_gt = 0
        for record in records:
            filtered = {**record, "gt": {key: value[record["gt"]["class"] == class_id] for key, value in record["gt"].items()}}
            filtered["pred_xyxy"] = record["pred_xyxy"][record["pred_class"] == class_id]
            filtered["pred_conf"] = record["pred_conf"][record["pred_class"] == class_id]
            filtered["pred_class"] = np.full(len(filtered["pred_xyxy"]), class_id)
            matched = match_at_confidence(filtered, args.confidence, small_area=args.small_area)
            class_tp += matched["tp"]; class_gt += matched["tp"] + matched["fn"]
        summary["class_wise_recall"][str(class_id)] = class_tp / max(class_gt, 1)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.enhancement}_seed_{args.seed}"
    (args.output / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / f"{stem}_per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
