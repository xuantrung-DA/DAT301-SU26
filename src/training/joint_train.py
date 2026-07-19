"""Stage C/D detection-guided training for LADD-UAV."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import LADDEnhancer, build_gaussian_box_heatmaps
from training.common import save_checkpoint, seed_everything, write_json
from training.data import YoloEnhancementDataset, detection_collate
from training.detector_bridge import FrozenYoloBridge
from training.losses import (
    background_smoothness_loss,
    foreground_edge_loss,
    heatmap_supervision_loss,
    identity_loss,
    latency_proxy_loss,
    reconstruction_loss,
    tiny_nwd_from_candidates,
)
from training.pretrain_enhancer import build_model


def _load_generator(model: LADDEnhancer, checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu")
    try:
        model.load_state_dict(state.get("generator", state))
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is not compatible with the proposal architecture. "
            "Run Stage B with training.pretrain_enhancer first."
        ) from exc
    return state if isinstance(state, dict) else {}


def _identity_on_clean(enhanced: torch.Tensor, source: torch.Tensor, clean_mask: torch.Tensor) -> torch.Tensor:
    if clean_mask.any():
        return identity_loss(enhanced[clean_mask], source[clean_mask])
    return enhanced.sum() * 0.0


def _detector_weight(epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 1:
        return 1.0
    return min(1.0, max(0.0, (epoch - 1) / float(ramp_epochs - 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stage_b_best.pt"))
    parser.add_argument("--stage", choices=["C", "D"], default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, hyperparameters = config["paths"], config["joint"]
    stage = (args.stage or hyperparameters.get("stage", "C")).upper()

    lowlight_root, clean_root = Path(paths["visdrone_llmix"]), Path(paths["visdrone_clean"])
    dataset = YoloEnhancementDataset(
        lowlight_root / "images/train",
        lowlight_root / "labels/train",
        clean_root / "images/train",
        config["image_size"],
        hyperparameters.get("max_samples"),
    )
    loader = DataLoader(
        dataset,
        hyperparameters["batch_size"],
        shuffle=True,
        num_workers=hyperparameters.get("workers", 2),
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    checkpoint_state = _load_generator(model, args.checkpoint)
    detector = FrozenYoloBridge(
        config["detector"]["weights"],
        attention_layer=config["detector"].get("attention_layer", 4),
        stage=stage,
        neck_start=config["detector"].get("neck_start", 10),
    ).to(device)
    if "detector" in checkpoint_state:
        detector.detector.load_state_dict(checkpoint_state["detector"])

    optimizer_groups = [{"params": model.parameters(), "lr": hyperparameters["lr_enhancer"]}]
    detector_parameters = detector.trainable_parameters()
    if detector_parameters:
        optimizer_groups.append({"params": detector_parameters, "lr": hyperparameters["lr_enhancer"] / 10.0})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=hyperparameters.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler(device.type, enabled=hyperparameters.get("amp", True) and device.type == "cuda")
    checkpoint_dir = Path(paths["checkpoints"])
    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tensorboard" / f"stage_{stage.lower()}"))
    history: list[dict[str, float]] = []
    weights = hyperparameters.get("loss_weights", {})
    compute_nwd = bool(hyperparameters.get("compute_nwd", False) and weights.get("nwd", 0.0) > 0)

    for epoch in range(1, hyperparameters["epochs"] + 1):
        model.train()
        detector.detector.eval()
        totals = {
            "total": 0.0, "det": 0.0, "nwd": 0.0, "fg_edge": 0.0,
            "bg_smooth": 0.0, "identity": 0.0, "rec": 0.0, "latency": 0.0,
            "heatmap": 0.0, "gate_supervision": 0.0, "skip_rate": 0.0, "residual_l1": 0.0,
        }
        det_scale = _detector_weight(epoch, hyperparameters.get("detector_ramp_epochs", 5))
        rec_start = hyperparameters.get("rec_start", 0.5)
        rec_end = weights.get("rec", 0.25)
        ramp = min(1.0, epoch / max(hyperparameters.get("detector_ramp_epochs", 5), 1))
        rec_weight = rec_start + (rec_end - rec_start) * ramp

        for batch in tqdm(loader, desc=f"stage-{stage.lower()} {epoch}/{hyperparameters['epochs']}"):
            low, high = batch["low"].to(device), batch["high"].to(device)
            labels = [rows.to(device) for rows in batch["labels"]]
            clean_mask = batch["is_clean"].to(device)
            heatmap = build_gaussian_box_heatmaps(
                labels,
                max(1, low.shape[-2] // 4),
                max(1, low.shape[-1] // 4),
                device=device,
                dtype=low.dtype,
            )
            detector_confidence = None
            if hyperparameters.get("use_detector_confidence", True):
                detector_confidence = detector.confidence(low)

            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                enhanced, details = model.forward_with_details(
                    low,
                    detector_confidence=detector_confidence,
                    heatmap=heatmap,
                    hard_gate=False,
                )
                detector_loss, _, _ = detector(enhanced, labels)
                if compute_nwd:
                    predicted_boxes, predicted_scores = detector.predict_candidates(enhanced)
                    nwd = tiny_nwd_from_candidates(
                        predicted_boxes,
                        predicted_scores,
                        labels,
                        enhanced.shape[-2:],
                        topk=hyperparameters.get("nwd_topk", 300),
                    )
                else:
                    nwd = enhanced.sum() * 0.0
                fg_edge = foreground_edge_loss(enhanced, high, heatmap)
                bg_smooth = background_smoothness_loss(details["applied_residual"], heatmap)
                identity = _identity_on_clean(enhanced, low, clean_mask)
                reconstruction = reconstruction_loss(enhanced, high)
                latency = latency_proxy_loss(details["gate_probabilities"])
                heatmap_loss = heatmap_supervision_loss(details["objectness_heatmap"], heatmap)
                # Prevent the learned correction from collapsing every image to
                # FULL. Clean samples must bypass, moderately dark samples use
                # LIGHT, and only very dark samples are routed to FULL.
                mean_luma = details["gate_features"][:, 0]
                gate_targets = torch.where(
                    clean_mask,
                    torch.zeros_like(mean_luma, dtype=torch.long),
                    torch.where(
                        mean_luma < hyperparameters.get("full_luma_threshold", 0.25),
                        torch.full_like(mean_luma, 2, dtype=torch.long),
                        torch.ones_like(mean_luma, dtype=torch.long),
                    ),
                )
                gate_supervision = F.cross_entropy(details["gate_logits"], gate_targets)
                total = (
                    weights.get("det", 1.0) * det_scale * detector_loss
                    + weights.get("nwd", 0.20) * nwd
                    + weights.get("fg_edge", 0.15) * fg_edge
                    + weights.get("bg_smooth", 0.05) * bg_smooth
                    + weights.get("identity", 0.10) * identity
                    + rec_weight * reconstruction
                    + weights.get("latency", 0.02) * latency
                    + weights.get("heatmap", 0.10) * heatmap_loss
                    + weights.get("gate_supervision", 0.50) * gate_supervision
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            trainable = list(model.parameters()) + detector_parameters
            torch.nn.utils.clip_grad_norm_(trainable, hyperparameters.get("gradient_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()

            batch_values = {
                "total": total, "det": detector_loss, "nwd": nwd, "fg_edge": fg_edge,
                "bg_smooth": bg_smooth, "identity": identity, "rec": reconstruction,
                "latency": latency, "heatmap": heatmap_loss, "gate_supervision": gate_supervision,
                "skip_rate": (details["gate_mode"] == 0).float().mean(),
                "residual_l1": details["applied_residual"].abs().mean(),
            }
            for key, value in batch_values.items():
                totals[key] += float(value.detach())

        metrics = {key: value / max(len(loader), 1) for key, value in totals.items()}
        metrics.update({"epoch": epoch, "detector_weight": det_scale, "reconstruction_weight": rec_weight, "stage": stage})
        history.append(metrics)
        print(metrics)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)
        state = {
            "epoch": epoch,
            "generator": model.state_dict(),
            "detector": detector.detector.state_dict(),
            "detector_source": config["detector"]["weights"],
            "config": config,
            "metrics": metrics,
            "architecture": "LADD-UAV",
            "stage": stage,
        }
        save_checkpoint(checkpoint_dir / f"stage_{stage.lower()}_last.pt", **state)
        save_checkpoint(checkpoint_dir / "joint_last.pt", **state)
        write_json(checkpoint_dir / f"stage_{stage.lower()}_history.json", history)

    writer.close()


if __name__ == "__main__":
    main()
