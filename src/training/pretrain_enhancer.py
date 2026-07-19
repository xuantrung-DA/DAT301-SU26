"""Stage B: warm up the bounded residual enhancer on paired LOL data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import LADDEnhancer
from training.common import save_checkpoint, seed_everything, write_json
from training.data import PairedImageDataset
from training.losses import identity_loss, image_quality_metrics, reconstruction_loss


def build_model(config: dict) -> LADDEnhancer:
    model = config.get("model", {})
    return LADDEnhancer(
        channels=model.get("channels", [24, 48, 96]),
        fusion_blocks=model.get("fusion_blocks", 3),
        max_residual=model.get("max_residual", 0.35),
        foreground_alpha=model.get("foreground_alpha", 1.0),
        background_alpha=model.get("background_alpha", 0.25),
        light_alpha=model.get("light_alpha", 0.35),
    )


def warmup_forward(model: LADDEnhancer, low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual, _ = model.enhancer(low)
    return torch.clamp(low + residual, 0.0, 1.0), residual


@torch.no_grad()
def validate(model: LADDEnhancer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_total = ssim_total = 0.0
    for batch in loader:
        prediction, _ = warmup_forward(model, batch["low"].to(device))
        psnr, ssim = image_quality_metrics(prediction, batch["high"].to(device))
        psnr_total += psnr
        ssim_total += ssim
    model.train()
    count = max(len(loader), 1)
    return {"psnr": psnr_total / count, "ssim": ssim_total / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, hyperparameters = config["paths"], config["pretrain"]

    train_dataset = PairedImageDataset(
        Path(paths["lol_train_low"]), Path(paths["lol_train_high"]), config["image_size"], True,
        hyperparameters.get("max_samples"),
    )
    validation_dataset = PairedImageDataset(
        Path(paths["lol_val_low"]), Path(paths["lol_val_high"]), config["image_size"],
        max_samples=hyperparameters.get("max_val_samples"),
    )
    train_loader = DataLoader(train_dataset, hyperparameters["batch_size"], shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, 1, num_workers=1)
    model = build_model(config).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("generator", state))

    optimizer = torch.optim.AdamW(model.enhancer.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler(device.type, enabled=hyperparameters.get("amp", True) and device.type == "cuda")
    best_psnr, history = float("-inf"), []
    checkpoint_dir = Path(paths["checkpoints"])
    for epoch in range(1, hyperparameters["epochs"] + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "identity": 0.0}
        for batch in tqdm(train_loader, desc=f"stage-b {epoch}/{hyperparameters['epochs']}"):
            low, high = batch["low"].to(device), batch["high"].to(device)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                prediction, _ = warmup_forward(model, low)
                reconstruction = reconstruction_loss(prediction, high)
                identity = identity_loss(prediction, low)
                loss = reconstruction + hyperparameters.get("lambda_identity", 0.1) * identity
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.enhancer.parameters(), hyperparameters.get("gradient_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            for key, value in (("loss", loss), ("reconstruction", reconstruction), ("identity", identity)):
                totals[key] += float(value.detach())

        metrics = validate(model, validation_loader, device)
        metrics.update({key: value / max(len(train_loader), 1) for key, value in totals.items()})
        metrics["epoch"] = epoch
        history.append(metrics)
        print(metrics)
        state = {"epoch": epoch, "generator": model.state_dict(), "config": config, "metrics": metrics, "architecture": "LADD-UAV"}
        save_checkpoint(checkpoint_dir / "stage_b_last.pt", **state)
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            save_checkpoint(checkpoint_dir / "stage_b_best.pt", **state)
        write_json(checkpoint_dir / "stage_b_history.json", history)


if __name__ == "__main__":
    main()
