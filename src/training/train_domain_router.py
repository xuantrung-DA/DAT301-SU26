"""Train the learned domain router without using any test split."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from models.domain_router import DOMAIN_NAMES, LearnedDomainRouter

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DomainDataset(Dataset):
    def __init__(self, roots: list[Path], size: int, training: bool) -> None:
        self.rows = [(path, label) for label, root in enumerate(roots) for path in root.rglob("*") if path.suffix.lower() in EXTENSIONS]
        ops = [transforms.Resize((size, size))]
        if training:
            ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.08, 0.08, 0.05, 0.02)]
        ops += [transforms.ToTensor()]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int):
        path, label = self.rows[index]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        for target, prediction in zip(labels, predictions): confusion[target, prediction] += 1
    per_class = (confusion.diag() / confusion.sum(1).clamp_min(1)).tolist()
    return {"accuracy": float(confusion.diag().sum() / confusion.sum()), "per_class_recall": dict(zip(DOMAIN_NAMES, per_class)), "confusion": confusion.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    for split in ("train", "val"):
        for domain in DOMAIN_NAMES: parser.add_argument(f"--{split}-{domain.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/domain_router")); parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--size", type=int, default=160); parser.add_argument("--batch", type=int, default=64); parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    train_roots = [getattr(args, f"train_{name}") for name in DOMAIN_NAMES]; val_roots = [getattr(args, f"val_{name}") for name in DOMAIN_NAMES]
    train = DomainDataset(train_roots, args.size, True); val = DomainDataset(val_roots, args.size, False)
    counts = torch.bincount(torch.tensor([label for _, label in train.rows]), minlength=3); weights = torch.tensor([1.0 / counts[label] for _, label in train.rows])
    train_loader = DataLoader(train, batch_size=args.batch, sampler=WeightedRandomSampler(weights, len(train)), num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch * 2, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = LearnedDomainRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    args.output.mkdir(parents=True, exist_ok=True); history=[]; best=-1.0
    for epoch in range(args.epochs):
        model.train(); total=correct=0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True); logits=model(images); loss=criterion(logits,labels); loss.backward(); optimizer.step()
            total += labels.numel(); correct += int((logits.argmax(1)==labels).sum())
        model.eval(); metrics=evaluate(model,val_loader,device); metrics.update(epoch=epoch+1,train_accuracy=correct/total); history.append(metrics); print(json.dumps(metrics))
        if metrics["accuracy"] > best:
            best=metrics["accuracy"]; torch.save({"model":model.state_dict(),"size":args.size,"domains":DOMAIN_NAMES,"metrics":metrics},args.output/"best.pt")
    (args.output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps({"parameters":sum(p.numel() for p in model.parameters()),"train_counts":counts.tolist(),"best_val_accuracy":best,"best_metrics":max(history,key=lambda x:x["accuracy"])},indent=2),encoding="utf-8")

if __name__ == "__main__": main()
