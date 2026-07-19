"""Differentiable bridge to Ultralytics YOLO for detection-aware training."""
from __future__ import annotations

import torch
from torch import nn


class FrozenYoloBridge(nn.Module):
    def __init__(self, weights: str, attention_layer: int = 4, stage: str = "C", neck_start: int = 10):
        super().__init__()
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
        except ImportError as exc:
            raise ImportError("Install ultralytics to use joint training") from exc
        wrapper = YOLO(weights)
        self.detector = wrapper.model
        # Checkpoints only retain a small args dict; the loss needs the full
        # current Ultralytics hyperparameter namespace (box/cls/dfl gains).
        self.detector.args = get_cfg(overrides=self.detector.args)
        self.neck_start = int(neck_start)
        self.set_stage(stage)
        # A dict input selects the loss path; eval mode also freezes BatchNorm buffers.
        self.detector.eval()
        self._features = None
        layer = self.detector.model[attention_layer]
        layer.register_forward_hook(self._capture_features)

    def set_stage(self, stage: str) -> None:
        """Stage C freezes YOLO; Stage D unfreezes neck/head only."""
        stage = stage.upper()
        if stage not in {"C", "D"}:
            raise ValueError("Detector stage must be C or D")
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        if stage == "D":
            for index, layer in enumerate(self.detector.model):
                if index >= self.neck_start:
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)
        self.stage = stage

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.detector.parameters() if parameter.requires_grad]

    def _capture_features(self, _module, _inputs, output):
        feature = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(feature):
            self._features = feature

    @staticmethod
    def build_batch(images: torch.Tensor, labels: list[torch.Tensor]) -> dict:
        cls, boxes, indices = [], [], []
        for batch_index, rows in enumerate(labels):
            if rows.numel():
                cls.append(rows[:, :1]); boxes.append(rows[:, 1:5])
                indices.append(torch.full((len(rows),), batch_index, device=images.device, dtype=torch.long))
        return {
            "img": images,
            "cls": torch.cat(cls).to(images.device) if cls else torch.empty((0, 1), device=images.device),
            "bboxes": torch.cat(boxes).to(images.device) if boxes else torch.empty((0, 4), device=images.device),
            "batch_idx": torch.cat(indices) if indices else torch.empty((0,), device=images.device, dtype=torch.long),
        }

    def forward(self, images: torch.Tensor, labels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        self._features = None
        output = self.detector(self.build_batch(images, labels))
        if not isinstance(output, tuple):
            raise RuntimeError("Ultralytics training API changed: expected (loss, loss_items)")
        loss, items = output
        if loss.ndim:
            loss = loss.sum()
        attention = None
        if self._features is not None:
            attention = self._features.abs().mean(dim=1, keepdim=True)
        return loss, attention, items

    def predict_candidates(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable decoded xywh candidates and max class scores."""
        output = self.detector.predict(images)
        prediction = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(prediction) or prediction.ndim != 3 or prediction.shape[1] < 5:
            raise RuntimeError("Ultralytics prediction layout changed; expected [B,4+classes,N]")
        boxes = prediction[:, :4].permute(0, 2, 1)
        scores = prediction[:, 4:].amax(dim=1)
        return boxes, scores

    @torch.no_grad()
    def confidence(self, images: torch.Tensor, topk: int = 20) -> torch.Tensor:
        _, scores = self.predict_candidates(images)
        k = min(int(topk), scores.shape[1])
        return scores.topk(k, dim=1).values.mean(dim=1) if k else scores.new_zeros(images.shape[0])
