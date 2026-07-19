"""Lightweight learned router for clean, synthetic-low-light and real-low-light domains."""
from __future__ import annotations

import torch
from torch import nn


DOMAIN_NAMES = ("clean", "synthetic_lowlight", "real_lowlight")


class LearnedDomainRouter(nn.Module):
    """A sub-50K parameter image router intended to run before one detector branch."""

    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1, groups=8, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 40, 3, 2, 1, groups=8, bias=False), nn.BatchNorm2d(40), nn.SiLU(),
            nn.Conv2d(40, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))

    @torch.inference_mode()
    def route(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self(image).softmax(1)
        return probabilities.argmax(1), probabilities
