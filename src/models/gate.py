"""Three-mode illumination/noise/confidence gate for LADD-UAV."""
from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F


class GateMode(IntEnum):
    BYPASS = 0
    LIGHT = 1
    FULL = 2


class AdaptiveIlluminationGate(nn.Module):
    """Return differentiable bypass/light/full probabilities.

    A deterministic illumination prior makes an untrained gate safe.  A tiny
    MLP learns a task-specific correction during Stage C/D.
    """

    def __init__(self, hidden_channels: int = 16, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.correction = nn.Sequential(
            nn.Linear(4, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    @staticmethod
    def statistics(image: torch.Tensor, detector_confidence: torch.Tensor | None = None) -> torch.Tensor:
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        mean_luma = luminance.mean(dim=(1, 2))
        dark_ratio = torch.sigmoid((0.30 - luminance) * 16.0).mean(dim=(1, 2))
        dx = torch.abs(luminance[:, :, 1:] - luminance[:, :, :-1]).mean(dim=(1, 2))
        dy = torch.abs(luminance[:, 1:, :] - luminance[:, :-1, :]).mean(dim=(1, 2))
        noise_proxy = (dx + dy).clamp(0.0, 1.0)
        if detector_confidence is None:
            confidence = torch.zeros_like(mean_luma)
        else:
            confidence = detector_confidence.to(image).reshape(image.shape[0], -1).mean(dim=1).clamp(0.0, 1.0)
        return torch.stack((mean_luma, dark_ratio, noise_proxy, confidence), dim=1)

    @staticmethod
    def prior_logits(features: torch.Tensor) -> torch.Tensor:
        mean_luma, _dark_ratio, noise, confidence = features.unbind(dim=1)
        bypass = 8.0 * (mean_luma - 0.55) + 1.5 * (confidence - 0.25) - noise
        light = 1.0 - 12.0 * torch.abs(mean_luma - 0.42) - 0.5 * noise
        full = 8.0 * (0.32 - mean_luma) + 2.0 * noise - 0.5 * confidence
        return torch.stack((bypass, light, full), dim=1)

    def forward(
        self,
        image: torch.Tensor,
        detector_confidence: torch.Tensor | None = None,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.statistics(image, detector_confidence)
        logits = self.prior_logits(features) + self.correction(features)
        probabilities = F.softmax(logits / max(self.temperature, 1e-4), dim=1)
        if hard:
            one_hot = F.one_hot(probabilities.argmax(dim=1), num_classes=3).to(probabilities.dtype)
            probabilities = one_hot if not self.training else one_hot + probabilities - probabilities.detach()
        return probabilities, features, logits
