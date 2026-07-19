"""Detection-Guided Region Decoupling Module (DG-RDM)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_gaussian_box_heatmaps(
    labels: list[torch.Tensor],
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create GT Gaussian heatmaps from YOLO ``class,cx,cy,w,h`` labels."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if device is None:
        device = labels[0].device if labels else torch.device("cpu")
    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)
    result = torch.zeros((len(labels), 1, height, width), device=device, dtype=dtype)
    for batch_index, rows in enumerate(labels):
        if rows.numel() == 0:
            continue
        for row in rows.to(device=device, dtype=dtype):
            cx, cy = row[1] * width, row[2] * height
            sigma_x = torch.clamp(row[3] * width / 3.0, min=1.0)
            sigma_y = torch.clamp(row[4] * height / 3.0, min=1.0)
            gaussian = torch.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
            result[batch_index, 0] = torch.maximum(result[batch_index, 0], gaussian)
    return result


class _RegionRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 6, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        foreground, background = self.net(x).chunk(2, dim=1)
        return foreground, background


class DetectionGuidedRDM(nn.Module):
    """Predict objectness and apply different residual bounds to FG/BG."""

    def __init__(
        self,
        feature_channels: int = 24,
        hidden_channels: int | None = None,
        max_residual: float = 0.35,
        foreground_alpha: float = 1.0,
        background_alpha: float = 0.25,
        background_bound: float = 0.10,
    ) -> None:
        super().__init__()
        hidden_channels = feature_channels if hidden_channels is None else int(hidden_channels)
        self.max_residual = float(max_residual)
        self.foreground_alpha = float(foreground_alpha)
        self.background_alpha = float(background_alpha)
        self.background_bound = float(background_bound)
        self.objectness = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1, groups=feature_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.refiner = _RegionRefiner(feature_channels + 3, hidden_channels)

    def forward(
        self,
        image: torch.Tensor,
        base_residual: torch.Tensor,
        features: torch.Tensor,
        target_heatmap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        predicted = torch.sigmoid(self.objectness(features))
        if target_heatmap is not None:
            mask = F.interpolate(target_heatmap.to(predicted), predicted.shape[-2:], mode="bilinear", align_corners=False)
            mask = mask.clamp(0.0, 1.0)
        else:
            mask = predicted

        fg_delta, bg_delta = self.refiner(torch.cat((features, base_residual), dim=1))
        foreground = torch.clamp(
            base_residual + 0.10 * torch.tanh(fg_delta),
            -self.max_residual,
            self.max_residual,
        )
        background = torch.clamp(
            base_residual + 0.05 * torch.tanh(bg_delta),
            -self.background_bound,
            self.background_bound,
        )
        applied = self.foreground_alpha * mask * foreground
        applied = applied + self.background_alpha * (1.0 - mask) * background
        enhanced = torch.clamp(image + applied, 0.0, 1.0)
        return enhanced, {
            "predicted_heatmap": predicted,
            "mask": mask,
            "foreground_residual": foreground,
            "background_residual": background,
            "applied_residual": applied,
        }
