"""Ultra-light bounded residual enhancer used by LADD-UAV.

The proposal explicitly avoids regenerating the complete image.  This module
predicts a bounded RGB residual with a 24/48/96 depthwise-separable backbone.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .dgrdm import DetectionGuidedRDM
from .gate import AdaptiveIlluminationGate, GateMode


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class FusionBlock(nn.Module):
    """DWConv + pointwise projection + a tiny channel gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = DepthwiseSeparableConv(channels, channels)
        self.gate = ChannelGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate(self.body(x))


class BoundedResidualEnhancer(nn.Module):
    """Three-scale 24/48/96 residual network with a hard parameter budget."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 96),
        fusion_blocks: int = 3,
        max_residual: float = 0.35,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("channels must contain exactly three scales")
        c1, c2, c3 = (int(value) for value in channels)
        self.channels = (c1, c2, c3)
        self.max_residual = float(max_residual)
        self.stem = DepthwiseSeparableConv(3, c1)
        self.down1 = DepthwiseSeparableConv(c1, c2, stride=2)
        self.down2 = DepthwiseSeparableConv(c2, c3, stride=2)
        self.bottleneck = nn.Sequential(*(FusionBlock(c3) for _ in range(fusion_blocks)))
        self.up2 = DepthwiseSeparableConv(c3 + c2, c2)
        self.up1 = DepthwiseSeparableConv(c2 + c1, c1)
        self.residual_head = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = self.stem(image)
        half = self.down1(full)
        quarter = self.bottleneck(self.down2(half))
        decoded_half = F.interpolate(quarter, size=half.shape[-2:], mode="bilinear", align_corners=False)
        decoded_half = self.up2(torch.cat((decoded_half, half), dim=1))
        decoded_full = F.interpolate(decoded_half, size=full.shape[-2:], mode="bilinear", align_corners=False)
        decoded_full = self.up1(torch.cat((decoded_full, full), dim=1))
        residual = torch.tanh(self.residual_head(decoded_full)) * self.max_residual
        return residual, decoded_full


class LADDEnhancer(nn.Module):
    """Adaptive bypass/light/full enhancement pipeline.

    ``forward`` returns only the enhanced image so the module remains easy to
    export.  Training and diagnostics should use ``forward_with_details``.
    """

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 96),
        fusion_blocks: int = 3,
        max_residual: float = 0.35,
        foreground_alpha: float = 1.0,
        background_alpha: float = 0.25,
        light_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        self.enhancer = BoundedResidualEnhancer(channels, fusion_blocks, max_residual)
        self.gate = AdaptiveIlluminationGate()
        self.dgrdm = DetectionGuidedRDM(
            feature_channels=int(channels[0]),
            max_residual=max_residual,
            foreground_alpha=foreground_alpha,
            background_alpha=background_alpha,
        )
        self.light_alpha = float(light_alpha)

    @staticmethod
    def _forced_probabilities(image: torch.Tensor, mode: str | int) -> torch.Tensor:
        if isinstance(mode, str):
            try:
                index = GateMode[mode.upper()].value
            except KeyError as exc:
                raise ValueError("force_mode must be bypass, light, or full") from exc
        else:
            index = int(mode)
        if index not in (0, 1, 2):
            raise ValueError("force_mode must be 0, 1, or 2")
        probabilities = image.new_zeros((image.shape[0], 3))
        probabilities[:, index] = 1.0
        return probabilities

    def forward_with_details(
        self,
        image: torch.Tensor,
        detector_confidence: torch.Tensor | None = None,
        heatmap: torch.Tensor | None = None,
        hard_gate: bool | None = None,
        force_mode: str | int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        probabilities, gate_features, gate_logits = self.gate(
            image,
            detector_confidence=detector_confidence,
            hard=(not self.training if hard_gate is None else hard_gate),
        )
        if force_mode is not None:
            probabilities = self._forced_probabilities(image, force_mode)

        efficient_hard = (not self.training) and image.shape[0] == 1 and (
            force_mode is not None or hard_gate is None or hard_gate
        )
        if efficient_hard and int(probabilities.argmax(dim=1).item()) == GateMode.BYPASS:
            zero_residual = torch.zeros_like(image)
            zero_heatmap = image.new_zeros((1, 1, image.shape[-2], image.shape[-1]))
            return image, {
                "gate_probabilities": probabilities,
                "gate_logits": gate_logits,
                "gate_features": gate_features,
                "gate_mode": probabilities.argmax(dim=1),
                "base_residual": zero_residual,
                "applied_residual": zero_residual,
                "objectness_heatmap": zero_heatmap,
                "region_mask": zero_heatmap,
                "foreground_residual": zero_residual,
                "background_residual": zero_residual,
            }

        residual, features = self.enhancer(image)
        light = torch.clamp(image + self.light_alpha * residual, 0.0, 1.0)
        if efficient_hard and int(probabilities.argmax(dim=1).item()) == GateMode.LIGHT:
            zero_heatmap = image.new_zeros((1, 1, image.shape[-2], image.shape[-1]))
            return light, {
                "gate_probabilities": probabilities,
                "gate_logits": gate_logits,
                "gate_features": gate_features,
                "gate_mode": probabilities.argmax(dim=1),
                "base_residual": residual,
                "applied_residual": light - image,
                "objectness_heatmap": zero_heatmap,
                "region_mask": zero_heatmap,
                "foreground_residual": residual,
                "background_residual": torch.clamp(residual, -self.dgrdm.background_bound, self.dgrdm.background_bound),
            }
        full, region = self.dgrdm(image, residual, features, target_heatmap=heatmap)

        weights = probabilities[:, :, None, None, None]
        candidates = torch.stack((image, light, full), dim=1)
        enhanced = (weights * candidates).sum(dim=1)
        details = {
            "gate_probabilities": probabilities,
            "gate_logits": gate_logits,
            "gate_features": gate_features,
            "gate_mode": probabilities.argmax(dim=1),
            "base_residual": residual,
            "applied_residual": enhanced - image,
            "objectness_heatmap": region["predicted_heatmap"],
            "region_mask": region["mask"],
            "foreground_residual": region["foreground_residual"],
            "background_residual": region["background_residual"],
        }
        return enhanced, details

    def forward(
        self,
        image: torch.Tensor,
        detector_confidence: torch.Tensor | None = None,
        heatmap: torch.Tensor | None = None,
        hard_gate: bool | None = None,
        force_mode: str | int | None = None,
    ) -> torch.Tensor:
        enhanced, _ = self.forward_with_details(
            image,
            detector_confidence=detector_confidence,
            heatmap=heatmap,
            hard_gate=hard_gate,
            force_mode=force_mode,
        )
        return enhanced


# Backwards-compatible name used by existing entry points and checkpoints.
EnhancementGenerator = LADDEnhancer
