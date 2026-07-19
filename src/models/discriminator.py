"""Multi-scale 70x70 PatchGAN discriminator using raw LSGAN logits."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        layers = [nn.Conv2d(3, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        in_c = base_channels
        for mult in (2, 4, 8):
            out_c = base_channels * mult
            layers += [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)]
            in_c = out_c
        layers += [nn.Conv2d(in_c, in_c, 4, 1, 1, bias=False), nn.BatchNorm2d(in_c), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, scales: int = 2, base_channels: int = 64):
        super().__init__()
        self.discriminators = nn.ModuleList(PatchDiscriminator(base_channels) for _ in range(scales))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(image))
            if index + 1 < len(self.discriminators):
                image = F.avg_pool2d(image, 3, 2, 1)
        return outputs
