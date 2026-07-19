"""Losses from the LADD-UAV proposal.

The GAN helpers are retained only for the optional legacy ablation.  The main
training path uses bounded reconstruction, detection, DG-RDM, identity, tiny
box geometry, and gate compute-proxy terms.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def ssim_index(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Differentiable channel-wise SSIM averaged over the batch."""
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_y = F.avg_pool2d(target, window_size, 1, padding)
    sigma_x = F.avg_pool2d(prediction.square(), window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.2) -> torch.Tensor:
    return charbonnier_loss(prediction, target) + float(ssim_weight) * (1.0 - ssim_index(prediction, target))


def _gradient_xy(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[:, :, :, 1:] - image[:, :, :, :-1], image[:, :, 1:, :] - image[:, :, :-1, :]


def foreground_edge_loss(prediction: torch.Tensor, reference: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    mask = F.interpolate(foreground_mask, prediction.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    pred_x, pred_y = _gradient_xy(prediction)
    ref_x, ref_y = _gradient_xy(reference)
    mask_x = 0.5 * (mask[:, :, :, 1:] + mask[:, :, :, :-1])
    mask_y = 0.5 * (mask[:, :, 1:, :] + mask[:, :, :-1, :])
    loss_x = ((pred_x - ref_x).abs() * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    loss_y = ((pred_y - ref_y).abs() * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def background_smoothness_loss(residual: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    mask = F.interpolate(foreground_mask, residual.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    background = 1.0 - mask
    dx, dy = _gradient_xy(residual)
    bg_x = 0.5 * (background[:, :, :, 1:] + background[:, :, :, :-1])
    bg_y = 0.5 * (background[:, :, 1:, :] + background[:, :, :-1, :])
    tv_x = (dx.abs() * bg_x).sum() / bg_x.sum().clamp_min(1.0)
    tv_y = (dy.abs() * bg_y).sum() / bg_y.sum().clamp_min(1.0)
    high_frequency = residual - F.avg_pool2d(residual, 3, 1, 1)
    noise_penalty = (high_frequency.abs() * background).sum() / background.sum().clamp_min(1.0)
    return 0.5 * (tv_x + tv_y) + 0.25 * noise_penalty


def identity_loss(prediction: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, source)


def heatmap_supervision_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.interpolate(target.to(predicted), predicted.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    logits = torch.logit(predicted.float().clamp(1e-5, 1.0 - 1e-5))
    return F.binary_cross_entropy_with_logits(logits, target.float())


def latency_proxy_loss(gate_probabilities: torch.Tensor, mode_costs: tuple[float, float, float] = (0.0, 0.35, 1.0)) -> torch.Tensor:
    costs = gate_probabilities.new_tensor(mode_costs)
    return (gate_probabilities * costs).sum(dim=1).mean()


def normalized_wasserstein_loss(
    predicted_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    constant: float = 12.8,
) -> torch.Tensor:
    """NWD loss for already-matched boxes in pixel-space xywh format."""
    if predicted_xywh.numel() == 0:
        return predicted_xywh.sum() * 0.0
    center_distance = (predicted_xywh[:, :2] - target_xywh[:, :2]).square().sum(dim=1)
    size_distance = ((predicted_xywh[:, 2:] - target_xywh[:, 2:]) / 2.0).square().sum(dim=1)
    wasserstein = torch.sqrt((center_distance + size_distance).clamp_min(1e-8))
    similarity = torch.exp(-wasserstein / float(constant))
    return (1.0 - similarity).mean()


def tiny_nwd_from_candidates(
    predicted_xywh: torch.Tensor,
    predicted_scores: torch.Tensor,
    labels: list[torch.Tensor],
    image_size: tuple[int, int],
    topk: int = 300,
    small_area: float = 32.0**2,
    constant: float = 12.8,
) -> torch.Tensor:
    """Match each tiny GT box to the nearest high-confidence candidate."""
    height, width = image_size
    losses: list[torch.Tensor] = []
    for batch_index, rows in enumerate(labels):
        if rows.numel() == 0:
            continue
        target = rows[:, 1:5].to(predicted_xywh)
        scale = target.new_tensor((width, height, width, height))
        target = target * scale
        tiny = target[:, 2] * target[:, 3] < float(small_area)
        target = target[tiny]
        if target.numel() == 0:
            continue
        scores = predicted_scores[batch_index]
        candidates = predicted_xywh[batch_index]
        keep = scores.topk(min(int(topk), scores.numel())).indices
        candidates = candidates[keep]
        pairwise = torch.cdist(target[:, :2], candidates[:, :2])
        nearest = candidates[pairwise.argmin(dim=1)]
        losses.append(normalized_wasserstein_loss(nearest, target, constant))
    return torch.stack(losses).mean() if losses else predicted_xywh.sum() * 0.0


class PerceptualLoss(nn.Module):
    """Legacy optional ablation; not used by the proposal's main path."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.features = vgg19(weights=weights).features[:16].eval()
        except Exception as exc:
            if pretrained:
                raise RuntimeError("Could not load pretrained VGG19. Download weights first or pass pretrained=False.") from exc
            raise
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.mse_loss(self.features(prediction), self.features(target))


def lsgan_discriminator_loss(real_outputs, fake_outputs):
    return sum(
        0.5 * (F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake)))
        for real, fake in zip(real_outputs, fake_outputs)
    ) / len(real_outputs)


def lsgan_generator_loss(fake_outputs):
    return sum(F.mse_loss(fake, torch.ones_like(fake)) for fake in fake_outputs) / len(fake_outputs)


def image_quality_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = F.mse_loss(prediction.clamp(0, 1), target.clamp(0, 1)).item()
    psnr = 100.0 if mse == 0 else -10.0 * torch.log10(torch.tensor(mse)).item()
    return psnr, float(ssim_index(prediction.float(), target.float()).item())
