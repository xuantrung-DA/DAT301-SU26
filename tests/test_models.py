import numpy as np
import torch

from src.models import LADDEnhancer, build_gaussian_box_heatmaps
from src.preprocess.lowlight import LEVEL_CONFIGS, degrade_image_bgr
from src.preprocess.make_lowlight_visdrone import choose_mix_level
from src.training.losses import (
    background_smoothness_loss,
    foreground_edge_loss,
    latency_proxy_loss,
    reconstruction_loss,
)


def test_proposal_model_budget_shape_and_range():
    model = LADDEnhancer().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters <= 350_000
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        output = model(image, force_mode="full")
    assert output.shape == image.shape
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_gate_probabilities_and_exact_bypass():
    model = LADDEnhancer(channels=(8, 16, 24), fusion_blocks=1).eval()
    bright = torch.full((1, 3, 32, 32), 0.9)
    dark = torch.full((1, 3, 32, 32), 0.05)
    with torch.no_grad():
        bypassed, bright_details = model.forward_with_details(bright, force_mode="bypass")
        _, dark_details = model.forward_with_details(dark)
    assert torch.equal(bypassed, bright)
    assert torch.allclose(bright_details["gate_probabilities"].sum(1), torch.ones(1))
    assert int(dark_details["gate_mode"][0]) == 2


def test_gt_gaussian_heatmap_covers_box_center():
    labels = [torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]])]
    heatmap = build_gaussian_box_heatmaps(labels, 32, 32)
    assert heatmap.shape == (1, 1, 32, 32)
    assert heatmap[0, 0, 16, 16] > 0.9
    assert heatmap[0, 0, 0, 0] < 0.01


def test_proposal_losses_are_finite_and_differentiable():
    source = torch.rand(1, 3, 32, 32)
    prediction = source.clone().requires_grad_(True)
    reference = torch.rand_like(source)
    mask = torch.zeros(1, 1, 8, 8); mask[:, :, 2:6, 2:6] = 1
    gate = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    loss = (
        reconstruction_loss(prediction, reference)
        + foreground_edge_loss(prediction, reference, mask)
        + background_smoothness_loss(prediction - source, mask)
        + latency_proxy_loss(gate)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert gate.grad is not None


def test_lowlight_protocol_is_deterministic_and_registered_ranges_match():
    image = np.full((32, 48, 3), 180, dtype=np.uint8)
    first, metadata = degrade_image_bgr(image, "LL2", 3407, return_metadata=True)
    second = degrade_image_bgr(image, "LL2", 3407)
    assert np.array_equal(first, second)
    config = LEVEL_CONFIGS["LL2"]
    assert config.exposure_range == (0.30, 0.55)
    assert config.gamma_range == (2.0, 3.0)
    assert config.read_noise_range[0] <= metadata["read_noise_sigma"] <= config.read_noise_range[1]


def test_llmix_exact_registered_distribution_over_seed_buckets():
    levels = [choose_mix_level(seed) for seed in range(100)]
    assert levels.count("CLEAN") == 20
    assert levels.count("LL1") == 32
    assert levels.count("LL2") == 32
    assert levels.count("LL3") == 16
