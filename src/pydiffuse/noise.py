import torch


def noise_tensor(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Adds a tensor of random noise to a tensor, with the amount determined by
    a noise level between 0 and 1. The tensor will be added to a tensor of
    noise, after each have been scaled by an amount that preserves a variance of
    1 while applying the given noise level."""

    noise = torch.randn_like(tensor)
    scale_noise, scale_original = _scaling_factors(noise_level)
    original = tensor * scale_original
    noise *= scale_noise
    return original + noise


def _scaling_factors(noise_level: float) -> tuple[float, float]:
    """For a given noise level, calculates the two scaling factors that the
    noise tensor and the image tensor should be scaled by to preserve a variance
    of 1."""

    return noise_level**0.5, (1 - noise_level) ** 0.5
