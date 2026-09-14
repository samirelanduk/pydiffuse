import math
import re

import safetensors
import torch

from .layers import linear, silu

MODEL_PREFIX = "model.diffusion_model"


def unet(
    latent: torch.Tensor,
    noise_level: float,
    conditioning: torch.Tensor,
    model: safetensors.safe_open,
) -> torch.Tensor:
    model_tensors = _get_unet_tensors(model)
    t = _noise_to_t(noise_level)
    sinusoid_width = model_tensors["time_embed"][0]["weight"].shape[1]
    sinusoids = _timestep_sinusoids(t, sinusoid_width)
    _time_embed(sinusoids, model_tensors["time_embed"])
    conditioning = _combine_chunks(conditioning)


def _get_unet_tensors(model: safetensors.safe_open) -> dict:
    """Finds the tensors used in the UNet. The time embedding is a list of
    linear layers, which have activations between them that have no tensors
    of their own."""

    return {
        "time_embed": [
            _get_layer(model, f"{MODEL_PREFIX}.time_embed.{index}")
            for index in _get_numbers(model, f"{MODEL_PREFIX}.time_embed")
        ],
    }


def _get_numbers(model: safetensors.safe_open, prefix: str) -> list[int]:
    """Gets a list of the numbers that the model's keys use directly under a
    prefix, such as the layers within the time embedding."""

    numbers = set()
    keys = model.keys()
    for key in keys:
        number_match = re.search(rf"^{prefix}\.(\d+)\.", key)
        if not number_match:
            continue
        numbers.add(int(number_match.group(1)))
    return sorted(numbers)


def _get_layer(model: safetensors.safe_open, prefix: str) -> dict | None:
    """Gets the weight and bias of a single layer, or None if the model has no
    such layer."""

    keys = model.keys()
    if f"{prefix}.weight" not in keys or f"{prefix}.bias" not in keys:
        return None
    return {
        "weight": model.get_tensor(f"{prefix}.weight").float(),
        "bias": model.get_tensor(f"{prefix}.bias").float(),
    }


def _noise_to_t(noise_level) -> int:
    if noise_level <= 0:
        return 0
    if noise_level >= 1:
        return 999
    target = math.log(noise_level / (1 - noise_level))
    start, end = 0.00085**0.5, 0.012**0.5
    image_fraction = 1.0
    previous = -math.inf
    for t in range(1000):
        image_fraction *= 1 - (start + (t / 999) * (end - start)) ** 2
        log_ratio = math.log((1 - image_fraction) / image_fraction)
        if log_ratio >= target:
            return t if log_ratio - target <= target - previous else t - 1
        previous = log_ratio
    return 999


def _timestep_sinusoids(t: int, width: int) -> torch.Tensor:
    """Converts a timestep into a vector of sinusoids, so that the UNet sees
    nearby timesteps as similar vectors. Half of the vector is cosines and half
    is sines, of t multiplied by frequencies spaced geometrically from 1 down to
    1 / 10000."""

    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t * frequencies
    return torch.cat([torch.cos(angles), torch.sin(angles)])


def _time_embed(x: torch.Tensor, time_embed: list[dict]) -> torch.Tensor:
    """Runs the timestep sinusoids through the time embedding layers, which
    turn the fixed sinusoids into a learned representation of the timestep that
    every residual block in the UNet is given. Each linear layer after the first
    is preceded by an activation."""

    for index, layer in enumerate(time_embed):
        if index > 0:
            x = silu(x)
        x = linear(layer["weight"], layer["bias"], x)
    return x


def _combine_chunks(conditioning: torch.Tensor) -> torch.Tensor:
    """Joins the chunks of a conditioning end to end into a single sequence of
    token vectors. CLIP encodes each chunk of a long prompt separately, but the
    UNet attends over every token vector at once, so a conditioning of shape
    (chunks, tokens, width) becomes (chunks * tokens, width)."""

    return conditioning.reshape(-1, conditioning.shape[-1])
