import re

import safetensors
import torch
from PIL import Image

from .layers import convolution, group_norm, silu

ENCODER_PREFIX = "first_stage_model.encoder"
DOWNSCALE_RATIO = 8
CONV_IN_PADDING = 1
CONV_PADDING = 1
DOWNSAMPLE_PADDING = 1
DOWNSAMPLE_STRIDE = 2
NORM_EPSILON = 1e-6
NORM_GROUPS = 32


def encode(image: Image.Image, model: safetensors.safe_open):
    model_tensors = _get_tensors(model)
    x = _image_to_tensor(image)
    x = convolution(
        model_tensors["conv_in"]["weight"],
        model_tensors["conv_in"]["bias"],
        x,
        padding=CONV_IN_PADDING,
    )
    x = _encode_down(x, model_tensors["down"])
    return x


def decode():
    pass


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = torch.tensor(rgb.getdata(), dtype=torch.float32).reshape(height, width, 3)
    pixels = pixels / 255.0 * 2.0 - 1.0
    pixels = pixels.permute(2, 0, 1).unsqueeze(0)
    for dimension, size in ((2, height), (3, width)):
        cropped = size - (size % DOWNSCALE_RATIO)
        if cropped != size:
            offset = (size % DOWNSCALE_RATIO) // 2
            pixels = pixels.narrow(dimension, offset, cropped)
    return pixels


def _get_tensors(model: safetensors.safe_open) -> dict:
    """Finds the tensors used in VAE encode's downsampling stages."""

    down = []
    for level in _get_down_level_numbers(model):
        prefix = f"{ENCODER_PREFIX}.down.{level}"
        blocks = [
            {
                name: _get_layer_tensors(model, f"{prefix}.block.{index}.{name}")
                for name in ("norm1", "conv1", "norm2", "conv2", "nin_shortcut")
            }
            for index in _get_down_block_numbers(model, level)
        ]
        downsample = _get_layer_tensors(model, f"{prefix}.downsample.conv")
        down.append({"block": blocks, "downsample": downsample})
    return {
        "conv_in": _get_layer_tensors(model, f"{ENCODER_PREFIX}.conv_in"),
        "down": down,
    }


def _get_layer_tensors(model: safetensors.safe_open, prefix: str) -> dict | None:
    """Gets the weight and bias of a single layer, or None if the model has no
    such layer."""

    keys = model.keys()
    if f"{prefix}.weight" not in keys:
        return None
    return {
        "weight": model.get_tensor(f"{prefix}.weight").float(),
        "bias": model.get_tensor(f"{prefix}.bias").float(),
    }


def _get_down_level_numbers(model: safetensors.safe_open) -> list[int]:
    """Gets a list of encoder downsampling level numbers present in the model."""

    level_numbers = set()
    keys = model.keys()
    for key in keys:
        level_number_match = re.search(rf"^{ENCODER_PREFIX}\.down\.(\d+)\.", key)
        if not level_number_match:
            continue
        level_numbers.add(int(level_number_match.group(1)))
    return sorted(level_numbers)


def _get_down_block_numbers(model: safetensors.safe_open, level: int) -> list[int]:
    """Gets a list of residual block numbers present in a downsampling level."""

    block_numbers = set()
    keys = model.keys()
    for key in keys:
        prefix = rf"^{ENCODER_PREFIX}\.down\.{level}\.block\.(\d+)\."
        block_number_match = re.search(prefix, key)
        if not block_number_match:
            continue
        block_numbers.add(int(block_number_match.group(1)))
    return sorted(block_numbers)


def _encode_down(x: torch.Tensor, down: list[dict]) -> torch.Tensor:
    """Runs the tensor through the downsampling half of the encoder. Each level
    is a run of residual blocks followed (usually) by a downsampling
    convolution, so the tensor gets deeper in channels and smaller in height and
    width as it goes."""

    for level, tensors in enumerate(down):
        for block in tensors["block"]:
            x = _resnet_block(x, block)
        if tensors["downsample"] is not None:
            x = convolution(
                tensors["downsample"]["weight"],
                tensors["downsample"]["bias"],
                x,
                padding=DOWNSAMPLE_PADDING,
                stride=DOWNSAMPLE_STRIDE,
                pad_at_end=True,  # assumes a 3x3 kernel
            )
    return x


def _resnet_block(x: torch.Tensor, block: dict) -> torch.Tensor:
    """Applies a residual block to the tensor. The tensor is normalised,
    activated and convolved twice, and the result is added back onto the
    original tensor.

    If the block changes the number of channels, the original
    tensor is first passed through a 1x1 convolution so that the two can be
    added together.

    The final tensor is the same shape as the input."""

    h = group_norm(
        block["norm1"]["weight"],
        block["norm1"]["bias"],
        x,
        groups=NORM_GROUPS,
    )
    h = silu(h)
    h = convolution(
        block["conv1"]["weight"], block["conv1"]["bias"], h, padding=CONV_PADDING
    )
    h = group_norm(
        block["norm2"]["weight"],
        block["norm2"]["bias"],
        h,
        groups=NORM_GROUPS,
    )
    h = silu(h)
    h = convolution(
        block["conv2"]["weight"], block["conv2"]["bias"], h, padding=CONV_PADDING
    )
    if block["nin_shortcut"] is not None:
        x = convolution(
            block["nin_shortcut"]["weight"], block["nin_shortcut"]["bias"], x
        )
    return x + h


def _encode_mid():
    pass


def _decode_mid():
    pass


def _decode_up():
    pass
