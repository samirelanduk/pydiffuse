import re

import safetensors
import torch
from PIL import Image

from .layers import convolution, group_norm, silu

MODEL_PREFIX = "first_stage_model"
ENCODER_PREFIX = f"{MODEL_PREFIX}.encoder"
DECODER_PREFIX = f"{MODEL_PREFIX}.decoder"
RESNET_LAYERS = ("norm1", "conv1", "norm2", "conv2", "nin_shortcut")
ATTENTION_LAYERS = ("norm", "q", "k", "v", "proj_out")

CONV_IN_PADDING = 1
CONV_PADDING = 1
DOWNSAMPLE_PADDING = 1
DOWNSAMPLE_STRIDE = 2
UPSAMPLE_SCALE = 2
NORM_GROUPS = 32


def encode(image: Image.Image, model: safetensors.safe_open) -> torch.Tensor:
    """Encodes an image into latent space using a VAE model."""

    model_tensors = _get_encode_tensors(model)
    downscale_ratio = _get_downscale_ratio(model_tensors["down"])
    x = _image_to_tensor(image, downscale_ratio)
    x = convolution(
        model_tensors["conv_in"]["weight"],
        model_tensors["conv_in"]["bias"],
        x,
        padding=CONV_IN_PADDING,
    )
    x = _encode_down(x, model_tensors["down"])
    x = _mid_blocks(x, model_tensors["mid"])
    x = _out_layers(x, model_tensors["norm_out"], model_tensors["conv_out"])
    x = convolution(
        model_tensors["quant_conv"]["weight"],
        model_tensors["quant_conv"]["bias"],
        x,
    )
    x = x.narrow(1, 0, x.shape[1] // 2)
    return x


def decode(latent: torch.Tensor, model: safetensors.safe_open) -> Image.Image:
    """Decodes a latent tensor into an image using a VAE model."""

    model_tensors = _get_decode_tensors(model)
    x = convolution(
        model_tensors["post_quant_conv"]["weight"],
        model_tensors["post_quant_conv"]["bias"],
        latent,
    )
    x = convolution(
        model_tensors["conv_in"]["weight"],
        model_tensors["conv_in"]["bias"],
        x,
        padding=CONV_IN_PADDING,
    )
    x = _mid_blocks(x, model_tensors["mid"])
    x = _decode_up(x, model_tensors["up"])
    x = _out_layers(x, model_tensors["norm_out"], model_tensors["conv_out"])
    return _tensor_to_image(x)


def _get_encode_tensors(model: safetensors.safe_open) -> dict:
    """Finds the tensors used in VAE encode."""

    down = []
    for level in _get_numbers(model, f"{ENCODER_PREFIX}.down"):
        prefix = f"{ENCODER_PREFIX}.down.{level}"
        blocks = [
            _get_layers(model, f"{prefix}.block.{index}", RESNET_LAYERS)
            for index in _get_numbers(model, f"{prefix}.block")
        ]
        downsample = _get_layer(model, f"{prefix}.downsample.conv")
        down.append({"block": blocks, "downsample": downsample})
    return {
        "conv_in": _get_layer(model, f"{ENCODER_PREFIX}.conv_in"),
        "down": down,
        "mid": _get_mid_tensors(model, ENCODER_PREFIX),
        "norm_out": _get_layer(model, f"{ENCODER_PREFIX}.norm_out"),
        "conv_out": _get_layer(model, f"{ENCODER_PREFIX}.conv_out"),
        "quant_conv": _get_layer(model, f"{MODEL_PREFIX}.quant_conv"),
    }


def _get_decode_tensors(model: safetensors.safe_open) -> dict:
    """Finds the tensors used in VAE decode."""

    up = []
    for level in _get_numbers(model, f"{DECODER_PREFIX}.up"):
        prefix = f"{DECODER_PREFIX}.up.{level}"
        blocks = [
            _get_layers(model, f"{prefix}.block.{index}", RESNET_LAYERS)
            for index in _get_numbers(model, f"{prefix}.block")
        ]
        upsample = _get_layer(model, f"{prefix}.upsample.conv")
        up.append({"block": blocks, "upsample": upsample})
    return {
        "post_quant_conv": _get_layer(model, f"{MODEL_PREFIX}.post_quant_conv"),
        "conv_in": _get_layer(model, f"{DECODER_PREFIX}.conv_in"),
        "mid": _get_mid_tensors(model, DECODER_PREFIX),
        "up": up,
        "norm_out": _get_layer(model, f"{DECODER_PREFIX}.norm_out"),
        "conv_out": _get_layer(model, f"{DECODER_PREFIX}.conv_out"),
    }


def _get_numbers(model: safetensors.safe_open, prefix: str) -> list[int]:
    """Gets a list of the numbers that the model's keys use directly under a
    prefix, such as the levels within a downsampling section, or the residual
    blocks within a level."""

    numbers = set()
    keys = model.keys()
    for key in keys:
        number_match = re.search(rf"^{prefix}\.(\d+)\.", key)
        if not number_match:
            continue
        numbers.add(int(number_match.group(1)))
    return sorted(numbers)


def _get_layers(
    model: safetensors.safe_open, prefix: str, names: tuple[str, ...]
) -> dict:
    """Gets the weight and bias of each of the given layers under a prefix."""

    return {name: _get_layer(model, f"{prefix}.{name}") for name in names}


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


def _get_mid_tensors(model: safetensors.safe_open, prefix: str) -> dict:
    """Finds the tensors of the residual, attention and residual blocks that sit
    in the middle of both the encoder and the decoder."""

    return {
        "block_1": _get_layers(model, f"{prefix}.mid.block_1", RESNET_LAYERS),
        "attn_1": _get_layers(model, f"{prefix}.mid.attn_1", ATTENTION_LAYERS),
        "block_2": _get_layers(model, f"{prefix}.mid.block_2", RESNET_LAYERS),
    }


def _get_downscale_ratio(down: list[dict]) -> int:
    """Works out how much smaller than the image the latent will be, which is
    the stride of every downsampling convolution multiplied together."""

    downsamples = [level for level in down if level["downsample"] is not None]
    return DOWNSAMPLE_STRIDE ** len(downsamples)


def _image_to_tensor(image: Image.Image, downscale_ratio: int) -> torch.Tensor:
    """Converts a PIL image into a tensor containing the same pixel information,
    but normalised to the range [-1, 1]."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.float32)
    pixels = pixels.reshape(height, width, 3)
    pixels = pixels / 255.0 * 2.0 - 1.0
    pixels = pixels.permute(2, 0, 1).unsqueeze(0)
    for dimension, size in ((2, height), (3, width)):
        cropped = size - (size % downscale_ratio)
        if cropped != size:
            offset = (size % downscale_ratio) // 2
            pixels = pixels.narrow(dimension, offset, cropped)
    return pixels


def _tensor_to_image(x: torch.Tensor) -> Image.Image:
    """Turns a decoded tensor back into an image, reversing the conversion that
    the encoder's input went through. Values outside the range the encoder was
    given are clamped, as the decoder is under no obligation to stay inside
    it."""

    pixels = x.squeeze(0).permute(1, 2, 0)
    pixels = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)
    pixels = (pixels * 255.0).round().to(torch.uint8)
    return Image.fromarray(pixels.contiguous().numpy(), mode="RGB")


def _encode_down(x: torch.Tensor, down: list[dict]) -> torch.Tensor:
    """Runs the tensor through the downsampling half of the encoder. Each level
    is a run of residual blocks followed (usually) by a downsampling
    convolution, so the tensor gets deeper in channels and smaller in height and
    width as it goes."""

    for tensors in down:
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


def _decode_up(x: torch.Tensor, up: list[dict]) -> torch.Tensor:
    """Runs the tensor through the upsampling half of the decoder. The levels
    are worked through in reverse, each one a run of residual blocks followed
    (usually) by an upsampling convolution, so the tensor gets shallower in
    channels and larger in height and width as it goes."""

    for tensors in reversed(up):
        for block in tensors["block"]:
            x = _resnet_block(x, block)
        if tensors["upsample"] is not None:
            x = _upsample(x, tensors["upsample"])
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


def _mid_blocks(x: torch.Tensor, mid: dict) -> torch.Tensor:
    """Runs the tensor through the middle of the encoder or decoder - a residual
    block, an attention block, and a second residual block. Nothing changes
    shape, as it is a refinement of the smallest representation."""

    x = _resnet_block(x, mid["block_1"])
    x = _attention_block(x, mid["attn_1"])
    x = _resnet_block(x, mid["block_2"])
    return x


def _attention_block(x: torch.Tensor, block: dict) -> torch.Tensor:
    """Applies an attention block to the tensor. Every position in the image is
    scored against every other position, and is then replaced by a weighted
    average of all of them, so that distant parts of the image can inform each
    other - something a convolution's small window cannot do. The result is
    added back onto the original tensor."""

    attn_x = group_norm(
        block["norm"]["weight"], block["norm"]["bias"], x, groups=NORM_GROUPS
    )
    Q = convolution(block["q"]["weight"], block["q"]["bias"], attn_x)
    K = convolution(block["k"]["weight"], block["k"]["bias"], attn_x)
    V = convolution(block["v"]["weight"], block["v"]["bias"], attn_x)
    batch, channels, height, width = Q.shape
    positions = height * width
    Q = Q.view(batch, channels, positions).transpose(1, 2)
    K = K.view(batch, channels, positions)
    V = V.view(batch, channels, positions).transpose(1, 2)
    scores = Q @ K / (Q.shape[-1] ** 0.5)
    attn_output = torch.softmax(scores, dim=-1) @ V
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(batch, channels, height, width)
    )
    attn_output = convolution(
        block["proj_out"]["weight"], block["proj_out"]["bias"], attn_output
    )
    return x + attn_output


def _out_layers(x: torch.Tensor, norm_out: dict, conv_out: dict) -> torch.Tensor:
    """Runs the tensor through the end of the encoder or decoder, normalising
    and activating it a final time before a convolution reduces its channels to
    however many the output needs."""

    x = group_norm(norm_out["weight"], norm_out["bias"], x, groups=NORM_GROUPS)
    x = silu(x)
    x = convolution(conv_out["weight"], conv_out["bias"], x, padding=CONV_PADDING)
    return x


def _upsample(x: torch.Tensor, upsample: dict) -> torch.Tensor:
    """Doubles the height and width of the tensor by repeating each value into a
    2x2 square of its own, then runs a convolution over the result to smooth out
    the blockiness that repeating produces."""

    x = x.repeat_interleave(UPSAMPLE_SCALE, dim=2)
    x = x.repeat_interleave(UPSAMPLE_SCALE, dim=3)
    x = convolution(upsample["weight"], upsample["bias"], x, padding=CONV_PADDING)
    return x
