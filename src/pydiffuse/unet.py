import math
import re

import safetensors
import torch

from .layers import convolution, gelu, group_norm, layer_norm, linear, silu

MODEL_PREFIX = "model.diffusion_model"
INPUT_BLOCKS_PREFIX = f"{MODEL_PREFIX}.input_blocks"
MIDDLE_BLOCK_PREFIX = f"{MODEL_PREFIX}.middle_block"
OUTPUT_BLOCKS_PREFIX = f"{MODEL_PREFIX}.output_blocks"
RESNET_LAYERS = (
    "in_layers.0",
    "in_layers.2",
    "emb_layers.1",
    "out_layers.0",
    "out_layers.3",
    "skip_connection",
)
TRANSFORMER_LAYERS = ("norm", "proj_in", "proj_out")
TRANSFORMER_BLOCK_LAYERS = ("norm1", "norm2", "norm3", "ff.net.0.proj", "ff.net.2")
ATTENTION_LAYERS = ("to_q", "to_k", "to_v", "to_out.0")
OUT_LAYERS = ("0", "2")

CONV_PADDING = 1
DOWNSAMPLE_PADDING = 1
DOWNSAMPLE_STRIDE = 2
UPSAMPLE_SCALE = 2
NORM_GROUPS = 32
ATTENTION_HEADS = 8


def unet(
    latent: torch.Tensor,
    noise_level: float,
    conditioning: torch.Tensor,
    model: safetensors.safe_open,
) -> torch.Tensor:
    model_tensors = _get_unet_tensors(model)
    time_embedding = _noise_level_to_embedding(noise_level, model_tensors)
    conditioning = _combine_chunks(conditioning)
    x, skips = _input_blocks(
        latent, model_tensors["input_blocks"], time_embedding, conditioning
    )
    x = _block(x, model_tensors["middle_block"], time_embedding, conditioning)
    x = _output_blocks(
        x, skips, model_tensors["output_blocks"], time_embedding, conditioning
    )
    return _out_layers(x, model_tensors["out"])


def _get_unet_tensors(model: safetensors.safe_open) -> dict:
    """Finds the tensors used in the UNet."""

    return {
        "time_embed": [
            _get_layer(model, f"{MODEL_PREFIX}.time_embed.{index}")
            for index in _get_numbers(model, f"{MODEL_PREFIX}.time_embed")
        ],
        "input_blocks": [
            _get_block_tensors(model, f"{INPUT_BLOCKS_PREFIX}.{index}")
            for index in _get_numbers(model, INPUT_BLOCKS_PREFIX)
        ],
        "middle_block": _get_block_tensors(model, MIDDLE_BLOCK_PREFIX),
        "output_blocks": [
            _get_block_tensors(model, f"{OUTPUT_BLOCKS_PREFIX}.{index}")
            for index in _get_numbers(model, OUTPUT_BLOCKS_PREFIX)
        ],
        "out": _get_layers(model, f"{MODEL_PREFIX}.out", OUT_LAYERS),
    }


def _get_layer(model: safetensors.safe_open, prefix: str) -> dict | None:
    """Gets the weight and bias of a single layer, or None if the model has no
    such layer. Some layers have a weight but no bias, in which case the bias is
    None."""

    keys = model.keys()
    if f"{prefix}.weight" not in keys:
        return None
    return {
        "weight": model.get_tensor(f"{prefix}.weight").float(),
        "bias": (
            model.get_tensor(f"{prefix}.bias").float()
            if f"{prefix}.bias" in keys
            else None
        ),
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


def _get_block_tensors(
    model: safetensors.safe_open, prefix: str
) -> list[tuple[str, dict]]:
    """Finds the tensors of a single block, such as one of the input blocks, the
    middle block, or one of the output blocks. A block is a numbered list of
    parts which are run in order, and each part is a convolution, a residual
    block, a transformer, a downsampling convolution or an upsampling
    convolution. Which one a part is depends on the names of its tensors, not on
    its position in the block, and each part is returned as its type and its
    tensors."""

    keys = model.keys()
    parts = []
    for index in _get_numbers(model, prefix):
        part_prefix = f"{prefix}.{index}"
        if f"{part_prefix}.weight" in keys:
            parts.append(("conv", _get_layer(model, part_prefix)))
        elif f"{part_prefix}.op.weight" in keys:
            parts.append(("downsample", _get_layer(model, f"{part_prefix}.op")))
        elif f"{part_prefix}.conv.weight" in keys:
            parts.append(("upsample", _get_layer(model, f"{part_prefix}.conv")))
        elif f"{part_prefix}.in_layers.2.weight" in keys:
            parts.append(("resnet", _get_layers(model, part_prefix, RESNET_LAYERS)))
        elif f"{part_prefix}.proj_in.weight" in keys:
            parts.append(("transformer", _get_transformer_tensors(model, part_prefix)))
        else:
            raise ValueError(f"Unrecognised UNet block part: {part_prefix}")
    return parts


def _get_layers(
    model: safetensors.safe_open, prefix: str, names: tuple[str, ...]
) -> dict:
    """Gets the weight and bias of each of the given layers under a prefix."""

    return {name: _get_layer(model, f"{prefix}.{name}") for name in names}


def _get_transformer_tensors(model: safetensors.safe_open, prefix: str) -> dict:
    """Finds the tensors of a transformer, including every transformer block
    inside it."""

    tensors = _get_layers(model, prefix, TRANSFORMER_LAYERS)
    tensors["transformer_blocks"] = []
    for index in _get_numbers(model, f"{prefix}.transformer_blocks"):
        block_prefix = f"{prefix}.transformer_blocks.{index}"
        tensors["transformer_blocks"].append(
            {
                **_get_layers(model, block_prefix, TRANSFORMER_BLOCK_LAYERS),
                "attn1": _get_layers(model, f"{block_prefix}.attn1", ATTENTION_LAYERS),
                "attn2": _get_layers(model, f"{block_prefix}.attn2", ATTENTION_LAYERS),
            }
        )
    return tensors


def _noise_level_to_embedding(noise_level: float, model_tensors: dict) -> torch.Tensor:
    t = _noise_to_t(noise_level)
    sinusoid_width = model_tensors["time_embed"][0]["weight"].shape[1]
    sinusoids = _timestep_sinusoids(t, sinusoid_width)
    return _time_embed(sinusoids, model_tensors["time_embed"])


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


def _input_blocks(
    x: torch.Tensor,
    input_blocks: list[list[tuple[str, dict]]],
    time_embedding: torch.Tensor,
    conditioning: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Runs the latent through the input blocks - the downsampling half of the
    UNet. The first block convolves the latent up to many more channels, and the
    rest are residual blocks, transformers and downsampling convolutions, so the
    tensor gets deeper in channels and smaller in height and width as it goes.

    The output of every block is kept, as the output blocks will be given them
    later to recover the detail that downsampling loses."""

    skips = []
    for block in input_blocks:
        x = _block(x, block, time_embedding, conditioning)
        skips.append(x)
    return x, skips


def _block(
    x: torch.Tensor,
    block: list[tuple[str, dict]],
    time_embedding: torch.Tensor,
    conditioning: torch.Tensor,
    upsample_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Runs the tensor through each part of a single block in order. Residual
    blocks are given the time embedding, transformers are given the
    conditioning, and convolutions need neither. An upsampling convolution
    grows the tensor to the upsample size if one is given, and doubles its
    height and width otherwise."""

    for part_type, tensors in block:
        if part_type == "conv":
            x = convolution(tensors["weight"], tensors["bias"], x, padding=CONV_PADDING)
        elif part_type == "resnet":
            x = _resnet_block(x, tensors, time_embedding)
        elif part_type == "transformer":
            x = _transformer(x, tensors, conditioning)
        elif part_type == "downsample":
            x = convolution(
                tensors["weight"],
                tensors["bias"],
                x,
                padding=DOWNSAMPLE_PADDING,
                stride=DOWNSAMPLE_STRIDE,
            )
        elif part_type == "upsample":
            x = _upsample(x, tensors, upsample_size)
    return x


def _output_blocks(
    x: torch.Tensor,
    skips: list[torch.Tensor],
    output_blocks: list[list[tuple[str, dict]]],
    time_embedding: torch.Tensor,
    conditioning: torch.Tensor,
) -> torch.Tensor:
    """Runs the tensor through the output blocks - the upsampling half of the
    UNet. Before each block, the most recently kept output of the input blocks
    is joined onto the tensor as extra channels, so the output blocks work
    through the input blocks' outputs in reverse. The tensor gets shallower in
    channels and larger in height and width as it goes.

    When a block upsamples, it grows the tensor to the height and width of the
    next output it will be joined with. This is usually exactly double, but not
    when downsampling had to round an odd height or width up."""

    for block in output_blocks:
        x = torch.cat([x, skips.pop()])
        upsample_size = (skips[-1].shape[1], skips[-1].shape[2]) if skips else None
        x = _block(x, block, time_embedding, conditioning, upsample_size)
    return x


def _out_layers(x: torch.Tensor, out: dict) -> torch.Tensor:
    """Runs the tensor through the end of the UNet, normalising and activating
    it a final time before a convolution reduces its channels to those of the
    latent. The result is the UNet's prediction of the noise in the latent."""

    x = group_norm(out["0"]["weight"], out["0"]["bias"], x, groups=NORM_GROUPS)
    x = silu(x)
    x = convolution(out["2"]["weight"], out["2"]["bias"], x, padding=CONV_PADDING)
    return x


def _resnet_block(
    x: torch.Tensor, block: dict, time_embedding: torch.Tensor
) -> torch.Tensor:
    """Applies a residual block to the tensor. The tensor is normalised,
    activated and convolved twice, and the result is added back onto the
    original tensor.

    Between the two convolutions, the time embedding is projected down to one
    value per channel and added on, so that every position in a channel is
    shifted by the same amount depending on the timestep.

    If the block changes the number of channels, the original tensor is first
    passed through a 1x1 convolution so that the two can be added together."""

    h = group_norm(
        block["in_layers.0"]["weight"],
        block["in_layers.0"]["bias"],
        x,
        groups=NORM_GROUPS,
    )
    h = silu(h)
    h = convolution(
        block["in_layers.2"]["weight"],
        block["in_layers.2"]["bias"],
        h,
        padding=CONV_PADDING,
    )
    embedding = silu(time_embedding)
    embedding = linear(
        block["emb_layers.1"]["weight"], block["emb_layers.1"]["bias"], embedding
    )
    h = h + embedding[:, None, None]
    h = group_norm(
        block["out_layers.0"]["weight"],
        block["out_layers.0"]["bias"],
        h,
        groups=NORM_GROUPS,
    )
    h = silu(h)
    h = convolution(
        block["out_layers.3"]["weight"],
        block["out_layers.3"]["bias"],
        h,
        padding=CONV_PADDING,
    )
    if block["skip_connection"] is not None:
        x = convolution(
            block["skip_connection"]["weight"], block["skip_connection"]["bias"], x
        )
    return x + h


def _transformer(
    x: torch.Tensor, block: dict, conditioning: torch.Tensor
) -> torch.Tensor:
    """Applies a transformer to the tensor. The tensor is normalised and
    projected, then flattened so that every position in the image becomes a
    vector in a sequence. This sequence is run through each transformer block,
    unflattened back into an image, projected again, and added back onto the
    original tensor."""

    h = group_norm(
        block["norm"]["weight"], block["norm"]["bias"], x, groups=NORM_GROUPS
    )
    h = convolution(block["proj_in"]["weight"], block["proj_in"]["bias"], h)
    channels, height, width = h.shape
    h = h.view(channels, height * width).transpose(0, 1)
    for transformer_block in block["transformer_blocks"]:
        h = _transformer_block(h, transformer_block, conditioning)
    h = h.transpose(0, 1).contiguous().view(channels, height, width)
    h = convolution(block["proj_out"]["weight"], block["proj_out"]["bias"], h)
    return x + h


def _transformer_block(
    x: torch.Tensor, block: dict, conditioning: torch.Tensor
) -> torch.Tensor:
    """Applies a transformer block to a sequence of image position vectors. The
    positions first attend to each other, then to the conditioning's token
    vectors - which is how the prompt influences the image - and are then each
    run through a feed forward network. Each of the three stages is normalised
    first and added back onto the sequence."""

    h = layer_norm(block["norm1"]["weight"], block["norm1"]["bias"], x)
    x = x + _attention(h, h, block["attn1"])
    h = layer_norm(block["norm2"]["weight"], block["norm2"]["bias"], x)
    x = x + _attention(h, conditioning, block["attn2"])
    h = layer_norm(block["norm3"]["weight"], block["norm3"]["bias"], x)
    x = x + _feed_forward(h, block)
    return x


def _attention(x: torch.Tensor, targets: torch.Tensor, block: dict) -> torch.Tensor:
    """Applies multi-head attention, in which every vector in x is scored
    against every vector in the targets, and is then replaced by a weighted
    average of them. When the targets are x itself this is self-attention, and
    when they are the conditioning it is cross-attention.

    The vectors are split into several heads which each do this independently,
    and the heads' results are joined back together."""

    Q = linear(block["to_q"]["weight"], block["to_q"]["bias"], x)
    K = linear(block["to_k"]["weight"], block["to_k"]["bias"], targets)
    V = linear(block["to_v"]["weight"], block["to_v"]["bias"], targets)
    Q = Q.unflatten(1, (ATTENTION_HEADS, -1)).transpose(0, 1)
    K = K.unflatten(1, (ATTENTION_HEADS, -1)).transpose(0, 1)
    V = V.unflatten(1, (ATTENTION_HEADS, -1)).transpose(0, 1)
    scores = Q @ K.transpose(1, 2) / (Q.shape[-1] ** 0.5)
    attn_output = torch.softmax(scores, dim=-1) @ V
    attn_output = attn_output.transpose(0, 1).flatten(1)
    return linear(block["to_out.0"]["weight"], block["to_out.0"]["bias"], attn_output)


def _feed_forward(x: torch.Tensor, block: dict) -> torch.Tensor:
    """Applies a transformer block's feed forward network to every vector in the
    sequence. The first linear layer produces two halves - one is activated and
    used as a gate that scales the other - and the second linear layer reduces
    the result back to the original width."""

    h = linear(block["ff.net.0.proj"]["weight"], block["ff.net.0.proj"]["bias"], x)
    h, gate = h.chunk(2, dim=-1)
    h = h * gelu(gate)
    return linear(block["ff.net.2"]["weight"], block["ff.net.2"]["bias"], h)


def _upsample(
    x: torch.Tensor, conv: dict, size: tuple[int, int] | None = None
) -> torch.Tensor:
    """Doubles the height and width of the tensor by repeating each value into a
    2x2 square of its own, then runs a convolution over the result to smooth out
    the blockiness that repeating produces.

    If a size is given, the repeated tensor is cropped from the bottom and right
    down to that height and width before the convolution."""

    x = x.repeat_interleave(UPSAMPLE_SCALE, dim=1)
    x = x.repeat_interleave(UPSAMPLE_SCALE, dim=2)
    if size is not None:
        x = x[:, : size[0], : size[1]]
    x = convolution(conv["weight"], conv["bias"], x, padding=CONV_PADDING)
    return x
