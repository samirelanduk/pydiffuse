import math

import torch


def linear(
    weight: torch.Tensor, bias: torch.Tensor | None, input: torch.Tensor
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data. The weight must be
    a 2D tensor of dimensions (output_dim, input_dim), and the bias must be a 1D
    tensor of dimensions (output_dim), or None if the layer has no bias."""

    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None, device="cpu"
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)


def group_norm(
    weight: torch.Tensor,
    bias: torch.Tensor,
    input: torch.Tensor,
    groups: int | None = None,
) -> torch.Tensor:
    """Applies a group normalization to the incoming data. The entries in
    dimension 0 are combined into groups (by default they all form a single
    group), and each group is adjusted so that its mean is 0 and its variance
    is 1, while preserving the relative gaps between the values. The values
    are then scaled by the weight and shifted by the bias, one of each per
    entry in dimension 0, applied to every value in that entry. Any dimensions
    after the first are carried along with the entry containing them.

    The weight and bias must be 1D tensors of dimensions matching dimension 0.
    The result is a tensor of the same shape as the input."""

    groups = 1 if groups is None else groups
    layer = torch.nn.GroupNorm(
        num_groups=groups, num_channels=input.shape[0], device="cpu"
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input.unsqueeze(0)).squeeze(0)


def layer_norm(
    weight: torch.Tensor, bias: torch.Tensor, input: torch.Tensor
) -> torch.Tensor:
    """Applies a layer normalization to the incoming data. Each vector in the
    final dimension is adjusted so that its mean is 0 and its variance is 1,
    while preserving the relative gaps between the values. It then scales the
    values by the weight and adds the bias.

    The weight and bias must be 1D tensors of dimensions (num_features). The
    result is a tensor of the same shape as the input."""

    layer = torch.nn.LayerNorm(weight.shape[0], device="cpu")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)


def convolution(
    weight: torch.Tensor,
    bias: torch.Tensor,
    input: torch.Tensor,
    padding: int = 0,
    stride: int = 1,
    pad_at_end: bool = False,
) -> torch.Tensor:
    """Applies a 2D convolution to the incoming data, which should be a 3D
    tensor of shape [channels, height, width]. The weight must be a 4D tensor
    of shape [output_channels, input_channels, kernel_height, kernel_width].
    The bias must be a 1D tensor of shape [output_channels]. The result is a 3D
    tensor of shape [output_channels, new_height, new_width].

    The kernel is run over every position in the height/width matrix, and
    outputs a single value for that position using the weights and biases.

    Padding adds a border of zeros of the given width to every edge of the
    input before the kernel is run. If pad_at_end is True the zeros are added
    only to the right and bottom edges instead of to all four."""

    if pad_at_end:
        input = torch.nn.functional.pad(input, (0, padding, 0, padding))
        padding = 0
    layer = torch.nn.Conv2d(
        in_channels=weight.shape[1],
        out_channels=weight.shape[0],
        kernel_size=(weight.shape[2], weight.shape[3]),
        padding=padding,
        stride=stride,
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)


def silu(input: torch.Tensor) -> torch.Tensor:
    """Applies the SiLU (or swish) activation function to every value in the
    incoming data, which is the value multiplied by its own sigmoid - that is,
    x * (1 / (1 + e^-x)). Large positive values are passed through more or less
    unchanged, and large negative values are squashed towards zero.

    The result is a tensor of the same shape as the input."""

    return input * torch.sigmoid(input)


def gelu(input: torch.Tensor) -> torch.Tensor:
    """Applies the GELU activation function to every value in the incoming
    data, which is the value multiplied by the chance that a value drawn from a
    standard normal distribution is smaller than it. Like SiLU, large positive
    values are passed through more or less unchanged, and large negative values
    are squashed towards zero.

    The result is a tensor of the same shape as the input."""

    return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
