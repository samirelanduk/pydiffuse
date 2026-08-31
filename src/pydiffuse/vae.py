import safetensors
import torch
from PIL import Image

from .layers import convolution

DOWNSCALE_RATIO = 8
CONV_IN_PADDING = 1


def encode(image: Image.Image, model: safetensors.safe_open):
    model_tensors = _get_tensors(model)
    x = _image_to_tensor(image)
    x = convolution(
        model_tensors["conv_in"]["weight"],
        model_tensors["conv_in"]["bias"],
        x,
        padding=CONV_IN_PADDING,
    )
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


def _get_tensors(model: safetensors.safe_open):
    return {
        "conv_in": {
            "weight": model.get_tensor(
                "first_stage_model.encoder.conv_in.weight"
            ).float(),
            "bias": model.get_tensor("first_stage_model.encoder.conv_in.bias").float(),
        }
    }


def _encode_down():
    pass


def _encode_mid():
    pass


def _decode_mid():
    pass


def _decode_up():
    pass
