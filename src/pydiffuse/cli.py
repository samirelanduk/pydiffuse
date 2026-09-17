import json
from pathlib import Path

import click
import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPTokenizer

from pydiffuse.clip import embed as clip_embed
from pydiffuse.clip import encode as clip_encode
from pydiffuse.clip import tokenize as clip_tokenize
from pydiffuse.noise import (
    create_noise,
    exponential_schedule,
    karras_schedule,
    noise_tensor,
)
from pydiffuse.sample import sample_euler, sample_heun
from pydiffuse.unet import unet as unet_predict
from pydiffuse.vae import decode as vae_decode
from pydiffuse.vae import encode as vae_encode


@click.group()
def cli():
    pass


@cli.group()
def clip():
    """CLIP text encoding commands."""


@cli.group()
def vae():
    """VAE encode/decode commands."""


@cli.group()
def noise():
    """Noise commands."""


@cli.group()
def unet():
    """UNet commands."""


@cli.group()
def sample():
    """Sampling commands."""


def check_parent(ctx, param, value):
    """Rejects an output file whose containing directory doesn't exist."""

    parent = Path(value).parent
    if not parent.exists():
        raise click.BadParameter(f"Directory '{parent}' does not exist.")
    return value


def check_schedule(ctx, param, value):
    """Reads a noise schedule file, rejecting it unless it has at least two
    noise levels which are each at least 0 and below 1."""

    with open(value) as f:
        lines = f.read().splitlines()
    levels = []
    for line in lines:
        try:
            level = float(line)
        except ValueError:
            raise click.BadParameter(f"'{line}' is not a valid noise level.")
        if not 0 <= level < 1:
            raise click.BadParameter(f"{level} is not in the range 0<=x<1.")
        levels.append(level)
    if len(levels) < 2:
        raise click.BadParameter("A schedule must have at least two noise levels.")
    return levels


@clip.command()
@click.argument("text")
@click.option(
    "--tokens",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="tokens.json",
    help="Path to save the tokens JSON to.",
)
@click.option(
    "--mappings",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="mappings.json",
    help="Path to save the mappings JSON to.",
)
@click.option(
    "--tokenizer",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to a custom CLIP tokenizer.",
)
def tokenize(text, tokens, mappings, tokenizer):
    """Tokenizes the given text with a CLIP tokenizer."""

    clip_tokenizer = None
    if tokenizer:
        try:
            clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        except Exception as e:
            raise click.ClickException(f"{tokenizer} is not a CLIP tokenizer.") from e
    token_lists, mapping_lists = clip_tokenize(text, clip_tokenizer=clip_tokenizer)
    with open(tokens, "w") as f:
        json.dump(token_lists, f)
    with open(mappings, "w") as f:
        json.dump(mapping_lists, f)


@clip.command("embed")
@click.argument("tokens", type=click.Path(exists=True, dir_okay=False))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--embedding",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="embedding.pt",
    help="Path to save the embedding to.",
)
def embed(tokens, model, embedding):
    """Embeds tokens using CLIP embedding weights from a model."""

    with open(tokens) as f:
        token_lists = json.load(f)
    with safe_open(model, framework="pt", device="cpu") as tensors:
        embedding_tensor = clip_embed(token_lists, tensors)
    torch.save(embedding_tensor, embedding)


@clip.command("encode")
@click.argument("embedding", type=click.Path(exists=True, dir_okay=False))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--conditioning",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="conditioning.pt",
    help="Path to save the conditioning to.",
)
def encode(embedding, model, conditioning):
    """Encodes embeddings using CLIP encoder weights from a model."""

    embedding_tensor = torch.load(embedding)
    with safe_open(model, framework="pt", device="cpu") as tensors:
        conditioning_tensor = clip_encode(embedding_tensor, tensors)
    torch.save(conditioning_tensor, conditioning)


@vae.command("encode")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--latent",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="latent.pt",
    help="Path to save the latent to.",
)
def encode_image(image, model, latent):
    """Encodes an image into a latent using a VAE."""

    with safe_open(model, framework="pt", device="cpu") as tensors:
        latent_tensor = vae_encode(Image.open(image), tensors)
    torch.save(latent_tensor, latent)


@vae.command("decode")
@click.argument("latent", type=click.Path(exists=True, dir_okay=False))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--image",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="image.jpg",
    help="Path to save the image to.",
)
def decode_latent(latent, model, image):
    """Decodes a latent into an image using a VAE."""

    latent_tensor = torch.load(latent)
    with safe_open(model, framework="pt", device="cpu") as tensors:
        image_obj = vae_decode(latent_tensor, tensors)
    image_obj.save(image)


@noise.command("create")
@click.argument("width", type=click.IntRange(1))
@click.argument("height", type=click.IntRange(1))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="latent.pt",
    help="Path to save the latent to.",
)
def create_latent(width, height, model, output):
    """Creates a latent of pure noise for an image of the given size."""

    with safe_open(model, framework="pt", device="cpu") as tensors:
        latent_tensor = create_noise(width, height, tensors)
    torch.save(latent_tensor, output)


@noise.command("apply")
@click.argument("tensor", type=click.Path(exists=True, dir_okay=False))
@click.argument("noise_level", type=click.FloatRange(0, 1))
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="noised.pt",
    help="Path to save the noised tensor to.",
)
def apply_noise(tensor, noise_level, output):
    """Adds noise to a tensor at the given noise level."""

    noised_tensor = noise_tensor(torch.load(tensor), noise_level)
    torch.save(noised_tensor, output)


@noise.command("schedule")
@click.argument("steps", type=click.IntRange(1))
@click.option(
    "--algorithm",
    type=click.Choice(["karras", "exponential"]),
    default="karras",
    help="The algorithm to generate the schedule with.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="schedule.txt",
    help="Path to save the schedule to.",
)
def noise_schedule(steps, algorithm, output):
    """Generates a noise schedule of the given number of steps."""

    schedules = {"karras": karras_schedule, "exponential": exponential_schedule}
    levels = schedules[algorithm](steps)
    with open(output, "w") as f:
        f.write("\n".join(str(round(level, 8)) for level in levels) + "\n")


@unet.command("predict")
@click.argument("latent", type=click.Path(exists=True, dir_okay=False))
@click.argument("noise_level", type=click.FloatRange(0, 1))
@click.argument("conditioning", type=click.Path(exists=True, dir_okay=False))
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--noise",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="noise.pt",
    help="Path to save the noise prediction to.",
)
def predict_noise(latent, noise_level, conditioning, model, noise):
    """Predicts the noise in a noised latent using a UNet."""

    latent_tensor = torch.load(latent)
    conditioning_tensor = torch.load(conditioning)
    with safe_open(model, framework="pt", device="cpu") as tensors:
        prediction_tensor = unet_predict(
            latent_tensor, noise_level, conditioning_tensor, tensors
        )
    torch.save(prediction_tensor, noise)


@sample.command("denoise")
@click.argument("latent", type=click.Path(exists=True, dir_okay=False))
@click.argument("positive", type=click.Path(exists=True, dir_okay=False))
@click.argument("negative", type=click.Path(exists=True, dir_okay=False))
@click.argument(
    "schedule", type=click.Path(exists=True, dir_okay=False), callback=check_schedule
)
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--cfg",
    type=float,
    default=1.0,
    help=(
        "How strongly to steer towards the positive conditioning. At 1, the "
        "negative conditioning has no effect."
    ),
)
@click.option(
    "--algorithm",
    type=click.Choice(["euler", "heun"]),
    default="euler",
    help="The algorithm to sample with.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    callback=check_parent,
    default="denoised.pt",
    help="Path to save the denoised latent to.",
)
def denoise_latent(latent, positive, negative, schedule, model, cfg, algorithm, output):
    """Denoises a latent by stepping through a noise schedule using a UNet."""

    samplers = {"euler": sample_euler, "heun": sample_heun}
    latent_tensor = torch.load(latent)
    positive_tensor = torch.load(positive)
    negative_tensor = torch.load(negative)
    with safe_open(model, framework="pt", device="cpu") as tensors:
        denoised_tensor = samplers[algorithm](
            positive_tensor, negative_tensor, latent_tensor, tensors, schedule, cfg
        )
    torch.save(denoised_tensor, output)


if __name__ == "__main__":
    cli()
