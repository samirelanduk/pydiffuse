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
from pydiffuse.noise import exponential_schedule, karras_schedule, noise_tensor
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


def check_parent(ctx, param, value):
    """Rejects an output file whose containing directory doesn't exist."""

    parent = Path(value).parent
    if not parent.exists():
        raise click.BadParameter(f"Directory '{parent}' does not exist.")
    return value


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


if __name__ == "__main__":
    cli()
