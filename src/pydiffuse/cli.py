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


if __name__ == "__main__":
    cli()
