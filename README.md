# pydiffuse

[![PyPI](https://img.shields.io/pypi/v/pydiffuse?logo=pypi&logoColor=white)](https://pypi.org/project/pydiffuse/)
[![CI](https://img.shields.io/github/actions/workflow/status/samirelanduk/pydiffuse/ci.yml?branch=master&logo=github&label=CI)](https://github.com/samirelanduk/pydiffuse/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white)](https://github.com/samirelanduk/pydiffuse/blob/master/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://img.shields.io/badge/pyright-checked-2a6db2)](https://github.com/microsoft/pyright)

A python library for generating media with diffusion.

## Development

The project uses [uv](https://docs.astral.sh/uv/). With it installed:

```bash
git clone git@github.com:samirelanduk/pydiffuse.git
cd pydiffuse
uv sync
pre-commit install
```

`uv sync` creates the virtualenv in `.venv` and installs the project with its dev dependencies from the lockfile.
`pre-commit install` sets up the ruff check and format hooks, which CI also enforces.

Then, to run the checks:

```bash
uv run python -m unittest discover
uv run pyright
uv run pre-commit run --all-files
```

## CLI

The library exposes most of its functionality as a Command-Line Interface.

```bash
pydiffuse <group> <command>
```

Commands are namespaced by group - currently `clip`, `vae`, `noise`, `unet` and `sample`.

### CLIP

CLIP turns a text prompt into a set of vectors that represent its semantic meaning, which diffusion models use to condition image generation.
The process is split into three commands - tokenising, embedding, and encoding - each of which writes its output to a file that the next one reads.
See [docs/clip.md](docs/clip.md) for an explanation of what each stage does.

The embedding and encoding steps need model weights in [safetensors](https://github.com/huggingface/safetensors) format - typically a Stable Diffusion checkpoint, whose CLIP text encoder tensors are stored under `cond_stage_model`.

#### Tokenising

`clip tokenize` converts a prompt into the integer token IDs that CLIP uses.

```bash
pydiffuse clip tokenize "a photo of a lighthouse"
```

This writes two files.
`tokens.json` is a list of lists of token IDs, where each inner list is a chunk of exactly 77 tokens - prompts longer than that are split over as many chunks as they need, and the last chunk is padded:

```json
[[49406, 320, 1125, 539, 320, 13717, 49407, 49407, ...]]
```

`mappings.json` has the same shape, but pairs each token ID with the substring it came from, so you can see how the prompt was split up:

```json
[[["<|startoftext|>", 49406], ["a</w>", 320], ["photo</w>", 1125], ...]]
```

| Option | Default | Description |
| --- | --- | --- |
| `--tokens` | `tokens.json` | Path to save the tokens JSON to. |
| `--mappings` | `mappings.json` | Path to save the mappings JSON to. |
| `--tokenizer` | bundled tokenizer | Directory containing a custom CLIP tokenizer. |

The library bundles the standard CLIP tokenizer, so `--tokenizer` is only needed if your model was trained with a different vocabulary.

#### Embedding

`clip embed` looks up the vector for each token, and adds the vector for its position in the chunk.
The result represents each token in isolation, with no context from the rest of the prompt.

```bash
pydiffuse clip embed tokens.json model.safetensors
```

This writes `embedding.pt`, a `torch.save`-d tensor of shape `(chunks, 77, width)`, where `width` is the embedding width of the model (768 for Stable Diffusion 1.x).
The model must contain tensors whose keys end in `token_embedding.weight` and `position_embedding.weight`, and it must use the same vocabulary as the tokenizer that produced the tokens.

| Option | Default | Description |
| --- | --- | --- |
| `--embedding` | `embedding.pt` | Path to save the embedding to. |

#### Encoding

`clip encode` runs the embeddings through the CLIP transformer layers, so that each vector is adjusted by the tokens before it.
This is the conditioning that gets fed to a diffusion model.

```bash
pydiffuse clip encode embedding.pt model.safetensors
```

This writes `conditioning.pt`, a tensor of the same shape as the embedding.
The model must contain the attention, MLP and layer norm tensors for every encoder layer under `cond_stage_model`, plus the final layer norm - any number of layers is supported.

| Option | Default | Description |
| --- | --- | --- |
| `--conditioning` | `conditioning.pt` | Path to save the conditioning to. |

### VAE

A VAE compresses an image into a much smaller latent representation that diffusion happens in, and turns that latent back into an image afterwards.
See [docs/vae.md](docs/vae.md) for an explanation of how the two networks work.

Both commands need model weights in [safetensors](https://github.com/huggingface/safetensors) format, whose VAE tensors are stored under `first_stage_model`.

#### Encoding

`vae encode` compresses an image into a latent.

```bash
pydiffuse vae encode photo.jpg model.safetensors
```

This writes `latent.pt`, a `torch.save`-d tensor of shape `(channels, height / ratio, width / ratio)`, where `ratio` is the product of the strides of the model's downsampling convolutions.
The latent is multiplied by 0.18215, which brings it to the scale Stable Diffusion's UNet works at, and `vae decode` divides by it again.
Images whose dimensions aren't a multiple of that ratio are centre-cropped to the nearest one that is.

| Option | Default | Description |
| --- | --- | --- |
| `--latent` | `latent.pt` | Path to save the latent to. |

#### Decoding

`vae decode` expands a latent back into an image.

```bash
pydiffuse vae decode latent.pt model.safetensors
```

This writes `image.jpg`, at the latent's resolution multiplied by the same ratio.
The decoder can produce values outside the range it was trained on, and those are clamped rather than wrapped.

| Option | Default | Description |
| --- | --- | --- |
| `--image` | `image.jpg` | Path to save the image to. |

### Noise

Diffusion works by learning to remove noise, so both training and sampling need a way of adding a known amount of noise to a latent, and a schedule of how much noise to use at each step.
See [docs/noise.md](docs/noise.md) for how noise is added.

A noise level is a number between 0 and 1 giving the proportion of the result that is noise rather than signal, so 0 is a clean latent and 1 is pure noise.

#### Creating noise

`noise create` creates a latent of pure noise, the starting point for generating an image from nothing.

```bash
pydiffuse noise create 512 512 model.safetensors
```

This writes `latent.pt`, unit noise in the shape `vae encode` would give an image of that width and height.
The channels and downscale ratio are read from the model's VAE tensors, and sizes that aren't a multiple of the ratio are rounded down.
Fresh noise is drawn on every run.

| Option | Default | Description |
| --- | --- | --- |
| `--output` | `latent.pt` | Path to save the latent to. |

#### Applying noise

`noise apply` adds noise to a tensor at a single noise level.

```bash
pydiffuse noise apply latent.pt 0.5
```

This writes `noised.pt`, a tensor of the same shape as the input.
The tensor and the noise are each scaled by the square root of their share, so the result keeps a variance of 1 (assuming the input had a variance of 1 to start with).
Fresh noise is drawn on every run, so the same inputs give a different result each time.

| Option | Default | Description |
| --- | --- | --- |
| `--output` | `noised.pt` | Path to save the noised tensor to. |

#### Generating a schedule

`noise schedule` produces the sequence of noise levels a sampler steps through, from the noisiest level down to a clean image.

```bash
pydiffuse noise schedule 5
```

This writes `schedule.txt`, one noise level per line rounded to eight decimal places.
There is always one more line than there are steps, because the schedule ends at 0:

```
0.9953399
0.95834503
0.61880525
0.05791837
0.00085
0.0
```

Two algorithms are available, both of which space the levels out by their ratio of noise to signal rather than by the level itself, which puts more steps where the latent is nearly clean.
`karras` is the schedule from [Karras et al. (2022)](https://arxiv.org/abs/2206.00364), which spaces that ratio raised to the power 1/7.
`exponential` spaces the log of the ratio, so each step reduces it by the same factor.

| Option | Default | Description |
| --- | --- | --- |
| `--algorithm` | `karras` | The algorithm to generate the schedule with - `karras` or `exponential`. |
| `--output` | `schedule.txt` | Path to save the schedule to. |

### UNet

A UNet predicts the noise in a noised latent, guided by a conditioning from CLIP.
See [docs/unet.md](docs/unet.md) for how the network is structured and what its prediction means.

The command needs model weights in [safetensors](https://github.com/huggingface/safetensors) format, whose UNet tensors are stored under `model.diffusion_model`.

#### Predicting noise

`unet predict` runs the UNet once on a latent.

```bash
pydiffuse unet predict noised.pt 0.5 conditioning.pt model.safetensors
```

This writes `noise.pt`, the predicted noise, in the same shape as the latent.
The latent should already be noised to the given noise level, which is rounded to the nearest of the 1000 levels the UNet was trained on.
A conditioning of several chunks is joined into a single sequence first.

| Option | Default | Description |
| --- | --- | --- |
| `--noise` | `noise.pt` | Path to save the noise prediction to. |

### Sampling

A single prediction from a very noisy latent is a blur, so images are generated by running the UNet repeatedly, removing a little noise at each step.
See [docs/sampling.md](docs/sampling.md) for how this works.

#### Denoising

`sample denoise` steps a latent down through a noise schedule.

```bash
pydiffuse sample denoise latent.pt positive.pt negative.pt schedule.txt model.safetensors --cfg 7
```

This writes `denoised.pt`, a latent in the same shape that `vae decode` can turn into an image.
The latent should be noised to the schedule's first level - for a schedule from `noise schedule`, that means pure noise from `noise create`.
The schedule is a file like the one `noise schedule` writes, and must have at least two levels, each at least 0 and below 1.

At every step the UNet predicts the noise with both conditionings, and the CFG scale pushes the prediction away from the negative one and towards the positive one - at 1, the negative conditioning has no effect.
`euler` uses one pair of predictions per step, while `heun` makes a second pair at the next noise level and averages them, which is more accurate but about twice as slow.

| Option | Default | Description |
| --- | --- | --- |
| `--cfg` | `1.0` | How strongly to steer towards the positive conditioning. |
| `--algorithm` | `euler` | The algorithm to sample with - `euler` or `heun`. |
| `--output` | `denoised.pt` | Path to save the denoised latent to. |
