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

Commands are namespaced by group - currently `clip` and `vae`.

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
