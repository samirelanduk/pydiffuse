import re
from pathlib import Path

import safetensors
import torch
from transformers import CLIPTokenizer

from pydiffuser.layers import layer_norm, linear

MAX_LENGTH = 77

TOKENIZER_DIR = Path(__file__).parent / "data" / "clip_tokenizer"


def tokenize(
    text: str,
    clip_tokenizer: CLIPTokenizer | None = None,
) -> tuple[list[list[int]], list[list[tuple[str, int]]]]:
    """Tokenizes the given text with a CLIP tokenizer. Tokens will be returned
    as a list of list of token integers, and a mapping of substrings to their
    integers to show how the original text was split into tokens."""

    tokenizer = clip_tokenizer or CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
    tokens = _text_to_tokens(text, tokenizer)
    tokens = _break_up_tokens(tokens, tokenizer)
    mappings = _create_token_string_mapping(tokens, tokenizer)
    return tokens, mappings


def embed(tokens: list[list[int]], model: safetensors.safe_open) -> None:
    """Takes a set of tokens and maps them to the correct embedding vectors for
    this model. The model should contain the two relevant tensors, and match the
    CLIP dictionary used during tokenization."""

    tokens_tensor = torch.tensor(tokens)
    token_vectors = _create_token_embedding(model, tokens_tensor)
    position_vectors = _create_position_embedding(model, tokens_tensor)
    return token_vectors + position_vectors


def encode(embedding: torch.Tensor, model: safetensors.safe_open) -> None:
    """Takes a set of embedding vectors, and produces a tensor of the same shape
    which represents the semanting meaning of each token. The model should
    contain all required tensors for each layer of this process, though it can
    have any number of layers."""

    conditioning = embedding.float()
    mask = _get_clip_mask(conditioning)
    layer_tensors = _get_layer_tensors(model)
    for layer_tensor in layer_tensors.values():
        conditioning = _apply_attention(conditioning, layer_tensor, mask)
        conditioning = _apply_mlp(conditioning, layer_tensor)
    conditioning = layer_norm(input=conditioning, **_get_norm_tensors(model))
    return conditioning


def _text_to_tokens(text: str, clip_tokenizer: CLIPTokenizer) -> list[int]:
    """Creates a list of tokens IDs from the given text. It is a single flat
    list regardless of the length of the text, and the start and end tokens are
    removed."""

    all_tokens = clip_tokenizer.encode(text)
    if all_tokens[0] == clip_tokenizer.bos_token_id:
        all_tokens.pop(0)
    if all_tokens[-1] == clip_tokenizer.eos_token_id:
        all_tokens.pop(-1)
    return all_tokens


def _break_up_tokens(
    tokens: list[int],
    clip_tokenizer: CLIPTokenizer,
    max_length: int = MAX_LENGTH,
) -> list[list[int]]:
    """Breaks a list of tokens into a list of lists of tokens, where each
    sublist is of length max_length, and the start and end tokens are added."""

    bos = clip_tokenizer.bos_token_id
    eos = clip_tokenizer.eos_token_id
    pad = clip_tokenizer.pad_token_id
    token_lists = [
        tokens[i : i + max_length - 2] for i in range(0, len(tokens), max_length - 2)
    ]
    token_lists = [[bos] + t + [eos] for t in token_lists]
    if len(token_lists[-1]) < max_length:
        token_lists[-1].pop(-1)
        token_lists[-1] += [pad] * (max_length - len(token_lists[-1]))
    return token_lists


def _create_token_string_mapping(
    tokens: list[list[int]], clip_tokenizer: CLIPTokenizer
) -> list[list[tuple[str, int]]]:
    """Creates a mapping of token values to token integers."""

    mappings = []
    for sub_list in tokens:
        strings = clip_tokenizer.convert_ids_to_tokens(sub_list)
        mappings.append([(s, t) for t, s in zip(sub_list, strings)])
    return mappings


def _create_token_embedding(
    tensors: safetensors.safe_open,
    tokens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Finds the correct token embedding tensor in a model, and runs the tokens
    through it."""

    keys = tensors.keys()
    for key in keys:
        if key.endswith("token_embedding.weight"):
            tensor = tensors.get_tensor(key)
            token_embedding = torch.nn.Embedding(
                num_embeddings=tensor.shape[0], embedding_dim=tensor.shape[1]
            ).from_pretrained(tensor)
            return token_embedding(tokens_tensor)
    raise ValueError("Token embedding tensor not found in model")


def _create_position_embedding(
    tensors: safetensors.safe_open,
    tokens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Finds the correct position embedding tensor in a model, and runs each of
    the positions from 0 to whatever the maximum position is through it."""

    keys = tensors.keys()
    for key in keys:
        if key.endswith("position_embedding.weight"):
            tensor = tensors.get_tensor(key)
            position_embedding = torch.nn.Embedding(
                num_embeddings=tensor.shape[0], embedding_dim=tensor.shape[1]
            ).from_pretrained(tensor)
            return position_embedding(torch.arange(tokens_tensor.shape[1]))
    raise ValueError("Position embedding tensor not found in model")


def _get_clip_mask(embedding: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the embedding tensor. The mask is a square matrix of
    the same dimensions as the embedding tensor, with all values set to -inf
    except for the upper triangular part, which is set to 0."""

    return torch.full(
        (embedding.shape[1], embedding.shape[1]),
        -torch.finfo(embedding.dtype).max,
        device="cpu",
    ).triu_(1)


def _get_layer_tensors(tensors: safetensors.safe_open) -> dict:
    """Finds the tensors used in CLIP encode's encoder layers."""

    layer_numbers = _get_encoder_layer_numbers(tensors)
    layer_tensor = {n: {} for n in layer_numbers}
    lookup = {
        "norm1.weight": "norm1_weight",
        "norm1.bias": "norm1_bias",
        "norm2.weight": "norm2_weight",
        "norm2.bias": "norm2_bias",
        "mlp.fc1.weight": "mlp1_weight",
        "mlp.fc1.bias": "mlp1_bias",
        "mlp.fc2.weight": "mlp2_weight",
        "mlp.fc2.bias": "mlp2_bias",
        "attn.q_proj.weight": "attn_q_weight",
        "attn.q_proj.bias": "attn_q_bias",
        "attn.k_proj.weight": "attn_k_weight",
        "attn.k_proj.bias": "attn_k_bias",
        "attn.v_proj.weight": "attn_v_weight",
        "attn.v_proj.bias": "attn_v_bias",
        "attn.out_proj.weight": "attn_out_weight",
        "attn.out_proj.bias": "attn_out_bias",
    }
    keys = tensors.keys()
    for key in keys:
        if "cond_stage_model" not in key.lower():
            continue
        layer_number_match = re.search(r"layers\.(\d+)\.(.*)", key)
        if not layer_number_match:
            continue
        layer_number = int(layer_number_match.group(1))
        for suffix, name in lookup.items():
            if key.endswith(suffix):
                layer_tensor[layer_number][name] = tensors.get_tensor(key).float()
                break
    for layer_number in layer_numbers:
        for value in lookup.values():
            if value not in layer_tensor[layer_number]:
                raise ValueError(
                    f"Layer {layer_number} {value} could not be found in the model"
                )
    return layer_tensor


def _get_encoder_layer_numbers(tensors: safetensors.safe_open) -> list[int]:
    """Gets a list of layer numbers present in the model."""

    layer_numbers = set()
    keys = tensors.keys()
    for key in keys:
        if "cond_stage_model" not in key.lower():
            continue
        layer_number_match = re.search(r"layers\.(\d+)", key)
        if not layer_number_match:
            continue
        layer_number = int(layer_number_match.group(1))
        layer_numbers.add(layer_number)
    return sorted(layer_numbers)


def _apply_attention(
    conditioning: torch.Tensor, layer_tensor: dict, mask: torch.Tensor
) -> torch.Tensor:
    """Applies the attention component of a CLIP encode layer."""

    attn_conditioning = layer_norm(
        layer_tensor["norm1_weight"], layer_tensor["norm1_bias"], conditioning
    )
    Q = linear(
        layer_tensor["attn_q_weight"], layer_tensor["attn_q_bias"], attn_conditioning
    )
    K = linear(
        layer_tensor["attn_k_weight"], layer_tensor["attn_k_bias"], attn_conditioning
    )
    V = linear(
        layer_tensor["attn_v_weight"], layer_tensor["attn_v_bias"], attn_conditioning
    )
    HEADS = 12
    head_dim = Q.shape[-1] // HEADS
    batch = Q.shape[0]
    seq_len = Q.shape[1]
    Q = Q.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, HEADS, head_dim).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
    scores = scores + mask
    attn_output = torch.softmax(scores, dim=-1) @ V
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    attn_output = linear(
        layer_tensor["attn_out_weight"], layer_tensor["attn_out_bias"], attn_output
    )
    return conditioning + attn_output


def _apply_mlp(conditioning: torch.Tensor, layer_tensor: dict) -> torch.Tensor:
    """Applies the MLP component of a CLIP encode layer. An approximation of
    GeLU is used for the activation function."""

    quick_gelu = lambda x: x * torch.sigmoid(1.702 * x)
    mlp_output = layer_norm(
        layer_tensor["norm2_weight"], layer_tensor["norm2_bias"], conditioning
    )
    mlp_output = linear(
        layer_tensor["mlp1_weight"], layer_tensor["mlp1_bias"], mlp_output
    )
    mlp_output = quick_gelu(mlp_output)
    mlp_output = linear(
        layer_tensor["mlp2_weight"], layer_tensor["mlp2_bias"], mlp_output
    )
    return conditioning + mlp_output


def _get_norm_tensors(tensors: safetensors.safe_open) -> dict:
    """Finds the tensors used in CLIP encode's final normalization stage."""

    norm_tensors = {"weight": None, "bias": None}
    lookup = {
        "final_layer_norm.weight": "weight",
        "final_layer_norm.bias": "bias",
    }
    keys = tensors.keys()
    for key in keys:
        if "cond_stage_model" not in key.lower():
            continue
        layer_number_match = re.search(r"layers\.(\d+)\.(.*)", key)
        if not layer_number_match:
            for suffix, name in lookup.items():
                if key.endswith(suffix):
                    norm_tensors[name] = tensors.get_tensor(key).float()
                    break
    for key, value in norm_tensors.items():
        if value is None:
            raise ValueError(f"Final norm {key} could not be found in the model")
    return norm_tensors
