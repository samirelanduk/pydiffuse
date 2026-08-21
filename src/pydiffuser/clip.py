from pathlib import Path

from transformers import CLIPTokenizer

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
