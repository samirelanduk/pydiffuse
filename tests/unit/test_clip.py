from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import safetensors
import torch
from transformers import CLIPTokenizer

from pydiffuser.clip import (
    TOKENIZER_DIR,
    _break_up_tokens,
    _create_position_embedding,
    _create_token_embedding,
    _create_token_string_mapping,
    _text_to_tokens,
    embed,
    tokenize,
)


class TokenizeTests(TestCase):
    @patch("pydiffuser.clip._text_to_tokens")
    @patch("pydiffuser.clip._break_up_tokens")
    @patch("pydiffuser.clip._create_token_string_mapping")
    def test_tokenize_custom_tokenizer(
        self,
        mock_create_mapping,
        mock_break_up,
        mock_to_tokens,
    ):
        mock_create_mapping.return_value = [
            [("the", 599)],
            [("big", 1915), ("one", 1980)],
            [("is", 595), ("coming", 14916)],
        ]
        tokenizer = Mock()
        tokens, mappings = tokenize("A photo of a cat.", tokenizer)

        mock_to_tokens.assert_called_once_with("A photo of a cat.", tokenizer)
        mock_break_up.assert_called_once_with(mock_to_tokens.return_value, tokenizer)
        mock_create_mapping.assert_called_once_with(
            mock_break_up.return_value, tokenizer
        )
        self.assertEqual(tokens, mock_break_up.return_value)
        self.assertEqual(mappings, mock_create_mapping.return_value)

    @patch("pydiffuser.clip.CLIPTokenizer.from_pretrained")
    @patch("pydiffuser.clip._text_to_tokens")
    @patch("pydiffuser.clip._break_up_tokens")
    @patch("pydiffuser.clip._create_token_string_mapping")
    def test_tokenize_default_tokenizer(
        self,
        mock_create_mapping,
        mock_break_up,
        mock_to_tokens,
        mock_from_pretrained,
    ):
        mock_create_mapping.return_value = [
            [("the", 599)],
            [("big", 1915), ("one", 1980)],
            [("is", 595), ("coming", 14916)],
        ]
        tokenize("A photo of a cat.")
        mock_from_pretrained.assert_called_once_with(TOKENIZER_DIR)
        mock_to_tokens.assert_called_once_with(
            "A photo of a cat.", mock_from_pretrained.return_value
        )
        mock_break_up.assert_called_once_with(
            mock_to_tokens.return_value, mock_from_pretrained.return_value
        )
        mock_create_mapping.assert_called_once_with(
            mock_break_up.return_value, mock_from_pretrained.return_value
        )


class EmbedTests(TestCase):
    @patch("pydiffuser.clip._create_token_embedding")
    @patch("pydiffuser.clip._create_position_embedding")
    def test_embed(self, mock_position, mock_token):
        tokens = [[0, 1, 2], [3, 4, 5]]
        model = Mock(safetensors.safe_open)
        mock_token.return_value = [10, 20]
        mock_position.return_value = [30, 40]
        result = embed(tokens, model)
        self.assertEqual(mock_token.call_count, 1)
        self.assertEqual(mock_token.call_args_list[0][0][0], model)
        self.assertTrue(
            torch.equal(mock_token.call_args_list[0][0][1], torch.tensor(tokens))
        )
        self.assertEqual(mock_position.call_count, 1)
        self.assertEqual(mock_position.call_args_list[0][0][0], model)
        self.assertTrue(
            torch.equal(mock_position.call_args_list[0][0][1], torch.tensor(tokens))
        )
        self.assertEqual(result, [10, 20, 30, 40])


class TextToTokensTests(TestCase):
    def test_text_to_tokens(self):
        prompt = "A photo of a cat."
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        tokens = _text_to_tokens(prompt, tokenizer)
        self.assertEqual(tokens, [320, 1125, 539, 320, 2368, 269])


class BreakUpTokensTests(TestCase):
    def test_short_list(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _break_up_tokens(tokens, tokenizer, max_length=20)
        self.assertEqual(
            result,
            [
                [
                    49406,
                    1,
                    2,
                    3,
                    4,
                    5,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    49407,
                    49407,
                    49407,
                    49407,
                    49407,
                    49407,
                    49407,
                    49407,
                ]
            ],
        )

    def test_list_of_length_max_length(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _break_up_tokens(tokens, tokenizer, max_length=13)
        self.assertEqual(
            result, [[49406, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 49407]]
        )

    def test_list_longer_than_max_length(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _break_up_tokens(tokens, tokenizer, max_length=8)
        self.assertEqual(
            result,
            [
                [49406, 1, 2, 3, 4, 5, 10, 49407],
                [49406, 11, 12, 13, 14, 15, 49407, 49407],
            ],
        )


class CreateTokenStringMappingTests(TestCase):
    def test_map_tokens_to_strings_single_list(self):
        tokens = [[599, 1915, 1980]]
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _create_token_string_mapping(tokens, tokenizer)
        self.assertEqual(result, [[("the", 599), ("big", 1915), ("one", 1980)]])

    def test_map_tokens_to_strings_multiple_lists(self):
        tokens = [[599, 1915, 1980], [595, 14916]]
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _create_token_string_mapping(tokens, tokenizer)
        self.assertEqual(
            result,
            [
                [("the", 599), ("big", 1915), ("one", 1980)],
                [("is", 595), ("coming", 14916)],
            ],
        )


class CreateTokenEmbeddingTests(TestCase):
    def test_creates_token_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "1",
            "2",
            "xxx.text_model.token_embedding.weight",
            "3",
        ]
        tensors.get_tensor.return_value = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        tokens = torch.tensor([[0, 2, 1]])
        result = _create_token_embedding(tensors, tokens)
        expected = torch.tensor([[[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]])
        self.assertTrue(torch.equal(result, expected))

    def test_raises_if_no_token_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = ["something_else.weight"]
        tokens = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            _create_token_embedding(tensors, tokens)


class CreatePositionEmbeddingTests(TestCase):
    def test_creates_position_embedding(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tensors = MagicMock()
        tensors.keys.return_value = [
            "1",
            "2",
            "xxx.text_model.position_embedding.weight",
            "3",
        ]
        tensors.get_tensor.return_value = weight
        tokens = torch.tensor([[10, 20, 30]])
        result = _create_position_embedding(tensors, tokens)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_raises_if_no_position_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = ["something_else.weight"]
        tokens = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            _create_position_embedding(tensors, tokens)
