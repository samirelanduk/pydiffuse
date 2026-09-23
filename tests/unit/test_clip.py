from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import safetensors
import torch
from transformers import CLIPTokenizer

from pydiffuse.clip import (
    TOKENIZER_DIR,
    _apply_attention,
    _apply_mlp,
    _break_up_tokens,
    _create_position_embedding,
    _create_token_embedding,
    _create_token_string_mapping,
    _get_clip_mask,
    _get_encoder_layer_numbers,
    _get_layer_tensors,
    _get_norm_tensors,
    _text_to_tokens,
    embed,
    encode,
    tokenize,
)


class TokenizeTests(TestCase):
    @patch("pydiffuse.clip._text_to_tokens")
    @patch("pydiffuse.clip._break_up_tokens")
    @patch("pydiffuse.clip._create_token_string_mapping")
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

    @patch("pydiffuse.clip.CLIPTokenizer.from_pretrained")
    @patch("pydiffuse.clip._text_to_tokens")
    @patch("pydiffuse.clip._break_up_tokens")
    @patch("pydiffuse.clip._create_token_string_mapping")
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
    @patch("pydiffuse.clip._create_token_embedding")
    @patch("pydiffuse.clip._create_position_embedding")
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


class EncodeTests(TestCase):
    @patch("pydiffuse.clip.layer_norm")
    @patch("pydiffuse.clip._get_norm_tensors")
    @patch("pydiffuse.clip._apply_mlp")
    @patch("pydiffuse.clip._apply_attention")
    @patch("pydiffuse.clip._get_layer_tensors")
    @patch("pydiffuse.clip._get_clip_mask")
    def test_encode(
        self,
        mock_get_mask,
        mock_get_layers,
        mock_attention,
        mock_mlp,
        mock_get_norm,
        mock_layer_norm,
    ):
        mock_get_mask.return_value = "MASK"
        mock_get_layers.return_value = {0: {"a": 1}, 1: {"b": 2}}
        mock_attention.side_effect = ["ATTN0", "ATTN1"]
        mock_mlp.side_effect = ["MLP0", "MLP1"]
        mock_get_norm.return_value = {"weight": "NORM_W", "bias": "NORM_B"}
        mock_layer_norm.return_value = "CONDITIONING"
        embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        model = Mock(safetensors.safe_open)
        result = encode(embedding, model)
        self.assertEqual(mock_get_mask.call_count, 1)
        self.assertTrue(
            torch.equal(mock_get_mask.call_args_list[0][0][0], embedding.float())
        )
        mock_get_layers.assert_called_once_with(model)
        self.assertEqual(mock_attention.call_count, 2)
        self.assertTrue(
            torch.equal(mock_attention.call_args_list[0][0][0], embedding.float())
        )
        self.assertEqual(mock_attention.call_args_list[0][0][1], {"a": 1})
        self.assertEqual(mock_attention.call_args_list[0][0][2], "MASK")
        self.assertEqual(mock_attention.call_args_list[1][0][0], "MLP0")
        self.assertEqual(mock_attention.call_args_list[1][0][1], {"b": 2})
        self.assertEqual(mock_attention.call_args_list[1][0][2], "MASK")
        self.assertEqual(mock_mlp.call_count, 2)
        self.assertEqual(mock_mlp.call_args_list[0][0], ("ATTN0", {"a": 1}))
        self.assertEqual(mock_mlp.call_args_list[1][0], ("ATTN1", {"b": 2}))
        mock_get_norm.assert_called_once_with(model)
        mock_layer_norm.assert_called_once_with(
            input="MLP1", weight="NORM_W", bias="NORM_B"
        )
        self.assertEqual(result, "CONDITIONING")


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

    def test_empty_list(self):
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        result = _break_up_tokens([], tokenizer, max_length=8)
        self.assertEqual(
            result, [[49406, 49407, 49407, 49407, 49407, 49407, 49407, 49407]]
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


class GetClipMaskTests(TestCase):
    def test_dim_2_tensor(self):
        embedding = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
        mask = _get_clip_mask(embedding)
        minus_inf = torch.finfo(torch.float32).min
        self.assertTrue(torch.equal(mask, torch.tensor([[0.0, minus_inf], [0.0, 0.0]])))

    def test_dim_3_tensor(self):
        embedding = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = _get_clip_mask(embedding)
        minus_inf = torch.finfo(torch.float32).min
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor(
                    [
                        [0.0, minus_inf, minus_inf],
                        [0.0, 0.0, minus_inf],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            )
        )


class GetLayerTensorsTests(TestCase):
    @patch("pydiffuse.clip._get_encoder_layer_numbers")
    def test_get_layer_tensors(self, mock_get_layer_numbers):
        mock_get_layer_numbers.return_value = [3, 4, 5]
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "cond_stage_model.layers.3.norm2.weight",
            "cond_stage_model.layers.3.norm2.bias",
            "cond_stage_model.layers.3.mlp.fc1.weight",
            "cond_stage_model.layers.3.mlp.fc1.bias",
            "cond_stage_model.layers.3.mlp.fc2.weight",
            "cond_stage_model.layers.3.mlp.fc2.bias",
            "cond_stage_model.layers.3.attn.q_proj.weight",
            "cond_stage_model.layers.3.attn.q_proj.bias",
            "cond_stage_model.layers.3.attn.k_proj.weight",
            "cond_stage_model.layers.3.attn.k_proj.bias",
            "cond_stage_model.layers.3.attn.v_proj.weight",
            "cond_stage_model.layers.3.attn.v_proj.bias",
            "cond_stage_model.layers.3.attn.out_proj.weight",
            "cond_stage_model.layers.3.attn.out_proj.bias",
            "cond_stage_model.layers.4.norm1.weight",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_model.layers.4.norm2.weight",
            "cond_stage_model.layers.4.norm2.bias",
            "cond_stage_model.layers.4.mlp.fc1.weight",
            "cond_stage_model.layers.4.mlp.fc1.bias",
            "cond_stage_model.layers.4.mlp.fc2.weight",
            "cond_stage_model.layers.4.mlp.fc2.bias",
            "cond_stage_model.layers.4.attn.q_proj.weight",
            "cond_stage_model.layers.4.attn.q_proj.bias",
            "cond_stage_model.layers.4.attn.k_proj.weight",
            "cond_stage_model.layers.4.attn.k_proj.bias",
            "cond_stage_model.layers.4.attn.v_proj.weight",
            "cond_stage_model.layers.4.attn.v_proj.bias",
            "cond_stage_model.layers.4.attn.out_proj.weight",
            "cond_stage_model.layers.4.attn.out_proj.bias",
            "cond_stage_model.layers.5.norm1.weight",
            "cond_stage_model.layers.5.norm1.bias",
            "cond_stage_model.layers.5.norm2.weight",
            "cond_stage_model.layers.5.norm2.bias",
            "cond_stage_model.layers.5.mlp.fc1.weight",
            "cond_stage_model.layers.5.mlp.fc1.bias",
            "cond_stage_model.layers.5.mlp.fc2.weight",
            "cond_stage_model.layers.5.mlp.fc2.bias",
            "cond_stage_model.layers.5.attn.q_proj.weight",
            "cond_stage_model.layers.5.attn.q_proj.bias",
            "cond_stage_model.layers.5.attn.k_proj.weight",
            "cond_stage_model.layers.5.attn.k_proj.bias",
            "cond_stage_model.layers.5.attn.v_proj.weight",
            "cond_stage_model.layers.5.attn.v_proj.bias",
            "cond_stage_model.layers.5.attn.out_proj.weight",
            "cond_stage_model.layers.5.attn.out_proj.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        layer_tensors = _get_layer_tensors(tensors)
        for n in layer_tensors:
            for key in layer_tensors[n]:
                layer_tensors[n][key] = layer_tensors[n][key].tolist()
        self.assertEqual(
            layer_tensors,
            {
                3: {
                    "norm1_weight": [38, 76, 114],
                    "norm1_bias": [36, 72, 108],
                    "norm2_weight": [38, 76, 114],
                    "norm2_bias": [36, 72, 108],
                    "mlp1_weight": [40, 80, 120],
                    "mlp1_bias": [38, 76, 114],
                    "mlp2_weight": [40, 80, 120],
                    "mlp2_bias": [38, 76, 114],
                    "attn_q_weight": [44, 88, 132],
                    "attn_q_bias": [42, 84, 126],
                    "attn_k_weight": [44, 88, 132],
                    "attn_k_bias": [42, 84, 126],
                    "attn_v_weight": [44, 88, 132],
                    "attn_v_bias": [42, 84, 126],
                    "attn_out_weight": [46, 92, 138],
                    "attn_out_bias": [44, 88, 132],
                },
                4: {
                    "norm1_weight": [38, 76, 114],
                    "norm1_bias": [36, 72, 108],
                    "norm2_weight": [38, 76, 114],
                    "norm2_bias": [36, 72, 108],
                    "mlp1_weight": [40, 80, 120],
                    "mlp1_bias": [38, 76, 114],
                    "mlp2_weight": [40, 80, 120],
                    "mlp2_bias": [38, 76, 114],
                    "attn_q_weight": [44, 88, 132],
                    "attn_q_bias": [42, 84, 126],
                    "attn_k_weight": [44, 88, 132],
                    "attn_k_bias": [42, 84, 126],
                    "attn_v_weight": [44, 88, 132],
                    "attn_v_bias": [42, 84, 126],
                    "attn_out_weight": [46, 92, 138],
                    "attn_out_bias": [44, 88, 132],
                },
                5: {
                    "norm1_weight": [38, 76, 114],
                    "norm1_bias": [36, 72, 108],
                    "norm2_weight": [38, 76, 114],
                    "norm2_bias": [36, 72, 108],
                    "mlp1_weight": [40, 80, 120],
                    "mlp1_bias": [38, 76, 114],
                    "mlp2_weight": [40, 80, 120],
                    "mlp2_bias": [38, 76, 114],
                    "attn_q_weight": [44, 88, 132],
                    "attn_q_bias": [42, 84, 126],
                    "attn_k_weight": [44, 88, 132],
                    "attn_k_bias": [42, 84, 126],
                    "attn_v_weight": [44, 88, 132],
                    "attn_v_bias": [42, 84, 126],
                    "attn_out_weight": [46, 92, 138],
                    "attn_out_bias": [44, 88, 132],
                },
            },
        )

    @patch("pydiffuse.clip._get_encoder_layer_numbers")
    def test_all_layers_must_be_present(self, mock_get_layer_numbers):
        mock_get_layer_numbers.return_value = [3, 4, 5]
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "cond_stage_model.layers.3.norm2.weight",
            "cond_stage_model.layers.3.norm2.bias",
            "cond_stage_model.layers.3.mlp.fc1.weight",
            "cond_stage_model.layers.3.mlp.fc1.bias",
            "cond_stage_model.layers.3.mlp.fc2.weight",
            "cond_stage_model.layers.3.attn.q_proj.weight",
            "cond_stage_model.layers.3.attn.q_proj.bias",
            "cond_stage_model.layers.3.attn.k_proj.weight",
            "cond_stage_model.layers.3.attn.k_proj.bias",
            "cond_stage_model.layers.3.attn.v_proj.weight",
            "cond_stage_model.layers.3.attn.v_proj.bias",
            "cond_stage_model.layers.3.attn.out_proj.weight",
            "cond_stage_model.layers.3.attn.out_proj.bias",
            "cond_stage_model.layers.4.norm1.weight",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_model.layers.4.norm2.weight",
            "cond_stage_model.layers.4.norm2.bias",
            "cond_stage_model.layers.4.mlp.fc1.weight",
            "cond_stage_model.layers.4.mlp.fc1.bias",
            "cond_stage_model.layers.4.mlp.fc2.weight",
            "cond_stage_model.layers.4.mlp.fc2.bias",
            "cond_stage_model.layers.4.attn.q_proj.weight",
            "cond_stage_model.layers.4.attn.q_proj.bias",
            "cond_stage_model.layers.4.attn.k_proj.weight",
            "cond_stage_model.layers.4.attn.k_proj.bias",
            "cond_stage_model.layers.4.attn.v_proj.weight",
            "cond_stage_model.layers.4.attn.v_proj.bias",
            "cond_stage_model.layers.4.attn.out_proj.weight",
            "cond_stage_model.layers.4.attn.out_proj.bias",
            "cond_stage_model.layers.5.norm1.weight",
            "cond_stage_model.layers.5.norm1.bias",
            "cond_stage_model.layers.5.norm2.weight",
            "cond_stage_model.layers.5.norm2.bias",
            "cond_stage_model.layers.5.mlp.fc1.weight",
            "cond_stage_model.layers.5.mlp.fc1.bias",
            "cond_stage_model.layers.5.mlp.fc2.weight",
            "cond_stage_model.layers.5.mlp.fc2.bias",
            "cond_stage_model.layers.5.attn.q_proj.weight",
            "cond_stage_model.layers.5.attn.q_proj.bias",
            "cond_stage_model.layers.5.attn.k_proj.weight",
            "cond_stage_model.layers.5.attn.k_proj.bias",
            "cond_stage_model.layers.5.attn.v_proj.weight",
            "cond_stage_model.layers.5.attn.v_proj.bias",
            "cond_stage_model.layers.5.attn.out_proj.weight",
            "cond_stage_model.layers.5.attn.out_proj.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        with self.assertRaises(ValueError) as context:
            _get_layer_tensors(tensors)
        self.assertEqual(
            str(context.exception), "Layer 3 mlp2_bias could not be found in the model"
        )


class GetEncoderLayerNumbersTests(TestCase):
    def test_get_encoder_layer_numbers(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.1.norm1.weight",
            "cond_stage_model.layers.1.norm1.bias",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_Model.layers.3.norm1.bias",
            "cond_stage_model.layers.20.norm1.bias",
            "sd_model.layers.10.norm1.bias",
            "xxx",
        ]
        result = _get_encoder_layer_numbers(tensors)
        self.assertEqual(result, [1, 3, 4, 20])


class ApplyAttentionTests(TestCase):
    @patch("pydiffuse.clip.layer_norm")
    @patch("pydiffuse.clip.linear")
    def test_can_apply_attention(self, mock_linear, mock_norm):
        conditioning = torch.tensor(
            [
                [
                    [
                        100.0,
                        200.0,
                        300.0,
                        100.0,
                        200.0,
                        300.0,
                        100.0,
                        200.0,
                        300.0,
                        100.0,
                        200.0,
                        300.0,
                    ]
                ]
                * 3,
                [
                    [
                        400.0,
                        500.0,
                        600.0,
                        400.0,
                        500.0,
                        600.0,
                        400.0,
                        500.0,
                        600.0,
                        400.0,
                        500.0,
                        600.0,
                    ]
                ]
                * 3,
            ]
        )
        tensors = {
            "attn_q_weight": "ATTN_Q_WEIGHT",
            "attn_q_bias": "ATTN_Q_BIAS",
            "attn_k_weight": "ATTN_K_WEIGHT",
            "attn_k_bias": "ATTN_K_BIAS",
            "attn_v_weight": "ATTN_V_WEIGHT",
            "attn_v_bias": "ATTN_V_BIAS",
            "attn_out_weight": "ATTN_OUT_WEIGHT",
            "attn_out_bias": "ATTN_OUT_BIAS",
            "norm1_weight": "NORM1_WEIGHT",
            "norm1_bias": "NORM1_BIAS",
        }
        mock_norm.return_value = torch.tensor(
            [
                [[5.0, 10.0, 15.0, 5.0, 10.0, 15.0, 5.0, 10.0, 15.0, 5.0, 10.0, 15.0]]
                * 3,
                [
                    [
                        20.0,
                        25.0,
                        30.0,
                        20.0,
                        25.0,
                        30.0,
                        20.0,
                        25.0,
                        30.0,
                        20.0,
                        25.0,
                        30.0,
                    ]
                ]
                * 3,
            ]
        )
        mock_linear.side_effect = [
            torch.tensor(
                [
                    [
                        [
                            11.0,
                            21.0,
                            31.0,
                            11.0,
                            21.0,
                            31.0,
                            11.0,
                            21.0,
                            31.0,
                            11.0,
                            21.0,
                            31.0,
                        ]
                    ]
                    * 3,
                    [
                        [
                            41.0,
                            51.0,
                            61.0,
                            41.0,
                            51.0,
                            61.0,
                            41.0,
                            51.0,
                            61.0,
                            41.0,
                            51.0,
                            61.0,
                        ]
                    ]
                    * 3,
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            12.0,
                            22.0,
                            32.0,
                            12.0,
                            22.0,
                            32.0,
                            12.0,
                            22.0,
                            32.0,
                            12.0,
                            22.0,
                            32.0,
                        ]
                    ]
                    * 3,
                    [
                        [
                            42.0,
                            52.0,
                            62.0,
                            42.0,
                            52.0,
                            62.0,
                            42.0,
                            52.0,
                            62.0,
                            42.0,
                            52.0,
                            62.0,
                        ]
                    ]
                    * 3,
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            13.0,
                            23.0,
                            33.0,
                            13.0,
                            23.0,
                            33.0,
                            13.0,
                            23.0,
                            33.0,
                            13.0,
                            23.0,
                            33.0,
                        ]
                    ]
                    * 3,
                    [
                        [
                            43.0,
                            53.0,
                            63.0,
                            43.0,
                            53.0,
                            63.0,
                            43.0,
                            53.0,
                            63.0,
                            43.0,
                            53.0,
                            63.0,
                        ]
                    ]
                    * 3,
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            40.0,
                            50.0,
                            60.0,
                            40.0,
                            50.0,
                            60.0,
                            40.0,
                            50.0,
                            60.0,
                            40.0,
                            50.0,
                            60.0,
                        ]
                    ]
                    * 3,
                    [
                        [
                            70.0,
                            80.0,
                            90.0,
                            70.0,
                            80.0,
                            90.0,
                            70.0,
                            80.0,
                            90.0,
                            70.0,
                            80.0,
                            90.0,
                        ]
                    ]
                    * 3,
                ]
            ),
        ]
        mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        result = _apply_attention(conditioning, tensors, mask)
        self.assertEqual(result.size(), (2, 3, 12))
        self.assertEqual(
            [round(float(v), 4) for v in result.reshape(-1).tolist()],
            [
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                140,
                250,
                360,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
                470,
                580,
                690,
            ],
        )
        self.assertEqual(mock_norm.call_args_list[0][0][0], "NORM1_WEIGHT")
        self.assertEqual(mock_norm.call_args_list[0][0][1], "NORM1_BIAS")
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][2], conditioning))
        self.assertEqual(mock_linear.call_args_list[0][0][0], "ATTN_Q_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[0][0][1], "ATTN_Q_BIAS")
        self.assertTrue(
            torch.equal(mock_linear.call_args_list[0][0][2], mock_norm.return_value)
        )
        self.assertEqual(mock_linear.call_args_list[1][0][0], "ATTN_K_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[1][0][1], "ATTN_K_BIAS")
        self.assertTrue(
            torch.equal(mock_linear.call_args_list[1][0][2], mock_norm.return_value)
        )
        self.assertEqual(mock_linear.call_args_list[2][0][0], "ATTN_V_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[2][0][1], "ATTN_V_BIAS")
        self.assertTrue(
            torch.equal(mock_linear.call_args_list[2][0][2], mock_norm.return_value)
        )
        self.assertEqual(mock_linear.call_args_list[3][0][2].size(), (2, 3, 12))
        self.assertEqual(
            [
                round(float(v), 4)
                for v in mock_linear.call_args_list[3][0][2].reshape(-1).tolist()
            ],
            [
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                13.0,
                23.0,
                33.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
                43.0,
                53.0,
                63.0,
            ],
        )


class ApplyMlpTests(TestCase):
    @patch("pydiffuse.clip.layer_norm")
    @patch("pydiffuse.clip.linear")
    def test_can_apply_mlp(self, mock_linear, mock_norm):
        conditioning = torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
        tensors = {
            "norm2_weight": "NORM2_WEIGHT",
            "norm2_bias": "NORM2_BIAS",
            "mlp1_weight": "MLP1_WEIGHT",
            "mlp1_bias": "MLP1_BIAS",
            "mlp2_weight": "MLP2_WEIGHT",
            "mlp2_bias": "MLP2_BIAS",
        }
        mock_norm.return_value = torch.tensor([[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]])
        mock_linear.side_effect = [
            torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]),
        ]
        result = _apply_mlp(conditioning, tensors)
        self.assertTrue(
            torch.equal(result, torch.tensor([[170.0, 280.0, 390.0], [500, 610, 720]]))
        )
        mock_norm.assert_called_once_with("NORM2_WEIGHT", "NORM2_BIAS", conditioning)
        self.assertEqual(mock_linear.call_args_list[0][0][0], "MLP1_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[0][0][1], "MLP1_BIAS")
        self.assertTrue(
            torch.equal(
                mock_linear.call_args_list[0][0][2],
                torch.tensor([[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]]),
            )
        )
        self.assertEqual(mock_linear.call_args_list[1][0][2].size(), (2, 3))
        self.assertEqual(mock_linear.call_args_list[1][0][0], "MLP2_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[1][0][1], "MLP2_BIAS")
        self.assertEqual(
            [
                round(float(v), 4)
                for v in mock_linear.call_args_list[1][0][2].reshape(-1).tolist()
            ],
            [0.8458, -0.0643, 2.9819, 3.9956, 4.9990, 5.9998],
        )


class GetNormTensorsTests(TestCase):
    def test_get_norm_tensors(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.weight",
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.bias",
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        norm_tensors = _get_norm_tensors(tensors)
        for k in norm_tensors:
            norm_tensors[k] = norm_tensors[k].tolist()
        self.assertEqual(
            norm_tensors,
            {
                "weight": [71, 142, 213],
                "bias": [69, 138, 207],
            },
        )

    def test_all_norm_tensors_must_be_present(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.weight",
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        with self.assertRaises(ValueError) as context:
            _get_norm_tensors(tensors)
        self.assertEqual(
            str(context.exception), "Final norm bias could not be found in the model"
        )
