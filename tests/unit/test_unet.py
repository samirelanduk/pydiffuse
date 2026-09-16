from unittest import TestCase
from unittest.mock import Mock, patch

import safetensors
import torch

from pydiffuse.unet import (
    _attention,
    _block,
    _combine_chunks,
    _feed_forward,
    _get_layers,
    _get_transformer_tensors,
    _input_blocks,
    _noise_level_to_embedding,
    _noise_to_t,
    _out_layers,
    _output_blocks,
    _resnet_block,
    _time_embed,
    _timestep_sinusoids,
    _transformer,
    _transformer_block,
    _upsample,
)


class GetLayersTests(TestCase):
    @patch("pydiffuse.unet._get_layer")
    def test_get_layers(self, mock_layer):
        mock_layer.side_effect = lambda model, prefix: f"layer {prefix}"
        model = Mock(safetensors.safe_open)
        layers = _get_layers(model, "prefix", ("norm1", "conv1", "nin_shortcut"))

        self.assertEqual(
            [call[0] for call in mock_layer.call_args_list],
            [
                (model, "prefix.norm1"),
                (model, "prefix.conv1"),
                (model, "prefix.nin_shortcut"),
            ],
        )
        self.assertEqual(
            layers,
            {
                "norm1": "layer prefix.norm1",
                "conv1": "layer prefix.conv1",
                "nin_shortcut": "layer prefix.nin_shortcut",
            },
        )


class GetTransformerTensorsTests(TestCase):
    @patch("pydiffuse.unet._get_numbers")
    @patch("pydiffuse.unet._get_layers")
    def test_get_transformer_tensors(self, mock_layers, mock_numbers):
        mock_numbers.return_value = [0, 2]
        mock_layers.side_effect = lambda model, prefix, names: {
            name: f"layers {prefix}.{name}" for name in names
        }
        model = Mock(safetensors.safe_open)
        tensors = _get_transformer_tensors(
            model, "model.diffusion_model.input_blocks.1.1"
        )
        mock_numbers.assert_called_once_with(
            model, "model.diffusion_model.input_blocks.1.1.transformer_blocks"
        )
        self.assertEqual(
            [call[0] for call in mock_layers.call_args_list],
            [
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1",
                    ("norm", "proj_in", "proj_out"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0",
                    ("norm1", "norm2", "norm3", "ff.net.0.proj", "ff.net.2"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1",
                    ("to_q", "to_k", "to_v", "to_out.0"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2",
                    ("to_q", "to_k", "to_v", "to_out.0"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.2",
                    ("norm1", "norm2", "norm3", "ff.net.0.proj", "ff.net.2"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn1",
                    ("to_q", "to_k", "to_v", "to_out.0"),
                ),
                (
                    model,
                    "model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn2",
                    ("to_q", "to_k", "to_v", "to_out.0"),
                ),
            ],
        )
        self.assertEqual(
            tensors,
            {
                "norm": "layers model.diffusion_model.input_blocks.1.1.norm",
                "proj_in": "layers model.diffusion_model.input_blocks.1.1.proj_in",
                "proj_out": "layers model.diffusion_model.input_blocks.1.1.proj_out",
                "transformer_blocks": [
                    {
                        "norm1": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1",
                        "norm2": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2",
                        "norm3": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3",
                        "ff.net.0.proj": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj",
                        "ff.net.2": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2",
                        "attn1": {
                            "to_q": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q",
                            "to_k": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k",
                            "to_v": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v",
                            "to_out.0": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0",
                        },
                        "attn2": {
                            "to_q": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q",
                            "to_k": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k",
                            "to_v": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v",
                            "to_out.0": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0",
                        },
                    },
                    {
                        "norm1": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.norm1",
                        "norm2": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.norm2",
                        "norm3": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.norm3",
                        "ff.net.0.proj": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.ff.net.0.proj",
                        "ff.net.2": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.ff.net.2",
                        "attn1": {
                            "to_q": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn1.to_q",
                            "to_k": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn1.to_k",
                            "to_v": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn1.to_v",
                            "to_out.0": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn1.to_out.0",
                        },
                        "attn2": {
                            "to_q": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn2.to_q",
                            "to_k": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn2.to_k",
                            "to_v": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn2.to_v",
                            "to_out.0": "layers model.diffusion_model.input_blocks.1.1.transformer_blocks.2.attn2.to_out.0",
                        },
                    },
                ],
            },
        )


class NoiseLevelToEmbeddingTests(TestCase):
    @patch("pydiffuse.unet._noise_to_t")
    @patch("pydiffuse.unet._timestep_sinusoids")
    @patch("pydiffuse.unet._time_embed")
    def test_noise_level_to_embedding(
        self, mock_time_embed, mock_sinusoids, mock_noise_to_t
    ):
        time_embed = [
            {"weight": torch.zeros(128, 32), "bias": torch.zeros(128)},
            {"weight": torch.zeros(128, 128), "bias": torch.zeros(128)},
        ]
        model_tensors = {"time_embed": time_embed}
        result = _noise_level_to_embedding(0.5, model_tensors)
        self.assertEqual(result, mock_time_embed.return_value)
        mock_noise_to_t.assert_called_once_with(0.5)
        mock_sinusoids.assert_called_once_with(mock_noise_to_t.return_value, 32)
        mock_time_embed.assert_called_once_with(mock_sinusoids.return_value, time_embed)


class NoiseToTTests(TestCase):
    def test_no_noise(self):
        self.assertEqual(_noise_to_t(0), 0)

    def test_full_noise(self):
        self.assertEqual(_noise_to_t(1), 999)

    def test_half_noise(self):
        self.assertEqual(_noise_to_t(0.5), 354)

    def test_quarter_noise(self):
        self.assertEqual(_noise_to_t(0.25), 202)

    def test_custom_beta_start(self):
        self.assertEqual(_noise_to_t(0.5, beta_start=0.0001), 494)

    def test_custom_beta_end(self):
        self.assertEqual(_noise_to_t(0.5, beta_end=0.02), 307)

    def test_custom_timesteps(self):
        self.assertEqual(_noise_to_t(0.5, timesteps=500), 260)


class TimestepSinusoidsTests(TestCase):
    def test_timestep_zero(self):
        sinusoids = _timestep_sinusoids(0, 4)
        self.assertEqual(sinusoids.tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_timestep_one(self):
        sinusoids = _timestep_sinusoids(1, 4)
        self.assertTrue(
            torch.allclose(
                sinusoids,
                torch.tensor([0.54030, 0.99995, 0.84147, 0.0099998]),
            )
        )

    def test_wider_sinusoids(self):
        sinusoids = _timestep_sinusoids(10, 6)
        self.assertTrue(
            torch.allclose(
                sinusoids,
                torch.tensor(
                    [-0.83907, 0.89420, 0.99977, -0.54402, 0.44767, 0.0215427]
                ),
            )
        )


class TimeEmbedTests(TestCase):
    @patch("pydiffuse.unet.silu")
    @patch("pydiffuse.unet.linear")
    def test_time_embed(self, mock_linear, mock_silu):
        x = Mock(torch.Tensor)
        time_embed = [
            {"weight": "layer 0 weight", "bias": "layer 0 bias"},
            {"weight": "layer 1 weight", "bias": "layer 1 bias"},
            {"weight": "layer 2 weight", "bias": "layer 2 bias"},
        ]
        mock_linear.side_effect = ["linear 0 output", "linear 1 output", "embedding"]
        mock_silu.side_effect = ["silu 0 output", "silu 1 output"]
        result = _time_embed(x, time_embed)
        self.assertEqual(result, "embedding")
        self.assertEqual(
            [call[0] for call in mock_linear.call_args_list],
            [
                ("layer 0 weight", "layer 0 bias", x),
                ("layer 1 weight", "layer 1 bias", "silu 0 output"),
                ("layer 2 weight", "layer 2 bias", "silu 1 output"),
            ],
        )
        self.assertEqual(
            [call[0] for call in mock_silu.call_args_list],
            [("linear 0 output",), ("linear 1 output",)],
        )

    @patch("pydiffuse.unet.silu")
    @patch("pydiffuse.unet.linear")
    def test_time_embed_single_layer(self, mock_linear, mock_silu):
        x = Mock(torch.Tensor)
        time_embed = [{"weight": "layer 0 weight", "bias": "layer 0 bias"}]
        mock_linear.side_effect = ["embedding"]
        result = _time_embed(x, time_embed)
        self.assertEqual(result, "embedding")
        self.assertEqual(
            [call[0] for call in mock_linear.call_args_list],
            [("layer 0 weight", "layer 0 bias", x)],
        )
        mock_silu.assert_not_called()


class CombineChunksTests(TestCase):
    def test_combine_chunks(self):
        conditioning = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )
        combined = _combine_chunks(conditioning)
        self.assertEqual(
            combined.tolist(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        )

    def test_combine_single_chunk(self):
        conditioning = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        combined = _combine_chunks(conditioning)
        self.assertEqual(combined.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class InputBlocksTests(TestCase):
    @patch("pydiffuse.unet._block")
    def test_input_blocks(self, mock_block):
        x = Mock(torch.Tensor)
        time_embedding = Mock(torch.Tensor)
        conditioning = Mock(torch.Tensor)
        input_blocks = [[("block 0", {})], [("block 1", {})], [("block 2", {})]]
        mock_block.side_effect = ["block 0 output", "block 1 output", "block 2 output"]
        result, skips = _input_blocks(x, input_blocks, time_embedding, conditioning)
        self.assertEqual(result, "block 2 output")
        self.assertEqual(skips, ["block 0 output", "block 1 output", "block 2 output"])
        self.assertEqual(
            [call[0] for call in mock_block.call_args_list],
            [
                (x, input_blocks[0], time_embedding, conditioning),
                ("block 0 output", input_blocks[1], time_embedding, conditioning),
                ("block 1 output", input_blocks[2], time_embedding, conditioning),
            ],
        )


class BlockTests(TestCase):
    @patch("pydiffuse.unet.convolution")
    @patch("pydiffuse.unet._resnet_block")
    @patch("pydiffuse.unet._transformer")
    @patch("pydiffuse.unet._upsample")
    def test_block(
        self, mock_upsample, mock_transformer, mock_resnet, mock_convolution
    ):
        x = Mock(torch.Tensor)
        time_embedding = Mock(torch.Tensor)
        conditioning = Mock(torch.Tensor)
        block = [
            ("conv", {"weight": "conv weight", "bias": "conv bias"}),
            ("resnet", {"name": "resnet tensors"}),
            ("transformer", {"name": "transformer tensors"}),
            ("downsample", {"weight": "downsample weight", "bias": "downsample bias"}),
            ("upsample", {"name": "upsample tensors"}),
        ]
        mock_convolution.side_effect = ["conv output", "downsample output"]
        result = _block(x, block, time_embedding, conditioning, (4, 6))
        self.assertEqual(result, mock_upsample.return_value)
        self.assertEqual(
            [call[0] for call in mock_convolution.call_args_list],
            [
                ("conv weight", "conv bias", x),
                ("downsample weight", "downsample bias", mock_transformer.return_value),
            ],
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{"padding": 1}, {"padding": 1, "stride": 2}],
        )
        mock_resnet.assert_called_once_with(
            "conv output", {"name": "resnet tensors"}, time_embedding
        )
        mock_transformer.assert_called_once_with(
            mock_resnet.return_value, {"name": "transformer tensors"}, conditioning
        )
        mock_upsample.assert_called_once_with(
            "downsample output", {"name": "upsample tensors"}, (4, 6)
        )

    @patch("pydiffuse.unet.convolution")
    @patch("pydiffuse.unet._resnet_block")
    @patch("pydiffuse.unet._transformer")
    @patch("pydiffuse.unet._upsample")
    def test_block_without_upsample_size(
        self, mock_upsample, mock_transformer, mock_resnet, mock_convolution
    ):
        x = Mock(torch.Tensor)
        time_embedding = Mock(torch.Tensor)
        conditioning = Mock(torch.Tensor)
        block = [
            ("conv", {"weight": "conv weight", "bias": "conv bias"}),
            ("resnet", {"name": "resnet tensors"}),
            ("transformer", {"name": "transformer tensors"}),
            ("downsample", {"weight": "downsample weight", "bias": "downsample bias"}),
            ("upsample", {"name": "upsample tensors"}),
        ]
        mock_convolution.side_effect = ["conv output", "downsample output"]
        result = _block(x, block, time_embedding, conditioning)
        self.assertEqual(result, mock_upsample.return_value)
        self.assertEqual(
            [call[0] for call in mock_convolution.call_args_list],
            [
                ("conv weight", "conv bias", x),
                ("downsample weight", "downsample bias", mock_transformer.return_value),
            ],
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{"padding": 1}, {"padding": 1, "stride": 2}],
        )
        mock_resnet.assert_called_once_with(
            "conv output", {"name": "resnet tensors"}, time_embedding
        )
        mock_transformer.assert_called_once_with(
            mock_resnet.return_value, {"name": "transformer tensors"}, conditioning
        )
        mock_upsample.assert_called_once_with(
            "downsample output", {"name": "upsample tensors"}, None
        )


class OutputBlocksTests(TestCase):
    @patch("pydiffuse.unet._block")
    def test_output_blocks(self, mock_block):
        x = torch.full((1, 2, 3), 1.0)
        skips = [
            torch.full((1, 4, 6), 10.0),
            torch.full((1, 2, 3), 20.0),
            torch.full((1, 2, 3), 30.0),
        ]
        output_blocks = [[("block 0", {})], [("block 1", {})], [("block 2", {})]]
        time_embedding = Mock(torch.Tensor)
        conditioning = Mock(torch.Tensor)
        mock_block.side_effect = [
            torch.full((1, 2, 3), 2.0),
            torch.full((1, 4, 6), 3.0),
            torch.full((1, 4, 6), 4.0),
        ]
        result = _output_blocks(x, skips, output_blocks, time_embedding, conditioning)
        self.assertTrue(torch.equal(result, torch.full((1, 4, 6), 4.0)))
        self.assertTrue(
            torch.equal(
                mock_block.call_args_list[0][0][0],
                torch.tensor([[[1.0] * 3] * 2, [[30.0] * 3] * 2]),
            )
        )
        self.assertTrue(
            torch.equal(
                mock_block.call_args_list[1][0][0],
                torch.tensor([[[2.0] * 3] * 2, [[20.0] * 3] * 2]),
            )
        )
        self.assertTrue(
            torch.equal(
                mock_block.call_args_list[2][0][0],
                torch.tensor([[[3.0] * 6] * 4, [[10.0] * 6] * 4]),
            )
        )
        self.assertEqual(
            [call[0][1:] for call in mock_block.call_args_list],
            [
                (output_blocks[0], time_embedding, conditioning, (2, 3)),
                (output_blocks[1], time_embedding, conditioning, (4, 6)),
                (output_blocks[2], time_embedding, conditioning, None),
            ],
        )


class OutLayersTests(TestCase):
    @patch("pydiffuse.unet.group_norm")
    @patch("pydiffuse.unet.silu")
    @patch("pydiffuse.unet.convolution")
    def test_out_layers(self, mock_convolution, mock_silu, mock_group_norm):
        x = Mock(torch.Tensor)
        out = {
            "0": {"weight": "norm weight", "bias": "norm bias"},
            "2": {"weight": "conv weight", "bias": "conv bias"},
        }
        result = _out_layers(x, out)
        self.assertEqual(result, mock_convolution.return_value)
        mock_group_norm.assert_called_once_with(
            "norm weight", "norm bias", x, groups=32
        )
        mock_silu.assert_called_once_with(mock_group_norm.return_value)
        mock_convolution.assert_called_once_with(
            "conv weight", "conv bias", mock_silu.return_value, padding=1
        )


class ResnetBlockTests(TestCase):
    @patch("pydiffuse.unet.group_norm")
    @patch("pydiffuse.unet.silu")
    @patch("pydiffuse.unet.convolution")
    @patch("pydiffuse.unet.linear")
    def test_resnet_block(self, mock_linear, mock_convolution, mock_silu, mock_norm):
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]] * 2)
        time_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4])
        block = {
            "in_layers.0": {"weight": "in norm weight", "bias": "in norm bias"},
            "in_layers.2": {"weight": "in conv weight", "bias": "in conv bias"},
            "emb_layers.1": {"weight": "emb weight", "bias": "emb bias"},
            "out_layers.0": {"weight": "out norm weight", "bias": "out norm bias"},
            "out_layers.3": {"weight": "out conv weight", "bias": "out conv bias"},
            "skip_connection": None,
        }
        mock_norm.side_effect = [
            torch.full((2, 2, 3), 10.0),
            torch.full((2, 2, 3), 30.0),
        ]
        mock_silu.side_effect = [
            torch.full((2, 2, 3), 11.0),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.full((2, 2, 3), 31.0),
        ]
        mock_convolution.side_effect = [
            torch.tensor(
                [
                    [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                ]
            ),
            torch.tensor(
                [
                    [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                    [[700.0, 800.0, 900.0], [1000.0, 1100.0, 1200.0]],
                ]
            ),
        ]
        mock_linear.return_value = torch.tensor([1000.0, 2000.0])
        result = _resnet_block(x, block, time_embedding)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [[101.0, 202.0, 303.0], [404.0, 505.0, 606.0]],
                        [[701.0, 802.0, 903.0], [1004.0, 1105.0, 1206.0]],
                    ]
                ),
            )
        )
        self.assertEqual(
            [call[0][:2] for call in mock_norm.call_args_list],
            [("in norm weight", "in norm bias"), ("out norm weight", "out norm bias")],
        )
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][2], x))
        self.assertTrue(
            torch.equal(
                mock_norm.call_args_list[1][0][2],
                torch.tensor(
                    [
                        [[1012.0, 1013.0, 1014.0], [1015.0, 1016.0, 1017.0]],
                        [[2018.0, 2019.0, 2020.0], [2021.0, 2022.0, 2023.0]],
                    ]
                ),
            )
        )
        self.assertEqual(
            [call[1] for call in mock_norm.call_args_list],
            [{"groups": 32}, {"groups": 32}],
        )
        self.assertTrue(
            torch.equal(mock_silu.call_args_list[0][0][0], torch.full((2, 2, 3), 10.0))
        )
        self.assertTrue(torch.equal(mock_silu.call_args_list[1][0][0], time_embedding))
        self.assertTrue(
            torch.equal(mock_silu.call_args_list[2][0][0], torch.full((2, 2, 3), 30.0))
        )
        self.assertEqual(
            mock_linear.call_args_list[0][0][:2], ("emb weight", "emb bias")
        )
        self.assertTrue(
            torch.equal(
                mock_linear.call_args_list[0][0][2], torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
        )
        self.assertEqual(
            [call[0][:2] for call in mock_convolution.call_args_list],
            [("in conv weight", "in conv bias"), ("out conv weight", "out conv bias")],
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2], torch.full((2, 2, 3), 11.0)
            )
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[1][0][2], torch.full((2, 2, 3), 31.0)
            )
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{"padding": 1}, {"padding": 1}],
        )

    @patch("pydiffuse.unet.group_norm")
    @patch("pydiffuse.unet.silu")
    @patch("pydiffuse.unet.convolution")
    @patch("pydiffuse.unet.linear")
    def test_resnet_block_with_skip_connection(
        self, mock_linear, mock_convolution, mock_silu, mock_norm
    ):
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]] * 2)
        time_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4])
        block = {
            "in_layers.0": {"weight": "in norm weight", "bias": "in norm bias"},
            "in_layers.2": {"weight": "in conv weight", "bias": "in conv bias"},
            "emb_layers.1": {"weight": "emb weight", "bias": "emb bias"},
            "out_layers.0": {"weight": "out norm weight", "bias": "out norm bias"},
            "out_layers.3": {"weight": "out conv weight", "bias": "out conv bias"},
            "skip_connection": {"weight": "skip weight", "bias": "skip bias"},
        }
        mock_norm.side_effect = [
            torch.full((2, 2, 3), 10.0),
            torch.full((2, 2, 3), 30.0),
        ]
        mock_silu.side_effect = [
            torch.full((2, 2, 3), 11.0),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.full((2, 2, 3), 31.0),
        ]
        mock_convolution.side_effect = [
            torch.tensor(
                [
                    [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                ]
            ),
            torch.tensor(
                [
                    [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                    [[700.0, 800.0, 900.0], [1000.0, 1100.0, 1200.0]],
                ]
            ),
            torch.full((2, 2, 3), 5.0),
        ]
        mock_linear.return_value = torch.tensor([1000.0, 2000.0])
        result = _resnet_block(x, block, time_embedding)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [[105.0, 205.0, 305.0], [405.0, 505.0, 605.0]],
                        [[705.0, 805.0, 905.0], [1005.0, 1105.0, 1205.0]],
                    ]
                ),
            )
        )
        self.assertEqual(
            [call[0][:2] for call in mock_norm.call_args_list],
            [("in norm weight", "in norm bias"), ("out norm weight", "out norm bias")],
        )
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][2], x))
        self.assertTrue(
            torch.equal(
                mock_norm.call_args_list[1][0][2],
                torch.tensor(
                    [
                        [[1012.0, 1013.0, 1014.0], [1015.0, 1016.0, 1017.0]],
                        [[2018.0, 2019.0, 2020.0], [2021.0, 2022.0, 2023.0]],
                    ]
                ),
            )
        )
        self.assertEqual(
            [call[1] for call in mock_norm.call_args_list],
            [{"groups": 32}, {"groups": 32}],
        )
        self.assertTrue(
            torch.equal(mock_silu.call_args_list[0][0][0], torch.full((2, 2, 3), 10.0))
        )
        self.assertTrue(torch.equal(mock_silu.call_args_list[1][0][0], time_embedding))
        self.assertTrue(
            torch.equal(mock_silu.call_args_list[2][0][0], torch.full((2, 2, 3), 30.0))
        )
        self.assertEqual(
            mock_linear.call_args_list[0][0][:2], ("emb weight", "emb bias")
        )
        self.assertTrue(
            torch.equal(
                mock_linear.call_args_list[0][0][2], torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
        )
        self.assertEqual(
            [call[0][:2] for call in mock_convolution.call_args_list],
            [
                ("in conv weight", "in conv bias"),
                ("out conv weight", "out conv bias"),
                ("skip weight", "skip bias"),
            ],
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2], torch.full((2, 2, 3), 11.0)
            )
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[1][0][2], torch.full((2, 2, 3), 31.0)
            )
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{"padding": 1}, {"padding": 1}, {}],
        )
        self.assertTrue(torch.equal(mock_convolution.call_args_list[2][0][2], x))


class TransformerTests(TestCase):
    @patch("pydiffuse.unet.group_norm")
    @patch("pydiffuse.unet.convolution")
    @patch("pydiffuse.unet._transformer_block")
    def test_transformer(self, mock_transformer_block, mock_convolution, mock_norm):
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.1, 0.2, 0.3]] * 2])
        conditioning = torch.tensor([[13.0, 14.0]])
        block = {
            "norm": {"weight": "norm weight", "bias": "norm bias"},
            "proj_in": {"weight": "proj_in weight", "bias": "proj_in bias"},
            "proj_out": {"weight": "proj_out weight", "bias": "proj_out bias"},
            "transformer_blocks": ["transformer block 1", "transformer block 2"],
        }
        mock_norm.return_value = torch.tensor([[[9.0] * 3] * 2] * 2)
        mock_convolution.side_effect = [
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                ]
            ),
            torch.tensor(
                [
                    [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                    [[700.0, 800.0, 900.0], [1000.0, 1100.0, 1200.0]],
                ]
            ),
        ]
        mock_transformer_block.side_effect = [
            torch.tensor([[0.5, 0.5, 0.5]] * 6),
            torch.tensor(
                [
                    [10.0, 70.0, 130.0],
                    [20.0, 80.0, 140.0],
                    [30.0, 90.0, 150.0],
                    [40.0, 100.0, 160.0],
                    [50.0, 110.0, 170.0],
                    [60.0, 120.0, 180.0],
                ]
            ),
        ]
        result = _transformer(x, block, conditioning)
        self.assertTrue(
            torch.allclose(
                result,
                torch.tensor(
                    [
                        [[101.0, 202.0, 303.0], [404.0, 505.0, 606.0]],
                        [[700.1, 800.2, 900.3], [1000.1, 1100.2, 1200.3]],
                    ]
                ),
            )
        )
        self.assertEqual(
            mock_norm.call_args_list[0][0][:2], ("norm weight", "norm bias")
        )
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][2], x))
        self.assertEqual(mock_norm.call_args_list[0][1], {"groups": 32})
        self.assertEqual(
            [call[0][:2] for call in mock_convolution.call_args_list],
            [("proj_in weight", "proj_in bias"), ("proj_out weight", "proj_out bias")],
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list], [{}, {}]
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2], mock_norm.return_value
            )
        )
        self.assertTrue(
            torch.equal(
                mock_transformer_block.call_args_list[0][0][0],
                torch.tensor(
                    [
                        [1.0, 7.0, 13.0],
                        [2.0, 8.0, 14.0],
                        [3.0, 9.0, 15.0],
                        [4.0, 10.0, 16.0],
                        [5.0, 11.0, 17.0],
                        [6.0, 12.0, 18.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                mock_transformer_block.call_args_list[1][0][0],
                torch.tensor([[0.5, 0.5, 0.5]] * 6),
            )
        )
        self.assertEqual(
            [call[0][1] for call in mock_transformer_block.call_args_list],
            ["transformer block 1", "transformer block 2"],
        )
        for call in mock_transformer_block.call_args_list:
            self.assertTrue(torch.equal(call[0][2], conditioning))
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[1][0][2],
                torch.tensor(
                    [
                        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                        [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
                        [[130.0, 140.0, 150.0], [160.0, 170.0, 180.0]],
                    ]
                ),
            )
        )


class TransformerBlockTests(TestCase):
    @patch("pydiffuse.unet.layer_norm")
    @patch("pydiffuse.unet._attention")
    @patch("pydiffuse.unet._feed_forward")
    def test_transformer_block(self, mock_feed_forward, mock_attention, mock_norm):
        x = torch.tensor([[1.0, 2.0]])
        conditioning = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        block = {
            "norm1": {"weight": "norm1 weight", "bias": "norm1 bias"},
            "norm2": {"weight": "norm2 weight", "bias": "norm2 bias"},
            "norm3": {"weight": "norm3 weight", "bias": "norm3 bias"},
            "attn1": "attn1",
            "attn2": "attn2",
        }
        mock_norm.side_effect = [
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[30.0, 40.0]]),
            torch.tensor([[50.0, 60.0]]),
        ]
        mock_attention.side_effect = [
            torch.tensor([[100.0, 200.0]]),
            torch.tensor([[1000.0, 2000.0]]),
        ]
        mock_feed_forward.return_value = torch.tensor([[10000.0, 20000.0]])
        result = _transformer_block(x, block, conditioning)
        self.assertTrue(torch.equal(result, torch.tensor([[11101.0, 22202.0]])))
        self.assertEqual(
            [call[0][:2] for call in mock_norm.call_args_list],
            [
                ("norm1 weight", "norm1 bias"),
                ("norm2 weight", "norm2 bias"),
                ("norm3 weight", "norm3 bias"),
            ],
        )
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][2], x))
        self.assertTrue(
            torch.equal(
                mock_norm.call_args_list[1][0][2], torch.tensor([[101.0, 202.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                mock_norm.call_args_list[2][0][2], torch.tensor([[1101.0, 2202.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                mock_attention.call_args_list[0][0][0], torch.tensor([[10.0, 20.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                mock_attention.call_args_list[0][0][1], torch.tensor([[10.0, 20.0]])
            )
        )
        self.assertEqual(mock_attention.call_args_list[0][0][2], "attn1")
        self.assertTrue(
            torch.equal(
                mock_attention.call_args_list[1][0][0], torch.tensor([[30.0, 40.0]])
            )
        )
        self.assertTrue(
            torch.equal(mock_attention.call_args_list[1][0][1], conditioning)
        )
        self.assertEqual(mock_attention.call_args_list[1][0][2], "attn2")
        self.assertTrue(
            torch.equal(
                mock_feed_forward.call_args_list[0][0][0],
                torch.tensor([[50.0, 60.0]]),
            )
        )
        self.assertEqual(mock_feed_forward.call_args_list[0][0][1], block)


class AttentionTests(TestCase):
    @patch("pydiffuse.unet.linear")
    def test_attention(self, mock_linear):
        x = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[4.0], [5.0]])
        block = {
            "to_q": {"weight": "q weight", "bias": "q bias"},
            "to_k": {"weight": "k weight", "bias": "k bias"},
            "to_v": {"weight": "v weight", "bias": "v bias"},
            "to_out.0": {"weight": "out weight", "bias": "out bias"},
        }
        values = torch.arange(1.0, 17.0)
        mock_linear.side_effect = [
            torch.tensor(
                [[1.0, 0.0, 0.0, 1.0] * 4, [0.0, 1.0, 1.0, 0.0] * 4, [0.0] * 16]
            ),
            torch.tensor([[1.0, 0.0] * 8, [0.0, 1.0] * 8]),
            torch.stack([values, values + 100.0]),
            torch.tensor([[10.0], [20.0], [30.0]]),
        ]
        result = _attention(x, targets, block)
        self.assertTrue(torch.equal(result, torch.tensor([[10.0], [20.0], [30.0]])))
        self.assertEqual(
            [call[0][:2] for call in mock_linear.call_args_list],
            [
                ("q weight", "q bias"),
                ("k weight", "k bias"),
                ("v weight", "v bias"),
                ("out weight", "out bias"),
            ],
        )
        self.assertTrue(torch.equal(mock_linear.call_args_list[0][0][2], x))
        self.assertTrue(torch.equal(mock_linear.call_args_list[1][0][2], targets))
        self.assertTrue(torch.equal(mock_linear.call_args_list[2][0][2], targets))
        self.assertTrue(
            torch.allclose(
                mock_linear.call_args_list[3][0][2],
                torch.stack([values, values, values])
                + torch.tensor(
                    [
                        [33.02385, 33.02385, 66.97615, 66.97615] * 4,
                        [66.97615, 66.97615, 33.02385, 33.02385] * 4,
                        [50.0] * 16,
                    ]
                ),
            )
        )


class FeedForwardTests(TestCase):
    @patch("pydiffuse.unet.gelu")
    @patch("pydiffuse.unet.linear")
    def test_feed_forward(self, mock_linear, mock_gelu):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        block = {
            "ff.net.0.proj": {"weight": "proj weight", "bias": "proj bias"},
            "ff.net.2": {"weight": "out weight", "bias": "out bias"},
        }
        mock_linear.side_effect = [
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            torch.tensor([[100.0, 200.0], [300.0, 400.0]]),
        ]
        mock_gelu.return_value = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        result = _feed_forward(x, block)
        self.assertTrue(
            torch.equal(result, torch.tensor([[100.0, 200.0], [300.0, 400.0]]))
        )
        self.assertEqual(
            mock_linear.call_args_list[0][0][:2], ("proj weight", "proj bias")
        )
        self.assertTrue(torch.equal(mock_linear.call_args_list[0][0][2], x))
        self.assertTrue(
            torch.equal(
                mock_gelu.call_args_list[0][0][0],
                torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
            )
        )
        self.assertEqual(
            mock_linear.call_args_list[1][0][:2], ("out weight", "out bias")
        )
        self.assertTrue(
            torch.equal(
                mock_linear.call_args_list[1][0][2],
                torch.tensor([[10.0, 40.0], [150.0, 240.0]]),
            )
        )


class UpsampleTests(TestCase):
    @patch("pydiffuse.unet.convolution")
    def test_upsample(self, mock_convolution):
        x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
        conv = {"weight": "conv weight", "bias": "conv bias"}
        result = _upsample(x, conv)
        self.assertEqual(result, mock_convolution.return_value)
        self.assertEqual(
            mock_convolution.call_args_list[0][0][:2],
            ("conv weight", "conv bias"),
        )
        self.assertEqual(
            mock_convolution.call_args_list[0][0][2].tolist(),
            [
                [
                    [1, 1, 2, 2, 3, 3, 4, 4],
                    [1, 1, 2, 2, 3, 3, 4, 4],
                    [5, 5, 6, 6, 7, 7, 8, 8],
                    [5, 5, 6, 6, 7, 7, 8, 8],
                ]
            ],
        )
        self.assertEqual(mock_convolution.call_args_list[0][1], {"padding": 1})

    @patch("pydiffuse.unet.convolution")
    def test_upsample_to_size(self, mock_convolution):
        x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
        conv = {"weight": "conv weight", "bias": "conv bias"}
        result = _upsample(x, conv, (3, 7))
        self.assertEqual(result, mock_convolution.return_value)
        self.assertEqual(
            mock_convolution.call_args_list[0][0][:2],
            ("conv weight", "conv bias"),
        )
        self.assertEqual(
            mock_convolution.call_args_list[0][0][2].tolist(),
            [
                [
                    [1, 1, 2, 2, 3, 3, 4],
                    [1, 1, 2, 2, 3, 3, 4],
                    [5, 5, 6, 6, 7, 7, 8],
                ]
            ],
        )
        self.assertEqual(mock_convolution.call_args_list[0][1], {"padding": 1})
