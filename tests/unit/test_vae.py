from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import safetensors
import torch
from PIL import Image

from pydiffuse.vae import (
    _attention_block,
    _decode_up,
    _encode_down,
    _get_decode_tensors,
    _get_encode_tensors,
    _get_layer,
    _get_layers,
    _get_mid_tensors,
    _get_numbers,
    _image_to_tensor,
    _mid_blocks,
    _out_layers,
    _resnet_block,
    _tensor_to_image,
    _upsample,
    decode,
    encode,
    get_downscale_ratio,
    get_latent_channels,
)


class EncodeTests(TestCase):
    @patch("pydiffuse.vae._get_encode_tensors")
    @patch("pydiffuse.vae.get_downscale_ratio")
    @patch("pydiffuse.vae._image_to_tensor")
    @patch("pydiffuse.vae.convolution")
    @patch("pydiffuse.vae._encode_down")
    @patch("pydiffuse.vae._mid_blocks")
    @patch("pydiffuse.vae._out_layers")
    def test_encode(
        self,
        mock_out,
        mock_mid,
        mock_down,
        mock_convolution,
        mock_to_tensor,
        mock_ratio,
        mock_tensors,
    ):
        mock_tensors.return_value = {
            "conv_in": {"weight": "conv_in weight", "bias": "conv_in bias"},
            "down": "down",
            "mid": "mid",
            "norm_out": "norm_out",
            "conv_out": "conv_out",
            "quant_conv": {"weight": "quant weight", "bias": "quant bias"},
        }
        mock_convolution.return_value.shape = (8, 4, 4)
        image = Mock(Image.Image)
        model = Mock(safetensors.safe_open)
        latent = encode(image, model)
        self.assertEqual(latent, mock_convolution.return_value.narrow.return_value)
        mock_tensors.assert_called_once_with(model)
        mock_ratio.assert_called_once_with(model)
        mock_to_tensor.assert_called_once_with(image, mock_ratio.return_value)
        self.assertEqual(
            [call[0] for call in mock_convolution.call_args_list],
            [
                ("conv_in weight", "conv_in bias", mock_to_tensor.return_value),
                ("quant weight", "quant bias", mock_out.return_value),
            ],
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{"padding": 1}, {}],
        )
        mock_down.assert_called_once_with(mock_convolution.return_value, "down")
        mock_mid.assert_called_once_with(mock_down.return_value, "mid")
        mock_out.assert_called_once_with(mock_mid.return_value, "norm_out", "conv_out")
        mock_convolution.return_value.narrow.assert_called_once_with(0, 0, 4)


class DecodeTests(TestCase):
    @patch("pydiffuse.vae._get_decode_tensors")
    @patch("pydiffuse.vae.convolution")
    @patch("pydiffuse.vae._mid_blocks")
    @patch("pydiffuse.vae._decode_up")
    @patch("pydiffuse.vae._out_layers")
    @patch("pydiffuse.vae._tensor_to_image")
    def test_decode(
        self,
        mock_to_image,
        mock_out,
        mock_up,
        mock_mid,
        mock_convolution,
        mock_tensors,
    ):
        mock_tensors.return_value = {
            "post_quant_conv": {"weight": "post weight", "bias": "post bias"},
            "conv_in": {"weight": "conv_in weight", "bias": "conv_in bias"},
            "mid": "mid",
            "up": "up",
            "norm_out": "norm_out",
            "conv_out": "conv_out",
        }
        latent = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        image = decode(latent, model)
        self.assertEqual(image, mock_to_image.return_value)
        mock_tensors.assert_called_once_with(model)
        self.assertEqual(
            [call[0] for call in mock_convolution.call_args_list],
            [
                ("post weight", "post bias", latent),
                ("conv_in weight", "conv_in bias", mock_convolution.return_value),
            ],
        )
        self.assertEqual(
            [call[1] for call in mock_convolution.call_args_list],
            [{}, {"padding": 1}],
        )
        mock_mid.assert_called_once_with(mock_convolution.return_value, "mid")
        mock_up.assert_called_once_with(mock_mid.return_value, "up")
        mock_out.assert_called_once_with(mock_up.return_value, "norm_out", "conv_out")
        mock_to_image.assert_called_once_with(mock_out.return_value)


class GetDownscaleRatioTests(TestCase):
    @patch("pydiffuse.vae._get_numbers")
    def test_get_downscale_ratio(self, mock_numbers):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.down.0.downsample.conv.weight",
            "first_stage_model.encoder.down.0.downsample.conv.bias",
            "first_stage_model.encoder.down.1.block.0.conv1.weight",
            "first_stage_model.encoder.down.2.downsample.conv.weight",
            "first_stage_model.encoder.down.3.downsample.conv.weight",
            "first_stage_model.decoder.up.4.upsample.conv.weight",
        ]
        mock_numbers.return_value = [0, 1, 2, 3, 4]
        self.assertEqual(get_downscale_ratio(model), 8)
        mock_numbers.assert_called_once_with(model, "first_stage_model.encoder.down")

    @patch("pydiffuse.vae._get_numbers")
    def test_get_downscale_ratio_with_no_downsampling(self, mock_numbers):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.down.0.block.0.conv1.weight",
            "first_stage_model.encoder.down.1.block.0.conv1.weight",
            "first_stage_model.decoder.up.1.upsample.conv.weight",
        ]
        mock_numbers.return_value = [0, 1]
        self.assertEqual(get_downscale_ratio(model), 1)
        mock_numbers.assert_called_once_with(model, "first_stage_model.encoder.down")


class GetLatentChannelsTests(TestCase):
    def test_get_latent_channels(self):
        model = MagicMock()
        model.get_tensor.return_value = torch.zeros(4, 6, 1, 1)
        self.assertEqual(get_latent_channels(model), 6)
        model.get_tensor.assert_called_once_with(
            "first_stage_model.post_quant_conv.weight"
        )


class GetEncodeTensorsTests(TestCase):
    @patch("pydiffuse.vae._get_numbers")
    @patch("pydiffuse.vae._get_layers")
    @patch("pydiffuse.vae._get_layer")
    @patch("pydiffuse.vae._get_mid_tensors")
    def test_get_encode_tensors(
        self,
        mock_mid,
        mock_layer,
        mock_layers,
        mock_numbers,
    ):
        mock_numbers.side_effect = lambda model, prefix: (
            [int(prefix.split(".")[-2])] if prefix.endswith(".block") else [0, 1]
        )
        mock_layers.side_effect = lambda model, prefix, names: f"layers {prefix}"
        mock_layer.side_effect = lambda model, prefix: f"layer {prefix}"
        model = Mock(safetensors.safe_open)
        tensors = _get_encode_tensors(model)
        self.assertEqual(
            [call[0] for call in mock_numbers.call_args_list],
            [
                (model, "first_stage_model.encoder.down"),
                (model, "first_stage_model.encoder.down.0.block"),
                (model, "first_stage_model.encoder.down.1.block"),
            ],
        )
        self.assertEqual(
            [call[0] for call in mock_layers.call_args_list],
            [
                (
                    model,
                    "first_stage_model.encoder.down.0.block.0",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
                (
                    model,
                    "first_stage_model.encoder.down.1.block.1",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
            ],
        )
        mock_mid.assert_called_once_with(model, "first_stage_model.encoder")
        self.assertEqual(
            tensors,
            {
                "conv_in": "layer first_stage_model.encoder.conv_in",
                "down": [
                    {
                        "block": ["layers first_stage_model.encoder.down.0.block.0"],
                        "downsample": (
                            "layer first_stage_model.encoder.down.0.downsample.conv"
                        ),
                    },
                    {
                        "block": ["layers first_stage_model.encoder.down.1.block.1"],
                        "downsample": (
                            "layer first_stage_model.encoder.down.1.downsample.conv"
                        ),
                    },
                ],
                "mid": mock_mid.return_value,
                "norm_out": "layer first_stage_model.encoder.norm_out",
                "conv_out": "layer first_stage_model.encoder.conv_out",
                "quant_conv": "layer first_stage_model.quant_conv",
            },
        )


class GetDecodeTensorsTests(TestCase):
    @patch("pydiffuse.vae._get_numbers")
    @patch("pydiffuse.vae._get_layers")
    @patch("pydiffuse.vae._get_layer")
    @patch("pydiffuse.vae._get_mid_tensors")
    def test_get_decode_tensors(
        self,
        mock_mid,
        mock_layer,
        mock_layers,
        mock_numbers,
    ):
        mock_numbers.side_effect = lambda model, prefix: (
            [int(prefix.split(".")[-2])] if prefix.endswith(".block") else [0, 1]
        )
        mock_layers.side_effect = lambda model, prefix, names: f"layers {prefix}"
        mock_layer.side_effect = lambda model, prefix: f"layer {prefix}"
        model = Mock(safetensors.safe_open)
        tensors = _get_decode_tensors(model)
        self.assertEqual(
            tensors,
            {
                "post_quant_conv": "layer first_stage_model.post_quant_conv",
                "conv_in": "layer first_stage_model.decoder.conv_in",
                "mid": mock_mid.return_value,
                "up": [
                    {
                        "block": ["layers first_stage_model.decoder.up.0.block.0"],
                        "upsample": (
                            "layer first_stage_model.decoder.up.0.upsample.conv"
                        ),
                    },
                    {
                        "block": ["layers first_stage_model.decoder.up.1.block.1"],
                        "upsample": (
                            "layer first_stage_model.decoder.up.1.upsample.conv"
                        ),
                    },
                ],
                "norm_out": "layer first_stage_model.decoder.norm_out",
                "conv_out": "layer first_stage_model.decoder.conv_out",
            },
        )
        self.assertEqual(
            [call[0] for call in mock_numbers.call_args_list],
            [
                (model, "first_stage_model.decoder.up"),
                (model, "first_stage_model.decoder.up.0.block"),
                (model, "first_stage_model.decoder.up.1.block"),
            ],
        )
        self.assertEqual(
            [call[0] for call in mock_layers.call_args_list],
            [
                (
                    model,
                    "first_stage_model.decoder.up.0.block.0",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
                (
                    model,
                    "first_stage_model.decoder.up.1.block.1",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
            ],
        )
        mock_mid.assert_called_once_with(model, "first_stage_model.decoder")


class GetNumbersTests(TestCase):
    def test_get_numbers(self):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.down.0.block.0.conv1.weight",
            "first_stage_model.encoder.down.0.downsample.conv.weight",
            "first_stage_model.encoder.down.2.block.1.conv2.bias",
            "first_stage_model.encoder.down.10.block.0.norm1.weight",
            "first_stage_model.encoder.mid.block_1.conv1.weight",
            "first_stage_model.encoder.down.x.block.0.conv1.weight",
            "first_stage_model.decoder.up.1.block.0.conv1.weight",
            "xxx",
        ]
        result = _get_numbers(model, "first_stage_model.encoder.down")
        self.assertEqual(result, [0, 2, 10])

    def test_get_numbers_of_blocks(self):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.down.1.block.0.conv1.weight",
            "first_stage_model.encoder.down.1.block.2.norm1.bias",
            "first_stage_model.encoder.down.1.downsample.conv.weight",
            "first_stage_model.encoder.down.0.block.5.conv1.weight",
            "first_stage_model.encoder.down.1.block.x.conv1.weight",
            "first_stage_model.decoder.up.1.block.3.conv1.weight",
            "xxx",
        ]
        result = _get_numbers(model, "first_stage_model.encoder.down.1.block")
        self.assertEqual(result, [0, 2])


class GetLayersTests(TestCase):
    @patch("pydiffuse.vae._get_layer")
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


class GetLayerTests(TestCase):
    def test_get_layer(self):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_in.bias",
            "first_stage_model.encoder.norm_out.weight",
            "first_stage_model.decoder.conv_in.weight",
            "first_stage_model.decoder.conv_in.bias",
            "xxx",
        ]
        model.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        layer = _get_layer(model, "first_stage_model.encoder.conv_in") or {}
        for key in layer:
            layer[key] = layer[key].tolist()
        self.assertEqual(
            layer,
            {
                "weight": [40, 80, 120],
                "bias": [38, 76, 114],
            },
        )

    def test_get_layer_weight_not_in_model(self):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_in.bias",
            "first_stage_model.encoder.norm_out.weight",
            "first_stage_model.encoder.conv_out.bias",
            "xxx",
        ]
        model.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        self.assertIsNone(_get_layer(model, "first_stage_model.encoder.conv_out"))

    def test_get_layer_bias_not_in_model(self):
        model = MagicMock()
        model.keys.return_value = [
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_in.bias",
            "first_stage_model.encoder.norm_out.weight",
            "first_stage_model.encoder.conv_out.weight",
            "xxx",
        ]
        model.get_tensor.side_effect = lambda key: torch.tensor(
            [len(key), len(key) * 2, len(key) * 3]
        )
        self.assertIsNone(_get_layer(model, "first_stage_model.encoder.conv_out"))


class GetMidTensorsTests(TestCase):
    @patch("pydiffuse.vae._get_layers")
    def test_get_mid_tensors(self, mock_layers):
        mock_layers.side_effect = lambda model, prefix, names: f"layers {prefix}"
        model = Mock(safetensors.safe_open)
        mid = _get_mid_tensors(model, "first_stage_model.encoder")
        self.assertEqual(
            mid,
            {
                "block_1": "layers first_stage_model.encoder.mid.block_1",
                "attn_1": "layers first_stage_model.encoder.mid.attn_1",
                "block_2": "layers first_stage_model.encoder.mid.block_2",
            },
        )
        self.assertEqual(
            [call[0] for call in mock_layers.call_args_list],
            [
                (
                    model,
                    "first_stage_model.encoder.mid.block_1",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
                (
                    model,
                    "first_stage_model.encoder.mid.attn_1",
                    ("norm", "q", "k", "v", "proj_out"),
                ),
                (
                    model,
                    "first_stage_model.encoder.mid.block_2",
                    ("norm1", "conv1", "norm2", "conv2", "nin_shortcut"),
                ),
            ],
        )


class ImageToTensorTests(TestCase):
    def test_image_to_tensor(self):
        image = Image.new("RGB", (2, 2))
        image.putdata([(0, 0, 0), (255, 255, 255), (0, 255, 0), (255, 0, 255)])
        result = _image_to_tensor(image, 1)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [[-1.0, 1.0], [-1.0, 1.0]],
                        [[-1.0, 1.0], [1.0, -1.0]],
                        [[-1.0, 1.0], [-1.0, 1.0]],
                    ]
                ),
            )
        )

    def test_image_to_tensor_crops_both_dimensions(self):
        image = Image.new("RGB", (6, 10))
        values = [255, 0, 51, 204, 255, 0, 51, 204, 255, 0]
        image.putdata([(values[x], values[y], 0) for y in range(10) for x in range(6)])
        result = _image_to_tensor(image, 4)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                            [-1.0, -0.6, 0.6, 1.0],
                        ],
                        [
                            [-1.0, -1.0, -1.0, -1.0],
                            [-0.6, -0.6, -0.6, -0.6],
                            [0.6, 0.6, 0.6, 0.6],
                            [1.0, 1.0, 1.0, 1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-0.6, -0.6, -0.6, -0.6],
                            [0.6, 0.6, 0.6, 0.6],
                            [1.0, 1.0, 1.0, 1.0],
                        ],
                        [
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
            )
        )

    def test_image_to_tensor_converts_to_rgb(self):
        image = Image.new("L", (2, 2))
        image.putdata([0, 255, 0, 255])
        result = _image_to_tensor(image, 1)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [[-1.0, 1.0], [-1.0, 1.0]],
                        [[-1.0, 1.0], [-1.0, 1.0]],
                        [[-1.0, 1.0], [-1.0, 1.0]],
                    ]
                ),
            )
        )


class TensorToImageTests(TestCase):
    def test_tensor_to_image(self):
        x = torch.tensor(
            [
                [[-1.0, 0.0], [1.0, 2.0]],
                [[-2.0, 1.0], [0.0, -1.0]],
                [[0.0, -1.0], [1.0, 0.0]],
            ]
        )
        image = _tensor_to_image(x)
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 128))
        self.assertEqual(image.getpixel((1, 0)), (128, 255, 0))
        self.assertEqual(image.getpixel((0, 1)), (255, 128, 255))
        self.assertEqual(image.getpixel((1, 1)), (255, 0, 128))


class EncodeDownTests(TestCase):
    @patch("pydiffuse.vae._resnet_block")
    @patch("pydiffuse.vae.convolution")
    def test_encode_down(self, mock_convolution, mock_resnet):
        down = [
            {
                "block": ["level 0 block 0", "level 0 block 1"],
                "downsample": {
                    "weight": "downsample weight",
                    "bias": "downsample bias",
                },
            },
            {"block": ["level 1 block 0"], "downsample": None},
        ]
        x = Mock(torch.Tensor)
        result = _encode_down(x, down)
        self.assertEqual(result, mock_resnet.return_value)
        self.assertEqual(
            [call[0] for call in mock_resnet.call_args_list],
            [
                (x, "level 0 block 0"),
                (mock_resnet.return_value, "level 0 block 1"),
                (mock_convolution.return_value, "level 1 block 0"),
            ],
        )
        mock_convolution.assert_called_once_with(
            "downsample weight",
            "downsample bias",
            mock_resnet.return_value,
            padding=1,
            stride=2,
            pad_at_end=True,
        )


class DecodeUpTests(TestCase):
    @patch("pydiffuse.vae._resnet_block")
    @patch("pydiffuse.vae._upsample")
    def test_decode_up(self, mock_upsample, mock_resnet):
        up = [
            {"block": ["level 0 block 0"], "upsample": None},
            {
                "block": ["level 1 block 0", "level 1 block 1"],
                "upsample": "level 1 upsample",
            },
        ]
        x = Mock(torch.Tensor)
        result = _decode_up(x, up)
        self.assertEqual(result, mock_resnet.return_value)
        self.assertEqual(
            [call[0] for call in mock_resnet.call_args_list],
            [
                (x, "level 1 block 0"),
                (mock_resnet.return_value, "level 1 block 1"),
                (mock_upsample.return_value, "level 0 block 0"),
            ],
        )
        mock_upsample.assert_called_once_with(
            mock_resnet.return_value, "level 1 upsample"
        )


class ResnetBlockTests(TestCase):
    @patch("pydiffuse.vae.group_norm")
    @patch("pydiffuse.vae.silu")
    @patch("pydiffuse.vae.convolution")
    def test_resnet_block(self, mock_convolution, mock_silu, mock_group_norm):
        x = torch.tensor([[[1.0, 2.0]]])
        block = {
            "norm1": {"weight": "norm1 weight", "bias": "norm1 bias"},
            "conv1": {"weight": "conv1 weight", "bias": "conv1 bias"},
            "norm2": {"weight": "norm2 weight", "bias": "norm2 bias"},
            "conv2": {"weight": "conv2 weight", "bias": "conv2 bias"},
            "nin_shortcut": None,
        }
        mock_group_norm.side_effect = [
            torch.tensor([[[10.0, 20.0]]]),
            torch.tensor([[[30.0, 40.0]]]),
        ]
        mock_silu.side_effect = [
            torch.tensor([[[11.0, 21.0]]]),
            torch.tensor([[[31.0, 41.0]]]),
        ]
        mock_convolution.side_effect = [
            torch.tensor([[[12.0, 22.0]]]),
            torch.tensor([[[100.0, 200.0]]]),
        ]
        result = _resnet_block(x, block)
        self.assertTrue(torch.equal(result, torch.tensor([[[101.0, 202.0]]])))
        self.assertEqual(
            mock_group_norm.call_args_list[0][0][:2], ("norm1 weight", "norm1 bias")
        )
        self.assertTrue(torch.equal(mock_group_norm.call_args_list[0][0][2], x))
        self.assertEqual(mock_group_norm.call_args_list[0][1], {"groups": 32})
        self.assertEqual(
            mock_group_norm.call_args_list[1][0][:2], ("norm2 weight", "norm2 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_group_norm.call_args_list[1][0][2],
                torch.tensor([[[12.0, 22.0]]]),
            )
        )
        self.assertEqual(mock_group_norm.call_args_list[1][1], {"groups": 32})
        self.assertTrue(
            torch.equal(
                mock_silu.call_args_list[0][0][0], torch.tensor([[[10.0, 20.0]]])
            )
        )
        self.assertTrue(
            torch.equal(
                mock_silu.call_args_list[1][0][0], torch.tensor([[[30.0, 40.0]]])
            )
        )
        self.assertEqual(
            mock_convolution.call_args_list[0][0][:2], ("conv1 weight", "conv1 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2],
                torch.tensor([[[11.0, 21.0]]]),
            )
        )
        self.assertEqual(mock_convolution.call_args_list[0][1], {"padding": 1})
        self.assertEqual(
            mock_convolution.call_args_list[1][0][:2], ("conv2 weight", "conv2 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[1][0][2],
                torch.tensor([[[31.0, 41.0]]]),
            )
        )
        self.assertEqual(mock_convolution.call_args_list[1][1], {"padding": 1})
        self.assertEqual(mock_convolution.call_count, 2)

    @patch("pydiffuse.vae.group_norm")
    @patch("pydiffuse.vae.silu")
    @patch("pydiffuse.vae.convolution")
    def test_resnet_block_with_shortcut(
        self, mock_convolution, mock_silu, mock_group_norm
    ):
        x = torch.tensor([[[1.0, 2.0]]])
        block = {
            "norm1": {"weight": "norm1 weight", "bias": "norm1 bias"},
            "conv1": {"weight": "conv1 weight", "bias": "conv1 bias"},
            "norm2": {"weight": "norm2 weight", "bias": "norm2 bias"},
            "conv2": {"weight": "conv2 weight", "bias": "conv2 bias"},
            "nin_shortcut": {"weight": "shortcut weight", "bias": "shortcut bias"},
        }
        mock_group_norm.side_effect = [
            torch.tensor([[[10.0, 20.0]]]),
            torch.tensor([[[30.0, 40.0]]]),
        ]
        mock_silu.side_effect = [
            torch.tensor([[[11.0, 21.0]]]),
            torch.tensor([[[31.0, 41.0]]]),
        ]
        mock_convolution.side_effect = [
            torch.tensor([[[12.0, 22.0]]]),
            torch.tensor([[[100.0, 200.0]]]),
            torch.tensor([[[5.0, 6.0]]]),
        ]
        result = _resnet_block(x, block)
        self.assertTrue(torch.equal(result, torch.tensor([[[105.0, 206.0]]])))
        self.assertEqual(
            mock_group_norm.call_args_list[0][0][:2], ("norm1 weight", "norm1 bias")
        )
        self.assertTrue(torch.equal(mock_group_norm.call_args_list[0][0][2], x))
        self.assertEqual(mock_group_norm.call_args_list[0][1], {"groups": 32})
        self.assertEqual(
            mock_group_norm.call_args_list[1][0][:2], ("norm2 weight", "norm2 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_group_norm.call_args_list[1][0][2],
                torch.tensor([[[12.0, 22.0]]]),
            )
        )
        self.assertEqual(mock_group_norm.call_args_list[1][1], {"groups": 32})
        self.assertTrue(
            torch.equal(
                mock_silu.call_args_list[0][0][0], torch.tensor([[[10.0, 20.0]]])
            )
        )
        self.assertTrue(
            torch.equal(
                mock_silu.call_args_list[1][0][0], torch.tensor([[[30.0, 40.0]]])
            )
        )
        self.assertEqual(
            mock_convolution.call_args_list[0][0][:2], ("conv1 weight", "conv1 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2],
                torch.tensor([[[11.0, 21.0]]]),
            )
        )
        self.assertEqual(mock_convolution.call_args_list[0][1], {"padding": 1})
        self.assertEqual(
            mock_convolution.call_args_list[1][0][:2], ("conv2 weight", "conv2 bias")
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[1][0][2],
                torch.tensor([[[31.0, 41.0]]]),
            )
        )
        self.assertEqual(mock_convolution.call_args_list[1][1], {"padding": 1})
        self.assertEqual(
            mock_convolution.call_args_list[2][0][:2],
            ("shortcut weight", "shortcut bias"),
        )
        self.assertTrue(torch.equal(mock_convolution.call_args_list[2][0][2], x))
        self.assertEqual(mock_convolution.call_args_list[2][1], {})
        self.assertEqual(mock_convolution.call_count, 3)


class MidBlocksTests(TestCase):
    @patch("pydiffuse.vae._resnet_block")
    @patch("pydiffuse.vae._attention_block")
    def test_mid_blocks(self, mock_attention, mock_resnet):
        mid = {"block_1": "block_1", "attn_1": "attn_1", "block_2": "block_2"}
        x = Mock(torch.Tensor)
        result = _mid_blocks(x, mid)
        self.assertEqual(result, mock_resnet.return_value)
        self.assertEqual(
            [call[0] for call in mock_resnet.call_args_list],
            [(x, "block_1"), (mock_attention.return_value, "block_2")],
        )
        mock_attention.assert_called_once_with(mock_resnet.return_value, "attn_1")


class AttentionBlockTests(TestCase):
    @patch("pydiffuse.vae.group_norm")
    @patch("pydiffuse.vae.convolution")
    def test_attention_block(self, mock_convolution, mock_group_norm):
        x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        block = {
            "norm": {"weight": "norm weight", "bias": "norm bias"},
            "q": {"weight": "q weight", "bias": "q bias"},
            "k": {"weight": "k weight", "bias": "k bias"},
            "v": {"weight": "v weight", "bias": "v bias"},
            "proj_out": {"weight": "proj weight", "bias": "proj bias"},
        }
        mock_group_norm.return_value = torch.tensor([[[9.0, 8.0]], [[7.0, 6.0]]])
        mock_convolution.side_effect = [
            torch.zeros(2, 1, 2),
            torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
            torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
            torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]]]),
        ]
        result = _attention_block(x, block)
        self.assertTrue(
            torch.equal(result, torch.tensor([[[11.0, 22.0]], [[33.0, 44.0]]]))
        )
        self.assertEqual(
            mock_group_norm.call_args_list[0][0][:2], ("norm weight", "norm bias")
        )
        self.assertTrue(torch.equal(mock_group_norm.call_args_list[0][0][2], x))
        self.assertEqual(mock_group_norm.call_args_list[0][1], {"groups": 32})
        self.assertEqual(
            [call[0][:2] for call in mock_convolution.call_args_list],
            [
                ("q weight", "q bias"),
                ("k weight", "k bias"),
                ("v weight", "v bias"),
                ("proj weight", "proj bias"),
            ],
        )
        for index in range(3):
            self.assertTrue(
                torch.equal(
                    mock_convolution.call_args_list[index][0][2],
                    torch.tensor([[[9.0, 8.0]], [[7.0, 6.0]]]),
                )
            )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[3][0][2],
                torch.tensor([[[1.5, 1.5]], [[3.5, 3.5]]]),
            )
        )


class OutLayersTests(TestCase):
    @patch("pydiffuse.vae.group_norm")
    @patch("pydiffuse.vae.silu")
    @patch("pydiffuse.vae.convolution")
    def test_out_layers(self, mock_convolution, mock_silu, mock_group_norm):
        x = Mock(torch.Tensor)
        norm_out = {"weight": "norm weight", "bias": "norm bias"}
        conv_out = {"weight": "conv weight", "bias": "conv bias"}
        result = _out_layers(x, norm_out, conv_out)
        self.assertEqual(result, mock_convolution.return_value)
        mock_group_norm.assert_called_once_with(
            "norm weight", "norm bias", x, groups=32
        )
        mock_silu.assert_called_once_with(mock_group_norm.return_value)
        mock_convolution.assert_called_once_with(
            "conv weight", "conv bias", mock_silu.return_value, padding=1
        )


class UpsampleTests(TestCase):
    @patch("pydiffuse.vae.convolution")
    def test_upsample(self, mock_convolution):
        x = torch.tensor([[[1.0, 2.0]]])
        upsample = {"weight": "upsample weight", "bias": "upsample bias"}
        result = _upsample(x, upsample)
        self.assertEqual(result, mock_convolution.return_value)
        self.assertEqual(
            mock_convolution.call_args_list[0][0][:2],
            ("upsample weight", "upsample bias"),
        )
        self.assertTrue(
            torch.equal(
                mock_convolution.call_args_list[0][0][2],
                torch.tensor([[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]]),
            )
        )
        self.assertEqual(mock_convolution.call_args_list[0][1], {"padding": 1})
