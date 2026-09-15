from unittest import TestCase
from unittest.mock import patch

import torch

from pydiffuse.unet import (
    _feed_forward,
    _noise_to_t,
    _timestep_sinusoids,
    _upsample,
)


class NoiseToTTests(TestCase):
    def test_no_noise(self):
        self.assertEqual(_noise_to_t(0), 0)

    def test_full_noise(self):
        self.assertEqual(_noise_to_t(1), 999)

    def test_half_noise(self):
        self.assertEqual(_noise_to_t(0.5), 354)

    def test_quarter_noise(self):
        self.assertEqual(_noise_to_t(0.25), 202)


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
