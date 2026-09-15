from unittest import TestCase
from unittest.mock import patch

import torch

from pydiffuse.unet import (
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
