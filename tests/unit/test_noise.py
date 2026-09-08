from unittest import TestCase
from unittest.mock import patch

import torch

from pydiffuse.noise import _scaling_factors, noise_tensor


class NoiseTensorTest(TestCase):
    @patch("pydiffuse.noise._scaling_factors")
    @patch("torch.randn_like")
    def test_can_noise_tensor(self, mock_randn_like, mock_scaling_factors):
        mock_scaling_factors.return_value = (0.25, 0.75)
        mock_randn_like.return_value = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        image = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        noised = noise_tensor(image, 0.2)
        mock_randn_like.assert_called_once_with(image)
        mock_scaling_factors.assert_called_once_with(0.2)
        self.assertTrue(
            torch.allclose(
                noised,
                torch.tensor([[0.775, 1.55, 2.325], [3.1, 3.875, 4.65]]),
            )
        )


class ScalingFactorsTest(TestCase):
    def test_scaling_factor_zero(self):
        scale_noise, scale_original = _scaling_factors(0.0)
        self.assertEqual(scale_noise, 0.0)
        self.assertEqual(scale_original, 1.0)

    def test_scaling_factor_one(self):
        scale_noise, scale_original = _scaling_factors(1.0)
        self.assertEqual(scale_noise, 1.0)
        self.assertEqual(scale_original, 0.0)

    def test_scaling_factor_half(self):
        scale_noise, scale_original = _scaling_factors(0.5)
        self.assertAlmostEqual(scale_noise, 0.7071, places=4)
        self.assertAlmostEqual(scale_original, 0.7071, places=4)
