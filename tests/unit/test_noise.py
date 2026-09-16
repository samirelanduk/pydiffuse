from unittest import TestCase
from unittest.mock import Mock, patch

import safetensors
import torch

from pydiffuse.noise import (
    _noise_level_to_ratio,
    _ratio_to_noise_level,
    _scaling_factors,
    create_noise,
    exponential_schedule,
    karras_schedule,
    noise_tensor,
)


class CreateNoiseTest(TestCase):
    @patch("pydiffuse.noise.get_downscale_ratio")
    @patch("pydiffuse.noise.get_latent_channels")
    @patch("torch.randn")
    def test_can_create_noise(self, mock_randn, mock_channels, mock_ratio):
        mock_ratio.return_value = 4
        mock_channels.return_value = 6
        model = Mock(safetensors.safe_open)
        noise = create_noise(64, 40, model)
        self.assertEqual(noise, mock_randn.return_value)
        mock_ratio.assert_called_once_with(model)
        mock_channels.assert_called_once_with(model)
        mock_randn.assert_called_once_with(6, 10, 16)

    @patch("pydiffuse.noise.get_downscale_ratio")
    @patch("pydiffuse.noise.get_latent_channels")
    @patch("torch.randn")
    def test_can_create_noise_with_uneven_size(
        self, mock_randn, mock_channels, mock_ratio
    ):
        mock_ratio.return_value = 4
        mock_channels.return_value = 6
        model = Mock(safetensors.safe_open)
        noise = create_noise(67, 43, model)
        self.assertEqual(noise, mock_randn.return_value)
        mock_ratio.assert_called_once_with(model)
        mock_channels.assert_called_once_with(model)
        mock_randn.assert_called_once_with(6, 10, 16)


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


class KarrasScheduleTest(TestCase):
    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_karras_schedule(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 128.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = karras_schedule(3)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [call[0] for call in mock_level.call_args_list],
            [(128.0,), (17.0859375,), (1.0,)],
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_karras_schedule_single_step(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 128.0]
        mock_level.side_effect = [0.9]
        schedule = karras_schedule(1)
        self.assertEqual(schedule, [0.9, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual([call[0] for call in mock_level.call_args_list], [(128.0,)])

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_karras_schedule_custom_minimum(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 128.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = karras_schedule(3, noise_level_min=0.1)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.1)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [call[0] for call in mock_level.call_args_list],
            [(128.0,), (17.0859375,), (1.0,)],
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_karras_schedule_custom_maximum(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 128.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = karras_schedule(3, noise_level_max=0.9)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9)
        self.assertEqual(
            [call[0] for call in mock_level.call_args_list],
            [(128.0,), (17.0859375,), (1.0,)],
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_karras_schedule_custom_rho(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 32.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = karras_schedule(3, rho=5)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [call[0] for call in mock_level.call_args_list],
            [(32.0,), (7.59375,), (1.0,)],
        )


class ExponentialScheduleTest(TestCase):
    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_exponential_schedule(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 64.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = exponential_schedule(3)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [round(call[0][0], 5) for call in mock_level.call_args_list],
            [64.0, 8.0, 1.0],
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_exponential_schedule_single_step(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 64.0]
        mock_level.side_effect = [0.9]
        schedule = exponential_schedule(1)
        self.assertEqual(schedule, [0.9, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [round(call[0][0], 5) for call in mock_level.call_args_list], [64.0]
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_exponential_schedule_custom_minimum(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 64.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = exponential_schedule(3, noise_level_min=0.1)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.1)
        mock_ratio.assert_any_call(0.9953399)
        self.assertEqual(
            [round(call[0][0], 5) for call in mock_level.call_args_list],
            [64.0, 8.0, 1.0],
        )

    @patch("pydiffuse.noise._ratio_to_noise_level")
    @patch("pydiffuse.noise._noise_level_to_ratio")
    def test_exponential_schedule_custom_maximum(self, mock_ratio, mock_level):
        mock_ratio.side_effect = [1.0, 64.0]
        mock_level.side_effect = [0.9, 0.5, 0.1]
        schedule = exponential_schedule(3, noise_level_max=0.9)
        self.assertEqual(schedule, [0.9, 0.5, 0.1, 0.0])
        mock_ratio.assert_any_call(0.00085)
        mock_ratio.assert_any_call(0.9)
        self.assertEqual(
            [round(call[0][0], 5) for call in mock_level.call_args_list],
            [64.0, 8.0, 1.0],
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


class NoiseLevelToRatioTest(TestCase):
    @patch("pydiffuse.noise._scaling_factors")
    def test_noise_level_to_ratio(self, mock_factors):
        mock_factors.return_value = (3.0, 4.0)
        self.assertEqual(_noise_level_to_ratio(0.36), 0.75)
        mock_factors.assert_called_once_with(0.36)


class RatioToNoiseLevelTest(TestCase):
    def test_ratio_zero(self):
        self.assertEqual(_ratio_to_noise_level(0.0), 0.0)

    def test_ratio_one(self):
        self.assertEqual(_ratio_to_noise_level(1.0), 0.5)

    def test_ratio_low(self):
        self.assertAlmostEqual(_ratio_to_noise_level(0.029167), 0.00085, places=6)

    def test_ratio_high(self):
        self.assertAlmostEqual(_ratio_to_noise_level(14.614641), 0.9953399, places=6)
