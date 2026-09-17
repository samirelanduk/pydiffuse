from unittest import TestCase
from unittest.mock import Mock, patch

import safetensors
import torch

from pydiffuse.sample import _predict_noise, _remove_noise, sample_euler, sample_heun


class SampleEulerTests(TestCase):
    @patch("pydiffuse.sample._predict_noise")
    @patch("pydiffuse.sample._remove_noise")
    def test_sample_euler(self, mock_remove_noise, mock_predict_noise):
        positive = Mock(torch.Tensor)
        negative = Mock(torch.Tensor)
        latent = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        mock_predict_noise.side_effect = ["noise 0", "noise 1"]
        mock_remove_noise.side_effect = ["latent 1", "latent 2"]
        result = sample_euler(positive, negative, latent, model, [0.9, 0.5, 0.0])
        self.assertEqual(result, "latent 2")
        self.assertEqual(
            [call[0] for call in mock_predict_noise.call_args_list],
            [
                (latent, 0.9, positive, negative, model, 1.0),
                ("latent 1", 0.5, positive, negative, model, 1.0),
            ],
        )
        self.assertEqual(
            [call[0] for call in mock_remove_noise.call_args_list],
            [
                (latent, "noise 0", 0.9, 0.5),
                ("latent 1", "noise 1", 0.5, 0.0),
            ],
        )

    @patch("pydiffuse.sample._predict_noise")
    @patch("pydiffuse.sample._remove_noise")
    def test_sample_euler_with_cfg(self, mock_remove_noise, mock_predict_noise):
        positive = Mock(torch.Tensor)
        negative = Mock(torch.Tensor)
        latent = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        mock_predict_noise.side_effect = ["noise 0", "noise 1"]
        mock_remove_noise.side_effect = ["latent 1", "latent 2"]
        result = sample_euler(
            positive, negative, latent, model, [0.9, 0.5, 0.0], cfg=7.0
        )
        self.assertEqual(result, "latent 2")
        self.assertEqual(
            [call[0] for call in mock_predict_noise.call_args_list],
            [
                (latent, 0.9, positive, negative, model, 7.0),
                ("latent 1", 0.5, positive, negative, model, 7.0),
            ],
        )
        self.assertEqual(
            [call[0] for call in mock_remove_noise.call_args_list],
            [
                (latent, "noise 0", 0.9, 0.5),
                ("latent 1", "noise 1", 0.5, 0.0),
            ],
        )


class SampleHeunTests(TestCase):
    @patch("pydiffuse.sample._predict_noise")
    @patch("pydiffuse.sample._remove_noise")
    def test_sample_heun(self, mock_remove_noise, mock_predict_noise):
        positive = Mock(torch.Tensor)
        negative = Mock(torch.Tensor)
        latent = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        mock_predict_noise.side_effect = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 6.0]),
            torch.tensor([5.0, 5.0]),
        ]
        mock_remove_noise.side_effect = ["trial 1", "latent 1", "latent 2"]
        result = sample_heun(positive, negative, latent, model, [0.9, 0.5, 0.0])
        self.assertEqual(result, "latent 2")
        self.assertEqual(
            [call[0][0:2] for call in mock_predict_noise.call_args_list],
            [(latent, 0.9), ("trial 1", 0.5), ("latent 1", 0.5)],
        )
        for call in mock_predict_noise.call_args_list:
            self.assertEqual(call[0][2:], (positive, negative, model, 1.0))
        self.assertEqual(
            [call[0][0] for call in mock_remove_noise.call_args_list],
            [latent, latent, "latent 1"],
        )
        self.assertEqual(
            [call[0][2:] for call in mock_remove_noise.call_args_list],
            [(0.9, 0.5), (0.9, 0.5), (0.5, 0.0)],
        )
        self.assertEqual(
            [call[0][1].tolist() for call in mock_remove_noise.call_args_list],
            [[1.0, 2.0], [2.0, 4.0], [5.0, 5.0]],
        )

    @patch("pydiffuse.sample._predict_noise")
    @patch("pydiffuse.sample._remove_noise")
    def test_sample_heun_with_cfg(self, mock_remove_noise, mock_predict_noise):
        positive = Mock(torch.Tensor)
        negative = Mock(torch.Tensor)
        latent = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        mock_predict_noise.side_effect = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 6.0]),
            torch.tensor([5.0, 5.0]),
        ]
        mock_remove_noise.side_effect = ["trial 1", "latent 1", "latent 2"]
        result = sample_heun(
            positive, negative, latent, model, [0.9, 0.5, 0.0], cfg=7.0
        )
        self.assertEqual(result, "latent 2")
        self.assertEqual(
            [call[0][0:2] for call in mock_predict_noise.call_args_list],
            [(latent, 0.9), ("trial 1", 0.5), ("latent 1", 0.5)],
        )
        for call in mock_predict_noise.call_args_list:
            self.assertEqual(call[0][2:], (positive, negative, model, 7.0))
        self.assertEqual(
            [call[0][0] for call in mock_remove_noise.call_args_list],
            [latent, latent, "latent 1"],
        )
        self.assertEqual(
            [call[0][2:] for call in mock_remove_noise.call_args_list],
            [(0.9, 0.5), (0.9, 0.5), (0.5, 0.0)],
        )
        self.assertEqual(
            [call[0][1].tolist() for call in mock_remove_noise.call_args_list],
            [[1.0, 2.0], [2.0, 4.0], [5.0, 5.0]],
        )


class PredictNoiseTests(TestCase):
    @patch("pydiffuse.sample.unet")
    def test_predict_noise(self, mock_unet):
        latent = Mock(torch.Tensor)
        positive = Mock(torch.Tensor)
        negative = Mock(torch.Tensor)
        model = Mock(safetensors.safe_open)
        mock_unet.side_effect = [torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5])]
        result = _predict_noise(latent, 0.5, positive, negative, model, 3.0)
        self.assertEqual(result.tolist(), [2.0, 5.0])
        self.assertEqual(
            [call[0] for call in mock_unet.call_args_list],
            [(latent, 0.5, positive, model), (latent, 0.5, negative, model)],
        )


class RemoveNoiseTests(TestCase):
    def test_remove_noise(self):
        latent = torch.tensor([1.0, 0.4])
        noise = torch.tensor([0.5, -1.0])
        result = _remove_noise(latent, noise, 0.64, 0.36)
        self.assertTrue(torch.allclose(result, torch.tensor([1.1, 1.0])))
