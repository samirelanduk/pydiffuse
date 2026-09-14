from unittest import TestCase

import torch

from pydiffuse.unet import (
    _noise_to_t,
    _timestep_sinusoids,
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
