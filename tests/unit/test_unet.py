from unittest import TestCase

from pydiffuse.unet import (
    _noise_to_t,
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
