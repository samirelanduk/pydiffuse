import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import torch


class UnetTestCase(TestCase):
    def setUp(self):
        self.test_dir = Path("test_dir").resolve()
        self.test_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.test_dir.parent)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_command(self, command, *args, **kwargs):
        params = [f"--{key}={value}" for key, value in kwargs.items()]
        return subprocess.run(
            [sys.executable, "-m", "pydiffuse.cli", "unet", command, *args, *params],
            capture_output=True,
            text=True,
            check=False,
        )


class UnetPredictTestCase(UnetTestCase):
    def setUp(self):
        super().setUp()
        self.latent_path = self.create_latent()
        data_path = Path(__file__).parent / "data"
        self.multi_chunk_conditioning_path = data_path / "positive-conditioning.pt"
        self.single_chunk_conditioning_path = data_path / "negative-conditioning.pt"
        self.model_path = Path(__file__).parent / "models" / "unet_model.safetensors"

    def create_latent(self):
        latent_path = self.test_dir / "latent.pt"
        torch.manual_seed(42)
        torch.save(torch.randn(4, 7, 9), latent_path)
        return latent_path

    def run_command(self, *args, **kwargs):
        return super().run_command("predict", *args, **kwargs)

    def check_noise(self, noise, values):
        self.assertEqual(noise.shape, (4, 7, 9))
        positions = [(c, h, w) for c in (0, 3) for h in (0, 6) for w in (0, 8)]
        self.assertEqual(
            [round(noise[position].item(), 3) for position in positions], values
        )

    def test_zero_noise(self):
        # Run the command with a noise level of zero
        result = self.run_command(
            self.latent_path, "0", self.multi_chunk_conditioning_path, self.model_path
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noise.pt").exists())

        # Prediction is correct
        with open(self.test_dir / "noise.pt", "rb") as f:
            noise = torch.load(f)
        self.check_noise(
            noise, [-0.237, 0.003, 0.053, 0.205, 0.254, -0.069, 0.077, -0.273]
        )

    def test_half_noise(self):
        # Run the command with only the required arguments
        result = self.run_command(
            self.latent_path, "0.5", self.multi_chunk_conditioning_path, self.model_path
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noise.pt").exists())

        # Prediction is correct
        with open(self.test_dir / "noise.pt", "rb") as f:
            noise = torch.load(f)
        self.check_noise(
            noise, [-0.237, 0.009, 0.059, 0.222, 0.25, -0.092, 0.074, -0.265]
        )

    def test_full_noise(self):
        # Run the command with a noise level of one
        result = self.run_command(
            self.latent_path, "1", self.multi_chunk_conditioning_path, self.model_path
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noise.pt").exists())

        # Prediction is correct
        with open(self.test_dir / "noise.pt", "rb") as f:
            noise = torch.load(f)
        self.check_noise(
            noise, [-0.238, 0.013, 0.063, 0.22, 0.249, -0.077, 0.069, -0.268]
        )

    def test_single_chunk_conditioning(self):
        # Run the command with a conditioning of a single chunk
        result = self.run_command(
            self.latent_path,
            "0.5",
            self.single_chunk_conditioning_path,
            self.model_path,
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noise.pt").exists())

        # Prediction is correct
        with open(self.test_dir / "noise.pt", "rb") as f:
            noise = torch.load(f)
        self.check_noise(
            noise, [-0.236, 0.018, 0.064, 0.206, 0.246, -0.091, 0.065, -0.266]
        )

    def test_can_set_noise_path(self):
        # Run the command with a custom noise path
        noise_path = self.test_dir / "custom_noise.pt"
        result = self.run_command(
            self.latent_path,
            "0.5",
            self.multi_chunk_conditioning_path,
            self.model_path,
            noise=noise_path,
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(noise_path.exists())
        self.assertFalse((self.test_dir / "noise.pt").exists())

        # Prediction is correct
        with open(noise_path, "rb") as f:
            noise = torch.load(f)
        self.check_noise(
            noise, [-0.237, 0.009, 0.059, 0.222, 0.25, -0.092, 0.074, -0.265]
        )

    def test_latent_is_required(self):
        # Run the command with no latent path
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'LATENT'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_latent_location_must_exist(self):
        # Run the command with an invalid latent path
        result = self.run_command(
            "/no/such/path/latent.pt",
            "0.5",
            self.multi_chunk_conditioning_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/latent.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_noise_level_is_required(self):
        # Run the command with no noise level
        result = self.run_command(self.latent_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'NOISE_LEVEL'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_noise_level_must_be_a_number(self):
        # Run the command with a non-numeric noise level
        result = self.run_command(
            self.latent_path,
            "abc",
            self.multi_chunk_conditioning_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'abc' is not a valid float range", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_noise_level_must_not_exceed_one(self):
        # Run the command with an out of range noise level
        result = self.run_command(
            self.latent_path,
            "1.5",
            self.multi_chunk_conditioning_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("1.5 is not in the range 0<=x<=1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_conditioning_is_required(self):
        # Run the command with no conditioning path
        result = self.run_command(self.latent_path, "0.5")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'CONDITIONING'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_conditioning_location_must_exist(self):
        # Run the command with an invalid conditioning path
        result = self.run_command(
            self.latent_path,
            "0.5",
            "/no/such/path/conditioning.pt",
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/conditioning.pt' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_model_is_required(self):
        # Run the command with no model path
        result = self.run_command(
            self.latent_path, "0.5", self.multi_chunk_conditioning_path
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_model_location_must_exist(self):
        # Run the command with an invalid model path
        result = self.run_command(
            self.latent_path,
            "0.5",
            self.multi_chunk_conditioning_path,
            "/no/such/path/model.safetensors",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/model.safetensors' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())

    def test_noise_location_must_exist(self):
        # Run the command with an invalid noise path
        result = self.run_command(
            self.latent_path,
            "0.5",
            self.multi_chunk_conditioning_path,
            self.model_path,
            noise="/no/such/path/noise.pt",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noise.pt").exists())
