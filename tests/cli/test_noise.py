import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import torch


class NoiseTestCase(TestCase):
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
            [sys.executable, "-m", "pydiffuse.cli", "noise", command, *args, *params],
            capture_output=True,
            text=True,
            check=False,
        )


class ApplyTestCase(NoiseTestCase):
    def setUp(self):
        super().setUp()
        self.tensor_path = Path(__file__).parent / "data" / "unit-latent.pt"
        self.original = torch.load(self.tensor_path)

    def run_command(self, *args, **kwargs):
        return super().run_command("apply", *args, **kwargs)

    def correlation(self, noised):
        stacked = torch.stack([self.original.flatten(), noised.flatten()])
        return torch.corrcoef(stacked)[0, 1].item()

    def check_noised(self, noised, correlation, delta=0.05):
        # The noise is independent of the original, so the correlation between
        # them is how much of the original survived
        self.assertEqual(noised.shape, self.original.shape)
        self.assertEqual(noised.dtype, self.original.dtype)
        self.assertAlmostEqual(noised.var().item(), 1, delta=0.1)
        self.assertAlmostEqual(self.correlation(noised), correlation, delta=delta)

    def test_apply_noise(self):
        # Run the command with only the required arguments
        result = self.run_command(self.tensor_path, "0.5")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noised.pt").exists())

        # Noising is correct
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0.707)

    def test_apply_zero_noise(self):
        # Run the command with a noise level of zero
        result = self.run_command(self.tensor_path, "0")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Original is returned unchanged
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.assertTrue(torch.equal(noised, self.original))

    def test_apply_full_noise(self):
        # Run the command with a noise level of one
        result = self.run_command(self.tensor_path, "1")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Nothing of the original remains
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0)

    def test_low_noise_level(self):
        # Run the command with a low noise level
        result = self.run_command(self.tensor_path, "0.1")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Noising is correct
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0.949, delta=0.01)

    def test_high_noise_level(self):
        # Run the command with a high noise level
        result = self.run_command(self.tensor_path, "0.9")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Noising is correct
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0.316)

    def test_noise_differs_between_runs(self):
        # Run the command twice at the same noise level
        self.run_command(self.tensor_path, "0.5", output="first.pt")
        self.run_command(self.tensor_path, "0.5", output="second.pt")

        # Different noise is drawn each time
        with open(self.test_dir / "first.pt", "rb") as f:
            first = torch.load(f)
        with open(self.test_dir / "second.pt", "rb") as f:
            second = torch.load(f)
        self.assertFalse(torch.equal(first, second))

        # Both noisings are correct
        self.check_noised(first, 0.707)
        self.check_noised(second, 0.707)

    def test_can_set_output_path(self):
        # Run the command with a custom output path
        output_path = self.test_dir / "custom_noised.pt"
        result = self.run_command(self.tensor_path, "0.5", output=output_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(output_path.exists())
        self.assertFalse((self.test_dir / "noised.pt").exists())

        # Noising is correct
        with open(output_path, "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0.707)

    def test_tensor_is_required(self):
        # Run the command with no tensor path
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'TENSOR'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())

    def test_tensor_location_must_exist(self):
        # Run the command with an invalid tensor path
        result = self.run_command("/no/such/path/tensor.pt", "0.5")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/tensor.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())

    def test_noise_level_is_required(self):
        # Run the command with no noise level
        result = self.run_command(self.tensor_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'NOISE_LEVEL'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())

    def test_noise_level_must_be_a_number(self):
        # Run the command with a non-numeric noise level
        result = self.run_command(self.tensor_path, "abc")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'abc' is not a valid float range", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())

    def test_noise_level_must_not_exceed_one(self):
        # Run the command with an out of range noise level
        result = self.run_command(self.tensor_path, "1.5")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("1.5 is not in the range 0<=x<=1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())

    def test_output_location_must_exist(self):
        # Run the command with an invalid output path
        result = self.run_command(
            self.tensor_path, "0.5", output="/no/such/path/noised.pt"
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "noised.pt").exists())
