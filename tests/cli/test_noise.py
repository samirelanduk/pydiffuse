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


class CreateTestCase(NoiseTestCase):
    def setUp(self):
        super().setUp()
        self.model_path = Path(__file__).parent / "models" / "sd15.safetensors"

    def run_command(self, *args, **kwargs):
        return super().run_command("create", *args, **kwargs)

    def check_latent(self, latent, shape):
        # The noise is unit random noise of the latent's shape
        self.assertEqual(latent.shape, shape)
        self.assertEqual(latent.dtype, torch.float32)
        self.assertAlmostEqual(latent.mean().item(), 0, delta=0.1)
        self.assertAlmostEqual(latent.var().item(), 1, delta=0.1)

    def test_create_latent(self):
        # Run the command with only the required arguments
        result = self.run_command("400", "296", self.model_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "latent.pt").exists())

        # Latent is correct
        with open(self.test_dir / "latent.pt", "rb") as f:
            latent = torch.load(f)
        self.check_latent(latent, (4, 37, 50))

    def test_create_latent_with_uneven_size(self):
        # Run the command with a size that isn't a multiple of the downscale ratio
        result = self.run_command("401", "303", self.model_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "latent.pt").exists())

        # Latent is rounded down to the same size as an even one
        with open(self.test_dir / "latent.pt", "rb") as f:
            latent = torch.load(f)
        self.check_latent(latent, (4, 37, 50))

    def test_can_set_output_path(self):
        # Run the command with a custom output path
        output_path = self.test_dir / "custom_latent.pt"
        result = self.run_command("400", "296", self.model_path, output=output_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(output_path.exists())
        self.assertFalse((self.test_dir / "latent.pt").exists())

        # Latent is correct
        with open(output_path, "rb") as f:
            latent = torch.load(f)
        self.check_latent(latent, (4, 37, 50))

    def test_width_is_required(self):
        # Run the command with no width
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'WIDTH'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_width_must_be_an_integer(self):
        # Run the command with a non-integer width
        result = self.run_command("100.5", "74", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'100.5' is not a valid integer", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_width_must_be_positive(self):
        # Run the command with a width of zero
        result = self.run_command("0", "74", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("0 is not in the range x>=1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_height_is_required(self):
        # Run the command with no height
        result = self.run_command("100")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'HEIGHT'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_height_must_be_an_integer(self):
        # Run the command with a non-integer height
        result = self.run_command("100", "74.5", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'74.5' is not a valid integer", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_height_must_be_positive(self):
        # Run the command with a height of zero
        result = self.run_command("100", "0", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("0 is not in the range x>=1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_model_is_required(self):
        # Run the command with no model path
        result = self.run_command("100", "74")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_model_location_must_exist(self):
        # Run the command with an invalid model path
        result = self.run_command("100", "74", "/no/such/path/model.safetensors")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/model.safetensors' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_output_location_must_exist(self):
        # Run the command with an invalid output path
        result = self.run_command(
            "100", "74", self.model_path, output="/no/such/path/latent.pt"
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())


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

        # File is created
        self.assertTrue((self.test_dir / "noised.pt").exists())

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

        # File is created
        self.assertTrue((self.test_dir / "noised.pt").exists())

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

        # File is created
        self.assertTrue((self.test_dir / "noised.pt").exists())

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

        # File is created
        self.assertTrue((self.test_dir / "noised.pt").exists())

        # Noising is correct
        with open(self.test_dir / "noised.pt", "rb") as f:
            noised = torch.load(f)
        self.check_noised(noised, 0.316)

    def test_noise_differs_between_runs(self):
        # Run the command twice at the same noise level
        self.run_command(self.tensor_path, "0.5", output="first.pt")
        self.run_command(self.tensor_path, "0.5", output="second.pt")

        # Files are created
        self.assertTrue((self.test_dir / "first.pt").exists())
        self.assertTrue((self.test_dir / "second.pt").exists())

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


class ScheduleTestCase(NoiseTestCase):
    def setUp(self):
        super().setUp()
        self.karras_5 = [0.9953399, 0.95834503, 0.61880525, 0.05791837, 0.00085, 0.0]

    def run_command(self, *args, **kwargs):
        return super().run_command("schedule", *args, **kwargs)

    def read_schedule(self, path):
        with open(path) as f:
            return [float(line) for line in f.read().splitlines()]

    def test_schedule(self):
        # Run the command with only the required arguments
        result = self.run_command("5")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "schedule.txt").exists())

        # Schedule is correct for the karras algorithm
        levels = self.read_schedule(self.test_dir / "schedule.txt")
        self.assertEqual(levels, self.karras_5)

    def test_single_step_schedule(self):
        # Run the command with a single step
        result = self.run_command("1")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "schedule.txt").exists())

        # Schedule is correct
        levels = self.read_schedule(self.test_dir / "schedule.txt")
        self.assertEqual(levels, [0.9953399, 0.0])

    def test_can_use_exponential_algorithm(self):
        # Run the command with the exponential algorithm
        result = self.run_command("5", algorithm="exponential")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "schedule.txt").exists())

        # Schedule is correct for the exponential algorithm
        levels = self.read_schedule(self.test_dir / "schedule.txt")
        self.assertEqual(
            levels, [0.9953399, 0.90513932, 0.29886924, 0.01868713, 0.00085, 0.0]
        )

    def test_can_use_karras_algorithm(self):
        # Run the command with the karras algorithm named explicitly
        result = self.run_command("5", algorithm="karras")

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "schedule.txt").exists())

        # Schedule is correct
        levels = self.read_schedule(self.test_dir / "schedule.txt")
        self.assertEqual(levels, self.karras_5)

    def test_can_set_output_path(self):
        # Run the command with a custom output path
        output_path = self.test_dir / "custom_schedule.txt"
        result = self.run_command("5", output=output_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(output_path.exists())
        self.assertFalse((self.test_dir / "schedule.txt").exists())

        # Schedule is correct
        levels = self.read_schedule(output_path)
        self.assertEqual(
            levels, [0.9953399, 0.95834503, 0.61880525, 0.05791837, 0.00085, 0.0]
        )

    def test_steps_is_required(self):
        # Run the command with no steps
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'STEPS'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "schedule.txt").exists())

    def test_steps_must_be_an_integer(self):
        # Run the command with a non-integer number of steps
        result = self.run_command("2.5")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'2.5' is not a valid integer", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "schedule.txt").exists())

    def test_steps_must_be_positive(self):
        # Run the command with zero steps
        result = self.run_command("0")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("0 is not in the range x>=1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "schedule.txt").exists())

    def test_algorithm_must_be_known(self):
        # Run the command with an unknown algorithm
        result = self.run_command("5", algorithm="linear")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'linear' is not one of 'karras', 'exponential'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "schedule.txt").exists())

    def test_output_location_must_exist(self):
        # Run the command with an output path in a non-existent directory
        result = self.run_command("5", output="/no/such/path/schedule.txt")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "schedule.txt").exists())
