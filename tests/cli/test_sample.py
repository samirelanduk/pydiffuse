import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import torch


class SampleTestCase(TestCase):
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
            [sys.executable, "-m", "pydiffuse.cli", "sample", command, *args, *params],
            capture_output=True,
            text=True,
            check=False,
        )


class DenoiseTestCase(SampleTestCase):
    def setUp(self):
        super().setUp()
        self.latent_path = self.create_latent()
        self.schedule_path = self.create_schedule()
        data_path = Path(__file__).parent / "data"
        self.positive_path = data_path / "positive-conditioning.pt"
        self.negative_path = data_path / "negative-conditioning.pt"
        self.model_path = Path(__file__).parent / "models" / "unet_model.safetensors"
        self.euler_denoised = [29.3, -12.4, -13.1, -22.9, -12.6, 6.5, -25.6, 2.5]

    def create_latent(self):
        latent_path = self.test_dir / "latent.pt"
        torch.manual_seed(42)
        torch.save(torch.randn(4, 7, 9), latent_path)
        return latent_path

    def create_schedule(self, levels="0.9953399\n0.61880525\n0.00085\n0.0\n"):
        schedule_path = self.test_dir / "schedule.txt"
        with open(schedule_path, "w") as f:
            f.write(levels)
        return schedule_path

    def run_command(self, *args, **kwargs):
        return super().run_command("denoise", *args, **kwargs)

    def check_denoised(self, denoised, values):
        self.assertEqual(denoised.shape, (4, 7, 9))
        positions = [(c, h, w) for c in (0, 3) for h in (0, 6) for w in (0, 8)]
        self.assertEqual(
            [round(denoised[position].item(), 1) for position in positions], values
        )

    def test_denoise(self):
        # Run the command with only the required arguments
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "denoised.pt").exists())

        # Denoising is correct
        with open(self.test_dir / "denoised.pt", "rb") as f:
            denoised = torch.load(f)
        self.check_denoised(denoised, self.euler_denoised)

    def test_can_use_euler_algorithm(self):
        # Run the command with the euler algorithm named explicitly
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            algorithm="euler",
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "denoised.pt").exists())

        # Denoising is correct
        with open(self.test_dir / "denoised.pt", "rb") as f:
            denoised = torch.load(f)
        self.check_denoised(denoised, self.euler_denoised)

    def test_can_use_heun_algorithm(self):
        # Run the command with the heun algorithm
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            algorithm="heun",
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "denoised.pt").exists())

        # Denoising is correct for the heun algorithm
        with open(self.test_dir / "denoised.pt", "rb") as f:
            denoised = torch.load(f)
        self.check_denoised(
            denoised, [26.0, -10.0, -16.4, -23.7, -11.5, 5.8, -24.5, 1.9]
        )

    def test_can_set_cfg(self):
        # Run the command with a custom CFG
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            cfg=5,
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "denoised.pt").exists())

        # Denoising is correct
        with open(self.test_dir / "denoised.pt", "rb") as f:
            denoised = torch.load(f)
        self.check_denoised(
            denoised, [29.5, -12.6, -13.6, -23.7, -12.8, 6.5, -25.7, 2.7]
        )

    def test_can_set_output_path(self):
        # Run the command with a custom output path
        output_path = self.test_dir / "custom_denoised.pt"
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            output=output_path,
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(output_path.exists())
        self.assertFalse((self.test_dir / "denoised.pt").exists())

        # Denoising is correct
        with open(output_path, "rb") as f:
            denoised = torch.load(f)
        self.check_denoised(denoised, self.euler_denoised)

    def test_latent_is_required(self):
        # Run the command with no latent path
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'LATENT'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_latent_location_must_exist(self):
        # Run the command with an invalid latent path
        result = self.run_command(
            "/no/such/path/latent.pt",
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/latent.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_positive_is_required(self):
        # Run the command with no positive path
        result = self.run_command(self.latent_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'POSITIVE'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_positive_location_must_exist(self):
        # Run the command with an invalid positive path
        result = self.run_command(
            self.latent_path,
            "/no/such/path/positive.pt",
            self.negative_path,
            self.schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/positive.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_negative_is_required(self):
        # Run the command with no negative path
        result = self.run_command(self.latent_path, self.positive_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'NEGATIVE'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_negative_location_must_exist(self):
        # Run the command with an invalid negative path
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            "/no/such/path/negative.pt",
            self.schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/negative.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_is_required(self):
        # Run the command with no schedule path
        result = self.run_command(
            self.latent_path, self.positive_path, self.negative_path
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'SCHEDULE'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_location_must_exist(self):
        # Run the command with an invalid schedule path
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            "/no/such/path/schedule.txt",
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/schedule.txt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_levels_must_be_numbers(self):
        # Run the command with a schedule containing a non-numeric level
        schedule_path = self.create_schedule("0.9953399\nabc\n0.0\n")
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'abc' is not a valid noise level", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_levels_must_be_below_one(self):
        # Run the command with a schedule containing a level of one
        schedule_path = self.create_schedule("1.0\n0.5\n0.0\n")
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("1.0 is not in the range 0<=x<1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_levels_must_not_be_negative(self):
        # Run the command with a schedule containing a negative level
        schedule_path = self.create_schedule("0.9953399\n0.5\n-0.1\n")
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("-0.1 is not in the range 0<=x<1", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_schedule_must_have_two_levels(self):
        # Run the command with a schedule of a single level
        schedule_path = self.create_schedule("0.9953399\n")
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            schedule_path,
            self.model_path,
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("A schedule must have at least two noise levels", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_model_is_required(self):
        # Run the command with no model path
        result = self.run_command(
            self.latent_path, self.positive_path, self.negative_path, self.schedule_path
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_model_location_must_exist(self):
        # Run the command with an invalid model path
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            "/no/such/path/model.safetensors",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/model.safetensors' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_cfg_must_be_a_number(self):
        # Run the command with a non-numeric CFG
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            cfg="abc",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'abc' is not a valid float", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_algorithm_must_be_known(self):
        # Run the command with an unknown algorithm
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            algorithm="linear",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("'linear' is not one of 'euler', 'heun'", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())

    def test_output_location_must_exist(self):
        # Run the command with an invalid output path
        result = self.run_command(
            self.latent_path,
            self.positive_path,
            self.negative_path,
            self.schedule_path,
            self.model_path,
            output="/no/such/path/denoised.pt",
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "denoised.pt").exists())
