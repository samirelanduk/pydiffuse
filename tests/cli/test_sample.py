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

    def create_latent(self):
        latent_path = self.test_dir / "latent.pt"
        torch.manual_seed(42)
        torch.save(torch.randn(4, 7, 9), latent_path)
        return latent_path

    def create_schedule(self):
        schedule_path = self.test_dir / "schedule.txt"
        with open(schedule_path, "w") as f:
            f.write("0.9953399\n0.61880525\n0.00085\n0.0\n")
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
        self.check_denoised(
            denoised, [29.3, -12.4, -13.1, -22.9, -12.6, 6.5, -25.6, 2.5]
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
