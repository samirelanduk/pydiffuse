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
        self.conditioning_path = (
            Path(__file__).parent / "data" / "positive-conditioning.pt"
        )
        self.model_path = Path(__file__).parent / "models" / "unet_model.safetensors"

    def create_latent(self):
        latent_path = self.test_dir / "latent.pt"
        torch.manual_seed(42)
        torch.save(torch.randn(1, 4, 8, 8), latent_path)
        return latent_path

    def run_command(self, *args, **kwargs):
        return super().run_command("predict", *args, **kwargs)

    def test(self):
        # Run the command with only the required arguments
        result = self.run_command(
            self.latent_path, "0.5", self.conditioning_path, self.model_path
        )

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "noise.pt").exists())
