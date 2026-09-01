import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import torch


class VaeTestCase(TestCase):
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
            [sys.executable, "-m", "pydiffuse.cli", "vae", command, *args, *params],
            capture_output=True,
            text=True,
            check=False,
        )

    def check_latent(self, latent):
        self.assertEqual(latent.shape, (1, 4, 36, 48))
        self.assertEqual(round(latent[0, 0, 0, 0].item(), 3), 0.226)
        self.assertEqual(round(latent[0, 0, 0, 47].item(), 3), 0.022)
        self.assertEqual(round(latent[0, 0, 35, 0].item(), 3), 0.252)
        self.assertEqual(round(latent[0, 0, 35, 47].item(), 3), 0.069)
        self.assertEqual(round(latent[0, 3, 0, 0].item(), 3), 0.275)
        self.assertEqual(round(latent[0, 3, 0, 47].item(), 3), -0.238)
        self.assertEqual(round(latent[0, 3, 35, 0].item(), 3), -0.13)
        self.assertEqual(round(latent[0, 3, 35, 47].item(), 3), -0.115)


class EncodeTestCase(VaeTestCase):
    def setUp(self):
        super().setUp()
        self.image_path = Path(__file__).parent / "data" / "small-flower.jpg"
        self.model_path = Path(__file__).parent / "models" / "vae_model.safetensors"

    def run_command(self, *args, **kwargs):
        return super().run_command("encode", *args, **kwargs)

    def test_encode_image(self):
        # Run the command with only the required arguments
        result = self.run_command(self.image_path, self.model_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "latent.pt").exists())

        # Encoding is correct
        with open(self.test_dir / "latent.pt", "rb") as f:
            latent = torch.load(f)
        self.check_latent(latent)
