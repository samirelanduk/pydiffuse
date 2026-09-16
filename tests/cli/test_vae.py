import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import torch
from PIL import Image


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


class EncodeTestCase(VaeTestCase):
    def setUp(self):
        super().setUp()
        self.image_path = Path(__file__).parent / "data" / "small-flower.jpg"
        self.model_path = Path(__file__).parent / "models" / "vae_model.safetensors"

    def run_command(self, *args, **kwargs):
        return super().run_command("encode", *args, **kwargs)

    def check_latent(self, latent):
        self.assertEqual(latent.shape, (4, 37, 50))
        self.assertEqual(round(latent[0, 0, 0].item(), 3), -0.006)
        self.assertEqual(round(latent[0, 0, 49].item(), 3), 0.011)
        self.assertEqual(round(latent[0, 36, 0].item(), 3), 0.042)
        self.assertEqual(round(latent[0, 36, 49].item(), 3), 0.018)
        self.assertEqual(round(latent[3, 0, 0].item(), 3), 0.086)
        self.assertEqual(round(latent[3, 0, 49].item(), 3), -0.068)
        self.assertEqual(round(latent[3, 36, 0].item(), 3), -0.011)
        self.assertEqual(round(latent[3, 36, 49].item(), 3), -0.033)

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

    def test_can_set_latent_path(self):
        # Run the command with a custom latent path
        latent_path = self.test_dir / "custom_latent.pt"
        result = self.run_command(self.image_path, self.model_path, latent=latent_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(latent_path.exists())
        self.assertFalse((self.test_dir / "latent.pt").exists())

        # Encoding is correct
        with open(latent_path, "rb") as f:
            latent = torch.load(f)
        self.check_latent(latent)

    def test_image_is_required(self):
        # Run the command with no image path
        result = self.run_command(self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_image_location_must_exist(self):
        # Run the command with an invalid image path
        result = self.run_command("/no/such/path/image.jpg", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/image.jpg' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_model_is_required(self):
        # Run the command with no model path
        result = self.run_command(self.image_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_model_location_must_exist(self):
        # Run the command with an invalid model path
        result = self.run_command(self.image_path, "/no/such/path/model.safetensors")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/model.safetensors' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())

    def test_latent_location_must_exist(self):
        # Run the command with an invalid latent path
        result = self.run_command(
            self.image_path, self.model_path, latent="/no/such/path/latent.pt"
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "latent.pt").exists())


class DecodeTestCase(VaeTestCase):
    def setUp(self):
        super().setUp()
        self.latent_path = Path(__file__).parent / "data" / "small-flower.pt"
        self.model_path = Path(__file__).parent / "models" / "vae_model.safetensors"

    def run_command(self, *args, **kwargs):
        return super().run_command("decode", *args, **kwargs)

    def check_image(self, image):
        self.assertEqual(image.size, (100, 74))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (136, 107, 161))
        self.assertEqual(image.getpixel((99, 73)), (143, 112, 128))

    def test_decode_latent(self):
        # Run the command with only the required arguments
        result = self.run_command(self.latent_path, self.model_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created
        self.assertTrue((self.test_dir / "image.jpg").exists())

        # Image is correct
        with Image.open(self.test_dir / "image.jpg") as image:
            self.check_image(image)

    def test_can_set_image_path(self):
        # Run the command with a custom image path
        image_path = self.test_dir / "custom_image.jpg"
        result = self.run_command(self.latent_path, self.model_path, image=image_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # File is created (in correct place)
        self.assertTrue(image_path.exists())
        self.assertFalse((self.test_dir / "image.jpg").exists())

        # Image is correct
        with Image.open(image_path) as image:
            self.check_image(image)

    def test_latent_is_required(self):
        # Run the command with no latent path
        result = self.run_command(self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "image.jpg").exists())

    def test_latent_location_must_exist(self):
        # Run the command with an invalid latent path
        result = self.run_command("/no/such/path/latent.pt", self.model_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("File '/no/such/path/latent.pt' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "image.jpg").exists())

    def test_model_is_required(self):
        # Run the command with no model path
        result = self.run_command(self.latent_path)

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'MODEL", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "image.jpg").exists())

    def test_model_location_must_exist(self):
        # Run the command with an invalid model path
        result = self.run_command(self.latent_path, "/no/such/path/model.safetensors")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn(
            "File '/no/such/path/model.safetensors' does not exist", result.stderr
        )

        # File is not created
        self.assertFalse((self.test_dir / "image.jpg").exists())

    def test_image_location_must_exist(self):
        # Run the command with an invalid image path
        result = self.run_command(
            self.latent_path, self.model_path, image="/no/such/path/image.jpg"
        )

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # File is not created
        self.assertFalse((self.test_dir / "image.jpg").exists())
