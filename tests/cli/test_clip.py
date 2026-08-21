import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

from pydiffuser.clip import TOKENIZER_DIR

PROMPT = """
ancient weathered stone lighthouse on a jagged basalt cliff edge, late dusk, the last embers of golden hour breaking through fractured storm clouds, volumetric god rays streaming down onto a churning slate-grey sea.

enormous waves crashing against the rocks below, white spray thrown high, wet stone glistening with reflected amber light, the lighthouse lamp glowing warm against cool blue shadows.

sea grass bending in the wind along the clifftop, distant seabirds over the water, low mist curling around the base of the cliffs, rain falling far off beneath the darkest clouds.

cinematic composition, epic landscape photography, wide angle lens, deep depth of field, moody teal and orange colour grading, ultra realistic, highly detailed, sharp focus.
""".strip()


class TokenizeTestCase(TestCase):
    def setUp(self):
        self.test_dir = Path("test_dir").resolve()
        self.test_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.test_dir.parent)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_command(self, *args, **kwargs):
        params = [f"--{key}={value}" for key, value in kwargs.items()]
        return subprocess.run(
            [sys.executable, "-m", "pydiffuser.cli", "tokenize", *args, *params],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_create_tokens(self):
        # Run the command with only the required arguments
        result = self.run_command(PROMPT)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Files are created
        self.assertTrue((self.test_dir / "tokens.json").exists())
        self.assertTrue((self.test_dir / "mappings.json").exists())

        # Tokens are correct
        with open(self.test_dir / "tokens.json") as f:
            tokens = json.load(f)
        self.assertEqual(tokens, TOKENS)

        # Mappings are correct
        with open(self.test_dir / "mappings.json") as f:
            mappings = json.load(f)
        self.assertEqual(mappings, MAPPINGS)

    def test_can_set_tokens_path(self):
        # Run the command with a custom tokens path
        tokens_path = self.test_dir / "custom_tokens.json"
        result = self.run_command(PROMPT, tokens=tokens_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Files are created (in correct places)
        self.assertTrue(tokens_path.exists())
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertTrue((self.test_dir / "mappings.json").exists())

        # Tokens are correct
        with open(tokens_path) as f:
            tokens = json.load(f)
        self.assertEqual(tokens, TOKENS)

        # Mappings are correct
        with open(self.test_dir / "mappings.json") as f:
            mappings = json.load(f)
        self.assertEqual(mappings, MAPPINGS)

    def test_can_set_mappings_path(self):
        # Run the command with a custom mappings path
        mappings_path = self.test_dir / "custom_mappings.json"
        result = self.run_command(PROMPT, mappings=mappings_path)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Files are created (in correct places)
        self.assertTrue(mappings_path.exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())
        self.assertTrue((self.test_dir / "tokens.json").exists())

        # Tokens are correct
        with open(self.test_dir / "tokens.json") as f:
            tokens = json.load(f)
        self.assertEqual(tokens, TOKENS)

        # Mappings are correct
        with open(mappings_path) as f:
            mappings = json.load(f)
        self.assertEqual(mappings, MAPPINGS)

    def test_can_set_tokenizer_path(self):
        # Copy the library tokenizer to a custom path
        custom_tokenizer = self.test_dir / "tokenizer"
        shutil.copytree(TOKENIZER_DIR, custom_tokenizer)

        # Give ancient</w> a different id (swap with parade</w>)
        with open(custom_tokenizer / "tokenizer.json") as f:
            tokenizer_data = json.load(f)
        vocab = tokenizer_data["model"]["vocab"]
        ancient_id = vocab["ancient</w>"]
        parade_id = vocab["parade</w>"]
        vocab["ancient</w>"] = parade_id
        vocab["parade</w>"] = ancient_id
        with open(custom_tokenizer / "tokenizer.json", "w") as f:
            json.dump(tokenizer_data, f)

        # Run the command with the custom tokenizer
        result = self.run_command(PROMPT, tokenizer=custom_tokenizer)

        # Process ran successfully
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertFalse(result.stderr.strip())

        # Files are created
        self.assertTrue((self.test_dir / "tokens.json").exists())
        self.assertTrue((self.test_dir / "mappings.json").exists())

        # Tokens are correct
        expected_tokens = [chunk[:] for chunk in TOKENS]
        expected_tokens[0][1] = parade_id
        with open(self.test_dir / "tokens.json") as f:
            tokens = json.load(f)
        self.assertEqual(tokens, expected_tokens)

        # Mappings are correct
        expected_mappings = [[pair[:] for pair in chunk] for chunk in MAPPINGS]
        expected_mappings[0][1] = ["ancient</w>", parade_id]
        with open(self.test_dir / "mappings.json") as f:
            mappings = json.load(f)
        self.assertEqual(mappings, expected_mappings)

    def test_prompt_is_required(self):
        # Run the command with no arguments
        result = self.run_command()

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Missing argument 'TEXT", result.stderr)

        # Files are not created
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())

    def test_tokens_location_must_exist(self):
        # Run the command with an invalid tokens path
        result = self.run_command(PROMPT, tokens="/no/such/path/tokens.json")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Invalid value for '--tokens'", result.stderr)
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # Files are not created
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())

    def test_mappings_location_must_exist(self):
        # Run the command with an invalid mappings path
        result = self.run_command(PROMPT, mappings="/no/such/path/mappings.json")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Invalid value for '--mappings'", result.stderr)
        self.assertIn("Directory '/no/such/path' does not exist", result.stderr)

        # Files are not created
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())

    def test_tokenizer_path_must_exist(self):
        # Run the command with a non-existent tokenizer path
        result = self.run_command(PROMPT, tokenizer="/no/such/path/tokenizer")

        # Process failed
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertIn("Invalid value for '--tokenizer'", result.stderr)
        self.assertIn(
            "Directory '/no/such/path/tokenizer' does not exist", result.stderr
        )

        # Files are not created
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())

    def test_tokenizer_must_be_valid(self):
        # Create a tokenizer directory with an unreadable tokenizer
        bad_tokenizer = self.test_dir / "bad_tokenizer"
        bad_tokenizer.mkdir()
        (bad_tokenizer / "tokenizer.json").write_text("{not json")

        # Run the command with the unloadable tokenizer
        result = self.run_command(PROMPT, tokenizer=bad_tokenizer)

        # Process failed
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertIn("is not a CLIP tokenizer", result.stderr)

        # Files are not created
        self.assertFalse((self.test_dir / "tokens.json").exists())
        self.assertFalse((self.test_dir / "mappings.json").exists())


TOKENS = [
    [
        49406,
        5810,
        598,
        34091,
        2441,
        13717,
        525,
        320,
        7984,
        3480,
        1244,
        10006,
        10625,
        5461,
        267,
        2325,
        19708,
        267,
        518,
        952,
        3692,
        612,
        539,
        3878,
        2232,
        2755,
        1417,
        37421,
        2819,
        6244,
        267,
        3563,
        19950,
        1731,
        11064,
        6278,
        1136,
        3141,
        320,
        2309,
        762,
        17919,
        268,
        5046,
        2102,
        269,
        20129,
        7882,
        24991,
        1601,
        518,
        6135,
        3788,
        267,
        1579,
        10449,
        15079,
        1400,
        267,
        6682,
        2441,
        70,
        3656,
        593,
        28732,
        9986,
        1395,
        267,
        518,
        13717,
        10725,
        18437,
        3616,
        1601,
        2077,
        1746,
        49407,
    ],
    [
        49406,
        12971,
        269,
        2102,
        5922,
        33897,
        530,
        518,
        3630,
        2528,
        518,
        23827,
        1253,
        267,
        18949,
        42558,
        646,
        962,
        518,
        1573,
        267,
        1042,
        11946,
        18543,
        1630,
        518,
        4579,
        539,
        518,
        20953,
        267,
        2443,
        7293,
        2384,
        1007,
        16850,
        518,
        25898,
        6244,
        269,
        25602,
        17510,
        267,
        4991,
        5727,
        2108,
        267,
        3184,
        6946,
        8666,
        267,
        3383,
        10350,
        539,
        1570,
        267,
        17170,
        22821,
        537,
        4287,
        4769,
        19016,
        267,
        8118,
        16157,
        267,
        5302,
        12609,
        267,
        8157,
        4353,
        269,
        49407,
        49407,
        49407,
        49407,
        49407,
    ],
]

MAPPINGS = [
    [
        ["<|startoftext|>", 49406],
        ["ancient</w>", 5810],
        ["we", 598],
        ["athered</w>", 34091],
        ["stone</w>", 2441],
        ["lighthouse</w>", 13717],
        ["on</w>", 525],
        ["a</w>", 320],
        ["jag", 7984],
        ["ged</w>", 3480],
        ["bas", 1244],
        ["alt</w>", 10006],
        ["cliff</w>", 10625],
        ["edge</w>", 5461],
        [",</w>", 267],
        ["late</w>", 2325],
        ["dusk</w>", 19708],
        [",</w>", 267],
        ["the</w>", 518],
        ["last</w>", 952],
        ["emb", 3692],
        ["ers</w>", 612],
        ["of</w>", 539],
        ["golden</w>", 3878],
        ["hour</w>", 2232],
        ["breaking</w>", 2755],
        ["through</w>", 1417],
        ["fractured</w>", 37421],
        ["storm</w>", 2819],
        ["clouds</w>", 6244],
        [",</w>", 267],
        ["volu", 3563],
        ["metric</w>", 19950],
        ["god</w>", 1731],
        ["rays</w>", 11064],
        ["streaming</w>", 6278],
        ["down</w>", 1136],
        ["onto</w>", 3141],
        ["a</w>", 320],
        ["chur", 2309],
        ["ning</w>", 762],
        ["slate</w>", 17919],
        ["-</w>", 268],
        ["grey</w>", 5046],
        ["sea</w>", 2102],
        [".</w>", 269],
        ["enormous</w>", 20129],
        ["waves</w>", 7882],
        ["crashing</w>", 24991],
        ["against</w>", 1601],
        ["the</w>", 518],
        ["rocks</w>", 6135],
        ["below</w>", 3788],
        [",</w>", 267],
        ["white</w>", 1579],
        ["spray</w>", 10449],
        ["thrown</w>", 15079],
        ["high</w>", 1400],
        [",</w>", 267],
        ["wet</w>", 6682],
        ["stone</w>", 2441],
        ["g", 70],
        ["listening</w>", 3656],
        ["with</w>", 593],
        ["reflected</w>", 28732],
        ["amber</w>", 9986],
        ["light</w>", 1395],
        [",</w>", 267],
        ["the</w>", 518],
        ["lighthouse</w>", 13717],
        ["lamp</w>", 10725],
        ["glowing</w>", 18437],
        ["warm</w>", 3616],
        ["against</w>", 1601],
        ["cool</w>", 2077],
        ["blue</w>", 1746],
        ["<|endoftext|>", 49407],
    ],
    [
        ["<|startoftext|>", 49406],
        ["shadows</w>", 12971],
        [".</w>", 269],
        ["sea</w>", 2102],
        ["grass</w>", 5922],
        ["bending</w>", 33897],
        ["in</w>", 530],
        ["the</w>", 518],
        ["wind</w>", 3630],
        ["along</w>", 2528],
        ["the</w>", 518],
        ["cliff", 23827],
        ["top</w>", 1253],
        [",</w>", 267],
        ["distant</w>", 18949],
        ["seabir", 42558],
        ["ds</w>", 646],
        ["over</w>", 962],
        ["the</w>", 518],
        ["water</w>", 1573],
        [",</w>", 267],
        ["low</w>", 1042],
        ["mist</w>", 11946],
        ["curling</w>", 18543],
        ["around</w>", 1630],
        ["the</w>", 518],
        ["base</w>", 4579],
        ["of</w>", 539],
        ["the</w>", 518],
        ["cliffs</w>", 20953],
        [",</w>", 267],
        ["rain</w>", 2443],
        ["falling</w>", 7293],
        ["far</w>", 2384],
        ["off</w>", 1007],
        ["beneath</w>", 16850],
        ["the</w>", 518],
        ["darkest</w>", 25898],
        ["clouds</w>", 6244],
        [".</w>", 269],
        ["cinematic</w>", 25602],
        ["composition</w>", 17510],
        [",</w>", 267],
        ["epic</w>", 4991],
        ["landscape</w>", 5727],
        ["photography</w>", 2108],
        [",</w>", 267],
        ["wide</w>", 3184],
        ["angle</w>", 6946],
        ["lens</w>", 8666],
        [",</w>", 267],
        ["deep</w>", 3383],
        ["depth</w>", 10350],
        ["of</w>", 539],
        ["field</w>", 1570],
        [",</w>", 267],
        ["moody</w>", 17170],
        ["teal</w>", 22821],
        ["and</w>", 537],
        ["orange</w>", 4287],
        ["colour</w>", 4769],
        ["grading</w>", 19016],
        [",</w>", 267],
        ["ultra</w>", 8118],
        ["realistic</w>", 16157],
        [",</w>", 267],
        ["highly</w>", 5302],
        ["detailed</w>", 12609],
        [",</w>", 267],
        ["sharp</w>", 8157],
        ["focus</w>", 4353],
        [".</w>", 269],
        ["<|endoftext|>", 49407],
        ["<|endoftext|>", 49407],
        ["<|endoftext|>", 49407],
        ["<|endoftext|>", 49407],
        ["<|endoftext|>", 49407],
    ],
]
