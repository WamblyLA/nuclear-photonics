import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from scripts.export_solution import export_solution, file_hash, plain_json, portable_state_dict
from src.models.networks import build_model
from src.protocol import PEAK_PROTOCOL


class PortableExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_checkpoint(self, directory):
        torch.manual_seed(7)
        model = build_model("cnn", width=4, phase_head="circular").double().eval()
        checkpoint = directory / "research.pt"
        saved = {
            "model": model.state_dict(),
            "optimizer": {"private_training_state": torch.ones(4)},
            "step": 23,
            "config": {
                "model": "cnn", "width": 4, "phase_head": "circular",
                "protocol": PEAK_PROTOCOL, "amplitude_scales": [0.4, 0.5, 0.6, 0.7],
                "seed": 7, "torch": TorchVersion("2.14.0"),
                "resume": Path("private/research.pt"), "private_note": "do not export",
            },
        }
        torch.save(saved, checkpoint)
        return checkpoint, model, saved

    def test_export_loads_without_pickle_allowlisting_and_preserves_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint, original, _ = self.make_checkpoint(directory)
            original_sha = file_hash(checkpoint)
            metadata = export_solution(checkpoint, directory / "release")
            weights_path = directory / "release" / "solution.pt"
            restored = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.assertEqual(set(restored), {"format_version", "model", "config", "step", "provenance"})
            self.assertEqual(restored["step"], 23)
            self.assertEqual(restored["config"]["protocol"], PEAK_PROTOCOL)
            self.assertEqual(set(restored["config"]), {"model", "width", "phase_head", "position", "protocol", "amplitude_scales"})
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in restored["model"].values()))
            self.assertTrue(all(tensor.dtype == torch.float32 for tensor in restored["model"].values()))
            self.assertEqual(metadata["source_checkpoint_sha256"], original_sha)
            self.assertEqual(file_hash(checkpoint), original_sha)
            self.assertEqual(metadata["checkpoint_sha256"], file_hash(weights_path))
            self.assertEqual(metadata["selected_step"], 23)
            self.assertEqual(metadata["training_hyperparameters"], {"seed": 7})
            self.assertFalse(metadata["optimizer_included"])
            self.assertEqual(json.loads((directory / "release" / "metadata.json").read_text(encoding="utf-8")), metadata)
            self.assertNotIn(str(directory), json.dumps(metadata))
            loaded = build_model("cnn", width=4, phase_head="circular").eval()
            loaded.load_state_dict(restored["model"], strict=True)
            inputs = torch.rand(1, 4, 128, 128)
            with torch.inference_mode():
                torch.testing.assert_close(original.float()(inputs), loaded(inputs), rtol=0, atol=0)

    def test_complex_buffers_keep_imaginary_components(self):
        state = portable_state_dict({
            "complex": torch.tensor([1 + 2j], dtype=torch.complex128),
            "integer": torch.tensor([3], dtype=torch.int64),
            "boolean": torch.tensor([True]),
        })
        self.assertEqual(state["complex"].dtype, torch.complex64)
        self.assertEqual(state["complex"].item(), 1 + 2j)
        self.assertEqual(state["integer"].dtype, torch.int64)
        self.assertEqual(state["boolean"].dtype, torch.bool)

    def test_existing_export_and_source_are_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint, _, _ = self.make_checkpoint(directory)
            export_solution(checkpoint, directory / "release")
            before = file_hash(directory / "release" / "solution.pt")
            with self.assertRaises(FileExistsError):
                export_solution(checkpoint, directory / "release")
            with self.assertRaises(ValueError):
                export_solution(directory / "release" / "solution.pt", directory / "release", overwrite=True)
            self.assertEqual(file_hash(directory / "release" / "solution.pt"), before)

    def test_invalid_checkpoint_does_not_create_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint, _, saved = self.make_checkpoint(directory)
            saved["config"]["protocol"] = "unknown"
            torch.save(saved, checkpoint)
            with self.assertRaises(ValueError):
                export_solution(checkpoint, directory / "release")
            self.assertFalse((directory / "release").exists())
            saved["config"]["protocol"] = PEAK_PROTOCOL
            next(iter(saved["model"].values())).flatten()[0] = float("nan")
            torch.save(saved, checkpoint)
            with self.assertRaises(ValueError):
                export_solution(checkpoint, directory / "release")
            self.assertFalse((directory / "release").exists())

    def test_plain_metadata_has_no_custom_objects_or_nonfinite_numbers(self):
        self.assertIs(type(plain_json(TorchVersion("2.6.0"))), str)
        for value in (Path("private"), float("nan"), float("inf"), {1: "bad key"}):
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    plain_json(value)

    def test_memorization_checkpoint_cannot_be_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint, _, saved = self.make_checkpoint(directory)
            saved["config"]["fixed_train_quartets"] = 8
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "memorization"):
                export_solution(checkpoint, directory / "release")
            self.assertFalse((directory / "release").exists())
            saved["config"].update({"fixed_train_quartets": 0, "diagnostic_only": True})
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "Diagnostic"):
                export_solution(checkpoint, directory / "release")
            saved["config"].pop("diagnostic_only")
            saved["config"].update({"fixed_train_quartets": 0, "lr_schedule": "cosine", "quantize_train": True})
            torch.save(saved, checkpoint)
            metadata = export_solution(checkpoint, directory / "release")
            self.assertEqual(metadata["training_hyperparameters"]["fixed_train_quartets"], 0)
            self.assertEqual(metadata["training_hyperparameters"]["lr_schedule"], "cosine")
            self.assertIs(metadata["training_hyperparameters"]["quantize_train"], True)


if __name__ == "__main__":
    unittest.main()
