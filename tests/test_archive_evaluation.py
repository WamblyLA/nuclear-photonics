import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.archive_evaluation import archive_evaluation, reject_links, sha256
from src.compare_candidates import load_evaluation, validate_alignment


class EvaluationArchiveTests(unittest.TestCase):
    def make_evaluation(self, root, test=False):
        folder = root / "outputs" / "evaluation"
        folder.mkdir(parents=True)
        manifest_name = "test_tuples.json" if test else "validation_tuples.json"
        manifest = {"mixed": [[[0, 10], [1, 11], [2, 12], [0, 13]]]}
        metrics = {"protocol": "fresnel_png_v2_peak", "test_used": test, "tuples": 1, "aggregate": {"ssim": 0.25, "pearson": 0.5}, "groups": {"mixed": {"ssim": 0.25, "pearson": 0.5}}}
        (folder / "metrics.json").write_bytes((json.dumps(metrics, indent=3) + "\r\n").encode())
        (folder / manifest_name).write_bytes((json.dumps(manifest, indent=1) + "\r\n").encode())
        np.savez_compressed(folder / "per_example_metrics.npz", mixed__ssim=np.full((1, 4), 0.25), mixed__pearson=np.full((1, 4), 0.5), mixed__pearson_valid=np.ones((1, 4), dtype=bool))
        hashes = {}
        for name in ("data.py", "metrics.py", "physics/fresnel.py", "train.py"):
            path = folder / "source/src" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = "# fixture source\r\n"
            if name == "train.py":
                text += "\r\n".join(f"def {function}():\r\n    pass\r\n" for function in ("evaluate", "png_roundtrip", "image_metrics", "summarize"))
            path.write_bytes(text.encode())
            hashes[("src/" + name).replace("/", "\\")] = sha256(path.read_bytes())
        (folder / "source_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
        config = {"training_config": {"protocol": "fresnel_png_v2_peak", "data_seed": 42, "resume": str(root / "outputs/prior/best.pt")}, "source_hashes": hashes, "external": "Z:/private/checkpoint.pt"}
        (folder / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
        if test:
            provenance = {"namespace": "synthetic_official_test", "sources": {"fixture": "0" * 64}, "preprocessing": "fixture", "test_tuples_sha256": sha256((folder / manifest_name).read_bytes()), "data_root": str(root / "data")}
            (folder / "test_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            (folder / "frozen_model.json").write_text(json.dumps({"checkpoint": str(root / "outputs/prior/best.pt")}), encoding="utf-8")
        (folder / "best.pt").write_bytes(b"must not publish")
        (folder / "example.png").write_bytes(b"must not publish")
        return folder

    def test_exact_bytes_and_comparison_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_evaluation(root)
            output = root / "experiments/final/reserve"
            before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            index = archive_evaluation(source, output, project_root=root)
            for record in index["files"]:
                name = record["path"]
                self.assertEqual(record["original_sha256"], sha256(before[name]))
                self.assertEqual(record["archived_sha256"], sha256((output / name).read_bytes()))
                if record["transformation"] == "exact_bytes":
                    self.assertEqual(before[name], (output / name).read_bytes())
            self.assertFalse((output / "best.pt").exists())
            self.assertFalse((output / "example.png").exists())
            self.assertEqual(before, {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()})
            validate_alignment(load_evaluation(source, root), load_evaluation(output, root))

    def test_test_metadata_paths_are_sanitized_without_changing_manifest_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_evaluation(root, test=True)
            output = root / "experiments/final/test"
            index = archive_evaluation(source, output, project_root=root)
            config = json.loads((output / "evaluation_config.json").read_text())
            self.assertEqual(config["training_config"]["resume"], "outputs/prior/best.pt")
            self.assertEqual(config["external"], "external_path_redacted_1")
            self.assertEqual(config["source_hashes"], json.loads((output / "source_hashes.json").read_text()))
            self.assertTrue(index["test_used"])
            self.assertEqual(index["source_evaluation"], "outputs/evaluation")
            for name in ("evaluation_config.json", "test_provenance.json", "frozen_model.json", "archive_index.json"):
                self.assertNotIn(str(root), (output / name).read_text())
            validate_alignment(load_evaluation(source, root), load_evaluation(output, root))

    def test_rejects_overwrite_missing_nonfinite_and_unsafe_paths_before_copying(self):
        for issue in ("overwrite", "missing", "nonfinite", "traversal", "link"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.make_evaluation(root)
                output = root / "archive"
                if issue == "overwrite":
                    output.mkdir()
                elif issue == "missing":
                    (source / "metrics.json").unlink()
                elif issue == "nonfinite":
                    np.savez(source / "per_example_metrics.npz", bad=np.array([np.inf]))
                elif issue == "traversal":
                    (source / "source_hashes.json").write_text(json.dumps({"src/../../private.py": "0" * 64}))
                if issue == "link":
                    with patch.object(Path, "lstat") as mocked:
                        mocked.return_value.st_mode = 0o120777
                        with self.assertRaises(ValueError):
                            reject_links(source)
                else:
                    with self.assertRaises((ValueError, FileExistsError, FileNotFoundError)):
                        archive_evaluation(source, output, project_root=root)
                self.assertFalse((output / "archive_index.json").exists())
                if issue != "overwrite":
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
