import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.compare_candidates import compare_evaluations, file_sha256, load_evaluation, paired_bootstrap, validate_alignment


class CompareCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def make_evaluation(self, name, shift=0.0):
        path = self.root / name
        path.mkdir()
        manifest = {}
        arrays = {}
        groups = {}
        for number, (group, count) in enumerate((("first", 3), ("second", 5))):
            rows = np.stack((np.full((count, 4), number), np.arange(count * 4).reshape(count, 4)), axis=-1)
            manifest[group] = rows.tolist()
            values = np.arange(count * 4).reshape(count, 4) / 100 + shift
            arrays[group + "__ssim"] = values
            arrays[group + "__pearson"] = 0.2 + values
            arrays[group + "__pearson_valid"] = np.ones((count, 4), dtype=bool)
            groups[group] = {metric: float(arrays[group + "__" + metric].mean()) for metric in ("ssim", "pearson")}
        self.write_json(path / "validation_tuples.json", manifest)
        np.savez_compressed(path / "per_example_metrics.npz", **arrays)
        self.write_json(path / "metrics.json", {
            "aggregate": {metric: float(np.concatenate([arrays[group + "__" + metric] for group in groups]).mean()) for metric in ("ssim", "pearson")},
            "groups": groups, "tuples": 8, "protocol": "fresnel_png_v2_peak",
            "model_input_protocol": "fresnel_png_v2_peak", "test_used": False,
        })
        self.write_json(path / "config.json", {"data_seed": 42, "protocol": "fresnel_png_v2_peak"})
        source = {"src/data.py": "PREPROCESS = 'synthetic'\n", "src/protocol.py": "PROTOCOL = 'peak'\n",
                  "src/metrics.py": "METRIC = 'synthetic'\n", "src/physics/fresnel.py": "PHYSICS = 'synthetic'\n",
                  "src/train.py": "\n\n".join(f"def {function}():\n    pass" for function in ("evaluate", "png_roundtrip", "image_metrics", "summarize"))}
        hashes = {}
        for relative, text in source.items():
            file = path / "source" / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
            hashes[relative] = file_sha256(file)
        self.write_json(path / "source_hashes.json", hashes)
        return path

    def load(self, path):
        return load_evaluation(path, repo_root=self.root)

    def test_valid_artifacts_constant_shift_and_portable_provenance(self):
        left = self.load(self.make_evaluation("reference"))
        right = self.load(self.make_evaluation("candidate", shift=0.125))
        result = compare_evaluations(left, [right])
        for scope in (result["comparisons"][0]["aggregate"], *result["comparisons"][0]["groups"].values()):
            for metric in scope["metrics"].values():
                self.assertAlmostEqual(metric["difference"], 0.125)
                self.assertAlmostEqual(metric["ci_low"], 0.125)
                self.assertAlmostEqual(metric["ci_high"], 0.125)
        self.assertEqual(result["reference"]["directory"], "reference")
        self.assertTrue(all(not Path(file["path"]).is_absolute() for file in result["reference"]["files"]))
        self.assertIn("not independent-source", result["method"]["interpretation"])
        self.assertEqual(len(result["reference"]["manifest_sha256"]), 64)

    def test_rows_and_plane_permutations_rejected_but_json_formatting_ignored(self):
        left = self.load(self.make_evaluation("reference"))
        path = self.make_evaluation("candidate")
        original = json.loads((path / "validation_tuples.json").read_text())
        (path / "validation_tuples.json").write_text(json.dumps(original, indent=4, sort_keys=True), encoding="utf-8")
        validate_alignment(left, self.load(path))
        for permutation in ("row", "plane"):
            manifest = json.loads(json.dumps(original))
            if permutation == "row":
                manifest["first"] = manifest["first"][::-1]
            else:
                manifest["first"][0] = manifest["first"][0][::-1]
            self.write_json(path / "validation_tuples.json", manifest)
            with self.subTest(permutation=permutation), self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_alignment(left, self.load(path))

    def test_protocol_and_input_preprocessing_mismatch_rejected(self):
        left = self.load(self.make_evaluation("reference"))
        for changed in ("protocol", "model_input_protocol", "source"):
            path = self.make_evaluation(changed)
            metrics = json.loads((path / "metrics.json").read_text())
            config = json.loads((path / "config.json").read_text())
            if changed == "protocol":
                metrics["protocol"] = "fresnel_png_v1"
                config["amplitude_scales"] = [1, 1, 1, 1]
            elif changed == "model_input_protocol":
                metrics["model_input_protocol"] = "fresnel_png_v1"
                config["protocol"] = "fresnel_png_v1"
            else:
                source = path / "source/src/data.py"
                source.write_text("PREPROCESS = 'different'\n", encoding="utf-8")
                hashes = json.loads((path / "source_hashes.json").read_text())
                hashes["src/data.py"] = file_sha256(source)
                self.write_json(path / "source_hashes.json", hashes)
            self.write_json(path / "metrics.json", metrics)
            self.write_json(path / "config.json", config)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Incompatible evaluation identity"):
                validate_alignment(left, self.load(path))

    def test_invalid_pearson_summary_and_snapshot_cannot_pass(self):
        for issue in ("pearson", "summary", "source"):
            path = self.make_evaluation(issue)
            if issue == "pearson":
                with np.load(path / "per_example_metrics.npz") as saved:
                    arrays = {key: saved[key] for key in saved.files}
                arrays["first__pearson_valid"][0, 0] = False
                np.savez_compressed(path / "per_example_metrics.npz", **arrays)
            elif issue == "summary":
                metrics = json.loads((path / "metrics.json").read_text())
                metrics["groups"]["first"]["ssim"] += 0.1
                self.write_json(path / "metrics.json", metrics)
            else:
                (path / "source/src/metrics.py").write_text("CHANGED = True\n", encoding="utf-8")
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                self.load(path)

    def test_quartet_not_individual_plane_is_resampling_unit(self):
        left = {"one": {metric: np.zeros((6, 4)) for metric in ("ssim", "pearson")}}
        # Every quartet has mean 0.1 despite substantial plane differences.
        planes = np.array([0.4, -0.4, 0.2, -0.2])
        right = {"one": {metric: np.stack([np.roll(planes, i) + 0.1 for i in range(6)]) for metric in ("ssim", "pearson")}}
        result = paired_bootstrap(left, right)
        for metric in result["aggregate"]["metrics"].values():
            self.assertAlmostEqual(metric["difference"], 0.1)
            self.assertAlmostEqual(metric["ci_low"], 0.1)
            self.assertAlmostEqual(metric["ci_high"], 0.1)
            self.assertLess(metric["bootstrap_std"], 1e-14)

    def test_fixed_strata_sizes_and_quartet_weights(self):
        left = {group: {metric: np.zeros((count, 4)) for metric in ("ssim", "pearson")} for group, count in (("one", 1), ("three", 3))}
        right = {group: {metric: np.full_like(values, 0.1 if group == "one" else 0.3) for metric, values in scores.items()} for group, scores in left.items()}
        result = paired_bootstrap(left, right)
        self.assertEqual(result["aggregate"]["quartets"], 4)
        for metric in result["aggregate"]["metrics"].values():
            self.assertAlmostEqual(metric["difference"], 0.25)
            self.assertAlmostEqual(metric["ci_low"], 0.25)
            self.assertAlmostEqual(metric["ci_high"], 0.25)

    def test_reproducible_joint_metric_resampling(self):
        rng = np.random.default_rng(42)
        difference = rng.normal(0.05, 0.2, (40, 4))
        left = {"one": {"ssim": np.zeros_like(difference), "pearson": np.zeros_like(difference)}}
        right = {"one": {"ssim": difference, "pearson": 2 * difference}}
        first = paired_bootstrap(left, right)
        self.assertEqual(first, paired_bootstrap(left, right))
        self.assertNotEqual(first, paired_bootstrap(left, right, seed=99))
        scores = first["aggregate"]["metrics"]
        for key in ("difference", "ci_low", "ci_high", "bootstrap_std"):
            self.assertAlmostEqual(scores["pearson"][key], 2 * scores["ssim"][key])

    def test_pair_shape_and_invalid_bootstrap_arguments_rejected(self):
        values = {"one": {metric: np.zeros((4, 4)) for metric in ("ssim", "pearson")}}
        other = {"one": {metric: np.zeros((3, 4)) for metric in ("ssim", "pearson")}}
        with self.assertRaisesRegex(ValueError, "row counts differ"):
            paired_bootstrap(values, other)
        for kwargs in ({"repetitions": 1}, {"seed": -1}, {"confidence": 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_bootstrap(values, values, **kwargs)


if __name__ == "__main__":
    unittest.main()
