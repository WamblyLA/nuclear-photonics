import json
import tempfile
import unittest
from pathlib import Path

from scripts.shortlist_checkpoints import Config, collect_shortlist, file_hash


class CheckpointShortlistTests(unittest.TestCase):
    def make_policy(self, root):
        path = root / "policy.json"
        path.write_text(json.dumps({"protocol": "fresnel_png_v2_peak", "minimum_selection_mean_pearson": 0.52, "minimum_mnist_foreground_brightness_ratio": 0.5}), encoding="utf-8")
        return path

    def make_run(self, root, name="r01_example", fixed=0, protocol="fresnel_png_v2_peak"):
        folder = root / "outputs" / name
        folder.mkdir(parents=True)
        config = {"run": name, "model": "example", "width": 4, "phase_head": "circular", "position": "xy", "protocol": protocol, "amplitude_scales": [1, 1, 1, 1], "steps": 3, "fixed_train_quartets": fixed}
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (folder / "metrics.json").write_text(json.dumps({"protocol": protocol}), encoding="utf-8")
        (folder / "validation_tuples.json").write_text('{"mixed": [[[0, 1], [1, 2], [2, 3], [0, 4]]]}', encoding="utf-8")
        rows = []
        for step, ssim, pearson in ((1, 0.45, 0.47), (2, 0.27, 0.56), (3, 0.26, 0.58)):
            aggregate = {"ssim": ssim, "pearson": pearson, "invalid_pearson_count": 0}
            rows.append({"step": step, "validation": {"protocol": protocol, "aggregate": aggregate, "groups": {"mixed": aggregate}}, "training_seconds": step * 0.1})
        (folder / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        metadata = {}
        for filename, step in (("best.pt", 1), ("best_pearson.pt", 3), ("last.pt", 3), ("step_000002.pt", 2)):
            path = folder / filename
            path.write_bytes((name + filename).encode())
            metadata[path.resolve()] = {"step": step, "config": config}
        return folder, metadata

    def test_keeps_later_good_checkpoints_when_best_ssim_fails_pearson(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, metadata = self.make_run(root)
            policy = self.make_policy(root)
            before = {path: file_hash(path) for path in folder.iterdir()}
            config = Config([folder], root / "shortlist.json", policy)
            result = collect_shortlist(config, root, lambda path: metadata[path.resolve()])
            self.assertEqual([item["step"] for item in result["candidates"]], [2, 3])
            self.assertEqual(result["candidates"][0]["checkpoint"], "outputs/r01_example/step_000002.pt")
            self.assertEqual(len(result["candidates"][1]["verified_same_step_aliases"]), 2)
            self.assertTrue(all(item["foreground_guard"] == "not evaluated" for item in result["candidates"]))
            self.assertFalse(result["winner_selected"])
            self.assertEqual(before, {path: file_hash(path) for path in before})
            self.assertNotIn(str(root), json.dumps(result))
            with self.assertRaises(FileExistsError):
                collect_shortlist(config, root, lambda path: metadata[path.resolve()])

    def test_mismatched_embedded_step_is_rejected_and_missing_step_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, metadata = self.make_run(root)
            path = (folder / "step_000002.pt").resolve()
            metadata[path] = {**metadata[path], "step": 3}
            result = collect_shortlist(Config([folder], root / "shortlist.json", self.make_policy(root)), root, lambda path: metadata[path.resolve()])
            self.assertEqual([item["step"] for item in result["candidates"]], [3])
            self.assertEqual(len(result["rejected_checkpoints"]), 1)
            self.assertEqual(result["eligible_steps_without_saved_checkpoint"][0]["step"], 2)

    def test_diagnostic_and_v1_runs_are_excluded_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic, _ = self.make_run(root, "r01_diagnostic", fixed=8)
            legacy, _ = self.make_run(root, "r02_legacy", protocol="fresnel_png_v1")
            def never_load(path):
                raise AssertionError("Excluded checkpoints must not be opened")
            result = collect_shortlist(Config([diagnostic, legacy], root / "shortlist.json", self.make_policy(root)), root, never_load)
            self.assertEqual(len(result["excluded_runs"]), 2)
            self.assertEqual(result["candidates"], [])

    def test_different_validation_manifests_cannot_be_ranked_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, metadata = self.make_run(root, "r01_first")
            second, second_metadata = self.make_run(root, "r02_second")
            metadata.update(second_metadata)
            (second / "validation_tuples.json").write_text('{"mixed": []}', encoding="utf-8")
            output = root / "shortlist.json"
            with self.assertRaisesRegex(ValueError, "different validation manifests"):
                collect_shortlist(Config([first, second], output, self.make_policy(root)), root, lambda path: metadata[path.resolve()])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
