import json
import tempfile
import unittest
from pathlib import Path

from scripts.archive_research import Config, archive_research, file_hash, public_values


class ResearchArchiveTests(unittest.TestCase):
    def make_run(self, root, name, protocol, fixed=0):
        folder = root / "outputs" / name
        folder.mkdir(parents=True)
        config = {"run": name, "model": "cnn", "steps": 4, "batch_size": 2, "protocol": protocol, "fixed_train_quartets": fixed, "resume": str(root / "outputs" / "previous" / "best.pt")}
        scalar = {"ssim": 0.2, "pearson": 0.4, "invalid_pearson_count": 0, "ssim_per_plane": [0.1, 0.2, 0.3, 0.2]}
        validation = {"protocol": protocol, "aggregate": scalar, "groups": {"mixed": scalar}, "tuples": 1}
        metrics = {**validation, "selected_step": 2, "selected_checkpoint": "best.pt", "training_seconds": 1.2, "total_seconds": 1.8}
        files = {
            "config.json": config,
            "metrics.json": metrics,
            "initial_metrics.json": validation,
            "source_hashes.json": {str(root / "src" / "model.py"): "0" * 64},
            "validation_tuples.json": {"mixed": [[[0, 1], [1, 2], [2, 3], [0, 4]]]},
        }
        for filename, value in files.items():
            (folder / filename).write_text(json.dumps(value), encoding="utf-8")
        rows = [{"step": 1, "loss": 1.2, "training_seconds": 0.2}, {"step": 2, "validation": validation, "training_seconds": 0.6}, {"step": 4, "loss": 1.0, "training_seconds": 1.2}, {"step": 4, "validation": validation, "training_seconds": 1.2}]
        if fixed:
            rows.append({"step": 4, "training_probe": validation})
        (folder / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (folder / "best.pt").write_bytes(b"fixture checkpoint; selected step is already in metrics")
        return folder

    def test_protocols_and_diagnostics_are_separate_without_mutating_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root, "r01_legacy", "fresnel_png_v1")
            self.make_run(root, "r02_peak", "fresnel_png_v2_peak")
            self.make_run(root, "r03_memorization", "fresnel_png_v2_peak", fixed=8)
            single = self.make_run(root, "r04_single_plane", "fresnel_png_v2_peak")
            single_config = json.loads((single / "config.json").read_text())
            single_config["diagnostic_only"] = True
            (single / "config.json").write_text(json.dumps(single_config))
            before = {path: file_hash(path) for path in (root / "outputs").rglob("*") if path.is_file()}
            config = Config(root / "outputs", root / "experiments" / "registry.json")
            result = archive_research(config, project_root=root)
            self.assertEqual(result["completed_runs"], 4)
            self.assertEqual(result["protocols"]["fresnel_png_v1"]["candidate_runs"], ["r01_legacy"])
            self.assertEqual(result["protocols"]["fresnel_png_v2_peak"]["candidate_runs"], ["r02_peak"])
            self.assertEqual(result["diagnostic_runs"], ["r03_memorization", "r04_single_plane"])
            record = result["runs"][0]
            self.assertEqual(record["selected_step"], 2)
            self.assertEqual(record["budget"]["completed_step"], 4)
            self.assertEqual(record["config"]["resume"], "outputs/previous/best.pt")
            self.assertEqual(list(record["source_hashes"]), ["src/model.py"])
            self.assertEqual([point["step"] for point in record["validation_curve"]], [0, 2, 4])
            self.assertNotIn("ssim_per_plane", record["validation_curve"][0]["aggregate"])
            self.assertIn("ssim_per_plane", record["metrics"]["aggregate"])
            self.assertEqual(record["training_log_summary"]["logged_rows"], 2)
            self.assertNotIn("training_log_samples", record)
            self.assertIn("training_probe_curve", result["runs"][2])
            self.assertEqual(before, {path: file_hash(path) for path in before})
            self.assertNotIn(str(root), json.dumps(result))
            first_sha = file_hash(config.output)
            archive_research(config, project_root=root)
            self.assertEqual(file_hash(config.output), first_sha)

    def test_incomplete_runs_remain_excluded_and_original_tree_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = self.make_run(root, "r01_incomplete", "fresnel_png_v2_peak")
            (folder / "metrics.json").unlink()
            result = archive_research(Config(root / "outputs", root / "registry.json"), project_root=root)
            self.assertEqual(result["completed_runs"], 0)
            self.assertEqual(result["not_archived"][0]["missing_files"], ["metrics.json"])
            with self.assertRaises(ValueError):
                archive_research(Config(root / "outputs", root / "outputs" / "registry.json"), project_root=root)

    def test_external_paths_and_nonfinite_values_are_not_published(self):
        redactions = []
        value = public_values({"resume": "Z:/private/model.pt"}, Path.cwd(), redactions)
        self.assertEqual(value["resume"], "external_path_redacted_1")
        self.assertEqual(len(redactions), 1)
        with self.assertRaises(ValueError):
            public_values({"ssim": float("nan")}, Path.cwd(), [])


if __name__ == "__main__":
    unittest.main()
