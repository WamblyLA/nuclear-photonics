import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank
from src.evaluate import manifest_records, reserve_manifest


class Cycle2ReserveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bank = ImageBank(seed=42)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "old_reserve.json"
        cls.old = reserve_manifest(cls.bank, per_group=448)
        cls.path.write_text(json.dumps({name: values.tolist() for name, values in cls.old.items()}), encoding="utf-8")
        cls.fresh, cls.provenance = reserve_manifest(cls.bank, per_group=128, exclude_manifests=[cls.path], manifest_seed=40_042, return_provenance=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_no_overlap_with_selection_previous_reserve_or_training(self):
        used = np.concatenate((manifest_records(self.old, self.bank), manifest_records(self.bank.validation_manifest(64), self.bank)))
        previous = set(map(tuple, used))
        fresh = set(map(tuple, manifest_records(self.fresh, self.bank)))
        self.assertFalse(fresh & previous)
        for dataset, index in fresh:
            self.assertIn(index, self.bank.val[dataset])
            self.assertNotIn(index, self.bank.train[dataset])

    def test_size_and_distinct_images_per_tuple(self):
        for records in self.fresh.values():
            self.assertEqual(records.shape, (128, 4, 2))
            for row in records:
                self.assertEqual(len(set(map(tuple, row))), 4)

    def test_provenance_matches_actual_sources(self):
        self.assertEqual(self.provenance["manifest_seed"], 40_042)
        self.assertEqual(self.provenance["exclude_manifests"][0]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        records = manifest_records(self.fresh, self.bank)
        for dataset, name in enumerate(self.bank.names):
            self.assertEqual(self.provenance["sampled_unique_sources"][name], len(np.unique(records[records[:, 0] == dataset, 1])))
            self.assertEqual(self.provenance["available_validation_sources"][name], len(self.bank.val[dataset]) - self.provenance["excluded_unique_sources"][name])

    def test_repeated_exclusions_deterministic_prefix_and_seed(self):
        repeated = reserve_manifest(self.bank, per_group=132, exclude_manifests=[self.path, self.path], manifest_seed=40_042)
        changed = reserve_manifest(self.bank, per_group=128, exclude_manifests=[self.path], manifest_seed=50_042)
        for name in self.fresh:
            np.testing.assert_array_equal(self.fresh[name], repeated[name][:128])
            self.assertFalse(np.array_equal(self.fresh[name], changed[name]))

    def test_zero_selection_is_supported(self):
        result = reserve_manifest(self.bank, per_group=2, selection_per_group=0)
        self.assertEqual(result["mixed"].shape, (2, 4, 2))

    def test_invalid_or_exhausted_exclusions_fail(self):
        bad = Path(self.directory.name) / "bad.json"
        bad.write_text(json.dumps({"group": [[[0, -1]] * 4]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            reserve_manifest(self.bank, per_group=2, exclude_manifests=[bad])
        values = np.stack((np.zeros(5000, dtype=int), self.bank.val[0]), axis=1).reshape(-1, 4, 2)
        bad.write_text(json.dumps({"group": values.tolist()}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "At least four"):
            reserve_manifest(self.bank, per_group=2, exclude_manifests=[bad])


if __name__ == "__main__":
    unittest.main()
