import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.data import NAMES
from src.final_evaluate import FinalTestBank, TEST_NAMESPACE, make_final_manifest, used_source_counts


class FinalManifestTests(unittest.TestCase):
    def setUp(self):
        # Synthetic arrays only: no training or official test file is opened.
        self.bank = SimpleNamespace(images=[np.zeros((size, 2, 2), dtype=np.uint8) for size in (31, 37, 43)])

    def test_group_balance_distinct_sources_and_bounds(self):
        with patch.object(FinalTestBank, "__init__", side_effect=AssertionError("Test files must stay closed")):
            manifest = make_final_manifest(self.bank, 128)
        self.assertEqual(list(manifest), NAMES + ["mixed"])
        for group, records in manifest.items():
            self.assertEqual(records.shape, (128, 4, 2))
            for row in records:
                self.assertEqual(len({tuple(source) for source in row}), 4)
                for dataset, index in row:
                    self.assertTrue(0 <= dataset < 3)
                    self.assertTrue(0 <= index < len(self.bank.images[dataset]))
            if group in NAMES:
                self.assertTrue(np.all(records[:, :, 0] == NAMES.index(group)))
        self.assertEqual(set(manifest["mixed"][:, :, 0].ravel()), {0, 1, 2})

    def test_reproducible_seed_and_prefixes(self):
        first = make_final_manifest(self.bank, 16, seed=20261108)
        repeated = make_final_manifest(self.bank, 16, seed=20261108)
        longer = make_final_manifest(self.bank, 32, seed=20261108)
        different = make_final_manifest(self.bank, 16, seed=20261109)
        for name in first:
            np.testing.assert_array_equal(first[name], repeated[name])
            np.testing.assert_array_equal(first[name], longer[name][:16])
        self.assertTrue(any(not np.array_equal(first[name], different[name]) for name in first))

    def test_small_pool_and_group_count_validation(self):
        for value in (0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_final_manifest(self.bank, per_group=value)
        with self.assertRaises(ValueError):
            make_final_manifest(self.bank, seed=-1)
        bank = SimpleNamespace(images=[np.zeros((size, 2, 2), dtype=np.uint8) for size in (3, 4, 4)])
        with self.assertRaises(ValueError):
            make_final_manifest(bank)

    def test_source_counts_use_dataset_and_file_namespace(self):
        bank = SimpleNamespace(images=[np.zeros((4, 2, 2), dtype=np.uint8) for _ in NAMES])
        manifest = make_final_manifest(bank, 2)
        self.assertEqual(used_source_counts(manifest), {name: 4 for name in NAMES})
        self.assertEqual(TEST_NAMESPACE, "public_dataset_official_test")
        # Index zero in distinct dataset files is a different source, not leakage.
        self.assertTrue(all(0 in manifest[name][:, :, 1] for name in NAMES))


if __name__ == "__main__":
    unittest.main()
