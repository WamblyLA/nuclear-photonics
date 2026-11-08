import unittest

import numpy as np
import torch

from src.data import ImageBank
from src.evaluate import reserve_manifest


class ReserveManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bank = ImageBank(seed=42)
        cls.selection = cls.bank.validation_manifest(per_group=64)
        cls.reserve = reserve_manifest(cls.bank, per_group=16, selection_per_group=64)

    def test_group_sizes_shapes_and_distinct_images(self):
        self.assertEqual(set(self.reserve), set(self.bank.names + ["mixed"]))
        for name, records in self.reserve.items():
            self.assertEqual(records.shape, (16, 4, 2))
            self.assertTrue(np.issubdtype(records.dtype, np.integer))
            for row in records:
                self.assertEqual(len(set(map(tuple, row))), 4)
            if name != "mixed":
                self.assertTrue((records[:, :, 0] == self.bank.names.index(name)).all())

    def test_no_source_overlap_with_any_selection_group(self):
        selected = set(map(tuple, np.concatenate(list(self.selection.values())).reshape(-1, 2)))
        for records in self.reserve.values():
            reserved = set(map(tuple, records.reshape(-1, 2)))
            self.assertFalse(selected & reserved)

    def test_reserved_sources_are_validation_only_and_not_training(self):
        records = np.concatenate(list(self.reserve.values())).reshape(-1, 2)
        selected = np.concatenate(list(self.selection.values())).reshape(-1, 2)
        for dataset in range(len(self.bank.names)):
            reserved_ids = set(records[records[:, 0] == dataset, 1].tolist())
            selected_ids = set(selected[selected[:, 0] == dataset, 1].tolist())
            allowed = set(self.bank.val[dataset].tolist()) - selected_ids
            self.assertTrue(reserved_ids <= allowed)
            self.assertFalse(reserved_ids & set(self.bank.train[dataset].tolist()))

    def test_deterministic_and_prefix_stable(self):
        self.bank.sample_train(32, np.random.default_rng(97531))
        repeated = reserve_manifest(self.bank, per_group=16, selection_per_group=64)
        extended = reserve_manifest(self.bank, per_group=24, selection_per_group=64)
        for name in self.reserve:
            np.testing.assert_array_equal(self.reserve[name], repeated[name])
            np.testing.assert_array_equal(self.reserve[name], extended[name][:16])


if __name__ == "__main__":
    unittest.main()
