import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank, NAMES, stratified_split


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bank = ImageBank()

    def test_split_counts_and_no_overlap(self):
        for number, (train_count, val_count) in enumerate([(45000, 5000), (54000, 6000), (54000, 6000)]):
            train, val = self.bank.train[number], self.bank.val[number]
            self.assertEqual(len(train), train_count)
            self.assertEqual(len(val), val_count)
            self.assertEqual(len(np.intersect1d(train, val)), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(train_count + val_count))
            for label in np.unique(self.bank.labels[number]):
                total = np.sum(self.bank.labels[number] == label)
                self.assertLessEqual(abs(np.sum(self.bank.labels[number][val] == label) - total * 0.1), 1)

    def test_sampler_membership_reproducibility_and_no_repeats(self):
        records = self.bank.sample_train(100, np.random.default_rng(17))
        np.testing.assert_array_equal(records, self.bank.sample_train(100, np.random.default_rng(17)))
        for row in records:
            self.assertEqual(len(set(map(tuple, row))), 4)
        for dataset in range(3):
            self.assertTrue(np.isin(records[:, :, 1][records[:, :, 0] == dataset], self.bank.train[dataset]).all())
        for name, number in [("cifar", 0), ("mnist", 1), ("fashion", 2)]:
            subset = self.bank.sample_train(8, np.random.default_rng(23), domain=name)
            self.assertTrue((subset[:, :, 0] == number).all())

    def test_validation_independent_of_training_rng_and_prefix_stable(self):
        before = self.bank.validation_manifest(7)
        self.bank.sample_train(100, np.random.default_rng(999))
        after = self.bank.validation_manifest(12)
        for name in NAMES + ["mixed"]:
            np.testing.assert_array_equal(before[name], after[name][:7])
            for row in after[name]:
                self.assertEqual(len(set(map(tuple, row))), 4)
                for dataset, source in row:
                    self.assertIn(source, self.bank.val[dataset])

    def test_fetch_uint8_targets_and_record_order(self):
        records = self.bank.sample_train(2, np.random.default_rng(31))
        images = self.bank.fetch(records)
        self.assertEqual(tuple(images.shape), (2, 4, 128, 128))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(images.min().item(), 0)
        self.assertLessEqual(images.max().item(), 1)
        torch.testing.assert_close(images * 255, (images * 255).round(), atol=2e-5, rtol=0)
        order = [2, 0, 3, 1]
        torch.testing.assert_close(self.bank.fetch(records[:, order]), images[:, order])

    def test_manifest_is_serializable_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            signature = self.bank.save_manifest(path, per_group=3)
            self.assertEqual(signature, self.bank.save_manifest(path, per_group=3))
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(content["test_used"])
            self.assertEqual(len(content["validation"]["MNIST"]), 3)

    def test_rms_calibration_does_not_use_validation(self):
        saved = [images[indices].copy() for images, indices in zip(self.bank.images, self.bank.val)]
        cached_rms = self.bank._rms
        try:
            values = []
            for brightness in [0, 255]:
                for images, indices in zip(self.bank.images, self.bank.val):
                    images[indices] = brightness
                self.bank._rms = None
                values.append(self.bank.fixed_train_rms())
            self.assertEqual(values[0], values[1])
            self.assertGreater(values[0], 0)
            self.assertLess(values[0], 1)
        finally:
            for images, indices, original in zip(self.bank.images, self.bank.val, saved):
                images[indices] = original
            self.bank._rms = cached_rms

    def test_split_seed_determinism(self):
        labels = np.repeat(np.arange(10), 100)
        first = stratified_split(labels, 42)
        second = stratified_split(labels, 42)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
