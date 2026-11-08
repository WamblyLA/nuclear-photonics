import unittest

import numpy as np

from src.data import ImageBank
from src.training_probe import fixed_training_manifest


class FakeBank:
    names = ["CIFAR10", "MNIST", "FashionMNIST"]

    def __init__(self, sources=200):
        self.train = [np.arange(sources, dtype=np.int64) + dataset * 1000 for dataset in range(3)]
        self.domains = []

    def _sample(self, batch_size, rng, pools, domain):
        self.domains.append(domain)
        return ImageBank._sample(self, batch_size, rng, pools, domain)

    @property
    def val(self):
        raise AssertionError("Validation sources must not be accessed")

    @property
    def test(self):
        raise AssertionError("Test sources must not be accessed")

    @property
    def images(self):
        raise AssertionError("Manifest construction must not load image pixels")


class TrainingProbeTests(unittest.TestCase):
    def test_balanced_groups_unique_train_sources_and_independent_mixed_sampler(self):
        bank = FakeBank()
        groups = fixed_training_manifest(bank)
        self.assertEqual(list(groups), bank.names + ["mixed"])
        records = np.concatenate(list(groups.values()))
        self.assertEqual(records.shape, (32, 4, 2))
        self.assertEqual(records.dtype, np.int64)
        self.assertEqual(len(set(map(tuple, records.reshape(-1, 2)))), 128)
        for number, name in enumerate(bank.names):
            self.assertEqual(groups[name].shape, (8, 4, 2))
            self.assertTrue((groups[name][:, :, 0] == number).all())
            selected = records[:, :, 1][records[:, :, 0] == number]
            self.assertTrue(np.isin(selected, bank.train[number]).all())
        self.assertEqual(bank.domains, ["cifar"] * 8 + ["mnist"] * 8 + ["fashion"] * 8 + ["independent"] * 8)

    def test_seed_determinism_and_bank_pools_unchanged(self):
        bank = FakeBank()
        originals = [pool.copy() for pool in bank.train]
        identities = [id(pool) for pool in bank.train]
        first = fixed_training_manifest(bank, seed=7)
        second = fixed_training_manifest(bank, seed=7)
        different = fixed_training_manifest(bank, seed=8)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.array_equal(first["mixed"], different["mixed"]))
        for index, original in enumerate(originals):
            np.testing.assert_array_equal(bank.train[index], original)
            self.assertEqual(id(bank.train[index]), identities[index])

    def test_returned_arrays_do_not_alias_original_pools(self):
        bank = FakeBank()
        before = [pool.copy() for pool in bank.train]
        manifest = fixed_training_manifest(bank, per_group=2)
        manifest["CIFAR10"][:] = -1
        for original, actual in zip(before, bank.train):
            np.testing.assert_array_equal(original, actual)

    def test_insufficient_sources_rejected_without_mutating_bank(self):
        for sources in (31, 32):
            bank = FakeBank(sources=sources)
            original = [pool.copy() for pool in bank.train]
            with self.assertRaisesRegex(ValueError, "Not enough distinct TRAIN sources"):
                fixed_training_manifest(bank)
            for first, second in zip(original, bank.train):
                np.testing.assert_array_equal(first, second)

    def test_bad_arguments_and_duplicate_sources_rejected(self):
        for per_group in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                fixed_training_manifest(FakeBank(), per_group=per_group)
        bank = FakeBank()
        bank.train[0][1] = bank.train[0][0]
        with self.assertRaisesRegex(ValueError, "unique nonnegative"):
            fixed_training_manifest(bank)


if __name__ == "__main__":
    unittest.main()
