import unittest

from src.training_schedule import cosine_lr_multiplier


class TrainingScheduleTests(unittest.TestCase):
    def test_no_warmup_endpoints_and_monotonic_decay(self):
        values = [cosine_lr_multiplier(step, 1000, min_multiplier=0.05) for step in range(1, 1001)]
        self.assertEqual(values[0], 1.0)
        self.assertAlmostEqual(values[-1], 0.05)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertTrue(all(0.05 <= value <= 1 for value in values))

    def test_warmup_and_absolute_step_reproducibility(self):
        values = [cosine_lr_multiplier(step, 100, 10, 0.1) for step in range(1, 101)]
        self.assertEqual(values[:10], [step / 10 for step in range(1, 11)])
        self.assertLess(values[10], 1)
        self.assertAlmostEqual(values[-1], 0.1)
        resumed = [cosine_lr_multiplier(step, 100, 10, 0.1) for step in range(58, 101)]
        self.assertEqual(resumed, values[57:])

    def test_edge_budgets_and_constant_decay_floor(self):
        self.assertEqual(cosine_lr_multiplier(1, 1), 1.0)
        self.assertEqual(cosine_lr_multiplier(2, 2, 1, 0.0), 0.0)
        self.assertTrue(all(cosine_lr_multiplier(step, 8, min_multiplier=1) == 1 for step in range(1, 9)))

    def test_invalid_configurations(self):
        for args in ((0, 10), (11, 10), (1, 0), (1, 10, -1), (1, 10, 10), (1, 10, 0, -0.1), (1, 10, 0, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cosine_lr_multiplier(*args)
        with self.assertRaises(TypeError):
            cosine_lr_multiplier(1.0, 10)


if __name__ == "__main__":
    unittest.main()
