import unittest

import torch

from src.curriculum import sample_forward_targets
from src.physics.fresnel import FresnelPropagator


class ForwardCurriculumTests(unittest.TestCase):
    def test_targets_are_deterministic_png_amplitudes_without_labels(self):
        torch.set_num_threads(2)
        physics = FresnelPropagator()
        scales = (0.44 / physics.field_rms.float()).reshape(1, 4, 1, 1)
        first, stats = sample_forward_targets(2, physics, scales, torch.Generator().manual_seed(123))
        second, _ = sample_forward_targets(2, physics, scales, torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 4, 128, 128))
        self.assertTrue(torch.equal(first * 255, (first * 255).round()))
        self.assertFalse(first.requires_grad)
        self.assertGreaterEqual(float(first.min()), 0)
        self.assertLessEqual(float(first.max()), 1)
        self.assertGreater(stats["synthetic_target_rms"], 0)
        self.assertLessEqual(stats["synthetic_target_rms"], 0.441)
        different, _ = sample_forward_targets(2, physics, scales, torch.Generator().manual_seed(124))
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":
    unittest.main()
