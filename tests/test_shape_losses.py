import math
import unittest

import torch

from src.shape_losses import excess_concentration_loss, log_lp_rms


class ShapeLossTests(unittest.TestCase):
    def test_known_uniform_and_single_hot_pixel_concentration(self):
        image = torch.ones(1, 4, 8, 8, dtype=torch.float64)
        torch.testing.assert_close(log_lp_rms(image), torch.zeros(1, 4, dtype=torch.float64))
        image.zero_()
        image[:, :, 3, 2] = 1
        expected = (0.5 - 1 / 8) * math.log(64)
        torch.testing.assert_close(log_lp_rms(image), torch.full((1, 4), expected, dtype=torch.float64))

    def test_scale_invariance_and_zero_for_matching_shapes(self):
        torch.manual_seed(1)
        target = torch.rand(2, 4, 8, 8, dtype=torch.float64)
        prediction = target.pow(5)
        scales = torch.tensor((0.001, 0.4, 7.0, 1000.0), dtype=torch.float64)[None, :, None, None]
        torch.testing.assert_close(excess_concentration_loss(prediction, target), excess_concentration_loss(prediction * scales, target / scales))
        torch.testing.assert_close(excess_concentration_loss(target * scales, target), torch.zeros(2, 4, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_spikes_penalized_but_lower_concentration_allowed(self):
        target = torch.zeros(1, 4, 8, 8)
        target[:, :, 2:6, 2:6] = 1
        spike = torch.zeros_like(target)
        spike[:, :, 3, 3] = 1
        self.assertGreater(float(excess_concentration_loss(spike, target).mean()), 0)
        torch.testing.assert_close(excess_concentration_loss(torch.ones_like(target), target), torch.zeros(1, 4))
        torch.testing.assert_close(excess_concentration_loss(spike, target, margin=10), torch.zeros(1, 4))

    def test_finite_gradient_and_target_detach_including_zero_profiles(self):
        target = torch.rand(1, 4, 8, 8, requires_grad=True)
        with torch.no_grad():
            target[:, 0] = 0
        prediction = torch.rand_like(target).pow(6).requires_grad_()
        loss = excess_concentration_loss(prediction, target)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[:, 1:].abs().sum()), 0)
        self.assertIsNone(target.grad)
        zero = torch.zeros(1, 4, 8, 8, requires_grad=True)
        excess_concentration_loss(zero, target).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_stable_scaling_gradient_matches_finite_difference(self):
        image = torch.tensor([[[[0.4, 0.2], [0.1, 0.9]]]], dtype=torch.float64, requires_grad=True)
        log_lp_rms(image).sum().backward()
        step = 1e-6
        for index in range(4):
            plus, minus = image.detach().clone(), image.detach().clone()
            plus.reshape(-1)[index] += step
            minus.reshape(-1)[index] -= step
            numeric = (log_lp_rms(plus) - log_lp_rms(minus)) / (2 * step)
            self.assertAlmostEqual(float(image.grad.reshape(-1)[index]), float(numeric.item()), places=6)

    def test_invalid_arguments(self):
        image = torch.ones(1, 4, 4, 4)
        with self.assertRaises(ValueError):
            log_lp_rms(image, p=2)
        with self.assertRaises(ValueError):
            excess_concentration_loss(image, image, margin=-1)
        with self.assertRaises(TypeError):
            log_lp_rms(image.to(torch.uint8))


if __name__ == "__main__":
    unittest.main()
