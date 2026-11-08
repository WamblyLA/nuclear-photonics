import unittest

import torch

from src.amplitude_losses import amplitude_cosine_loss, balanced_pixel_mse


class AmplitudeLossTests(unittest.TestCase):
    def test_cosine_is_scale_invariant_and_matching_targets_have_zero_loss(self):
        torch.manual_seed(17)
        target = torch.rand(2, 4, 8, 8, dtype=torch.float64)
        prediction = torch.rand_like(target)
        scales = torch.tensor((0.01, 0.3, 2.0, 100.0), dtype=torch.float64)[None, :, None, None]
        torch.testing.assert_close(amplitude_cosine_loss(prediction, target), amplitude_cosine_loss(prediction * scales, target / scales))
        torch.testing.assert_close(amplitude_cosine_loss(target * scales, target), torch.zeros(2, 4, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_wrong_hot_pixel_and_wasted_energy_are_penalized(self):
        target = torch.zeros(1, 4, 8, 8)
        target[:, :, 1, 1] = 1
        wrong = torch.zeros_like(target)
        wrong[:, :, 6, 6] = 100
        torch.testing.assert_close(amplitude_cosine_loss(wrong, target), torch.ones(1, 4))
        wasted = target.clone()
        wasted[:, :, 6, 6] = 1
        torch.testing.assert_close(amplitude_cosine_loss(wasted, target), torch.full((1, 4), 1 - 2**-0.5))

    def test_zero_target_is_ignored_but_zero_prediction_can_learn(self):
        prediction = torch.zeros(1, 4, 4, 4, requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=True)
        with torch.no_grad():
            target[:, 1:, 1, 1] = 1
        loss = amplitude_cosine_loss(prediction, target)
        torch.testing.assert_close(loss, torch.tensor(((0.0, 1.0, 1.0, 1.0),)))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[:, 1:].abs().sum()), 0)
        self.assertIsNone(target.grad)

    def test_balanced_mse_gives_equal_region_weight(self):
        target = torch.zeros(1, 4, 4, 4)
        target[:, :, 0, 0] = 1
        prediction = torch.zeros_like(target)
        torch.testing.assert_close(balanced_pixel_mse(prediction, target), torch.full((1, 4), 0.5))
        self.assertLess(float((prediction - target).square().mean()), float(balanced_pixel_mse(prediction, target).mean()))

    def test_balanced_mse_handles_absent_regions_and_detaches_target(self):
        for value in (0.0, 1.0):
            target = torch.full((2, 4, 4, 4), value, requires_grad=True)
            prediction = torch.full_like(target, 0.25, requires_grad=True)
            loss = balanced_pixel_mse(prediction, target)
            torch.testing.assert_close(loss, torch.full((2, 4), (0.25 - value)**2))
            loss.sum().backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertIsNone(target.grad)
        target = torch.rand(2, 4, 4, 4)
        torch.testing.assert_close(balanced_pixel_mse(target, target), torch.zeros(2, 4))

    def test_invalid_arguments(self):
        images = torch.ones(1, 4, 4, 4)
        for value in (0, -1, float("nan")):
            with self.subTest(eps=value), self.assertRaises(ValueError):
                amplitude_cosine_loss(images, images, eps=value)
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                balanced_pixel_mse(images, images, threshold=value)
        with self.assertRaises(ValueError):
            amplitude_cosine_loss(images[0], images)
        with self.assertRaises(TypeError):
            balanced_pixel_mse(images.to(torch.uint8), images)


if __name__ == "__main__":
    unittest.main()
