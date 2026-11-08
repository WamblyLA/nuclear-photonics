import unittest

import torch

from src.metrics import pearson, ssim, global_ssim


class MetricTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.x = torch.rand(2, 4, 17, 19, generator=generator, dtype=torch.float64)

    def test_pearson_invariance_and_validity(self):
        values, valid = pearson(self.x, 0.3 * self.x + 0.2, return_valid=True)
        torch.testing.assert_close(values, torch.ones(2, 4, dtype=torch.float64))
        self.assertTrue(valid.all())
        torch.testing.assert_close(pearson(self.x, 1 - self.x), -torch.ones_like(values))
        values, valid = pearson(torch.zeros_like(self.x), self.x, return_valid=True)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(values).all())
        prediction = torch.zeros_like(self.x, requires_grad=True)
        (1 - pearson(prediction, self.x)).mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_ssim_identity_and_brightness(self):
        torch.testing.assert_close(ssim(self.x, self.x), torch.ones(2, 4, dtype=torch.float64))
        torch.testing.assert_close(global_ssim(self.x, self.x), torch.ones(2, 4, dtype=torch.float64))
        zero = torch.zeros_like(self.x)
        one = torch.ones_like(self.x)
        torch.testing.assert_close(ssim(zero, zero), torch.ones(2, 4, dtype=torch.float64))
        expected = torch.full((2, 4), 0.0001 / 1.0001, dtype=torch.float64)
        torch.testing.assert_close(ssim(zero, one), expected)

    def test_ssim_direct_weighted_window(self):
        # Independent single-window reference with explicit weighted moments.
        x = self.x[:1, :1, :11, :11]
        y = torch.flip(x, dims=(-1,)) * 0.8
        coordinates = torch.arange(-5, 6, dtype=torch.float64)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        weights = torch.exp(-(xx.square() + yy.square()) / (2 * 1.5**2))
        weights /= weights.sum()
        mean_x = (weights * x).sum()
        mean_y = (weights * y).sum()
        var_x = (weights * (x - mean_x).square()).sum()
        var_y = (weights * (y - mean_y).square()).sum()
        covariance = (weights * (x - mean_x) * (y - mean_y)).sum()
        expected = (2 * mean_x * mean_y + 0.0001) * (2 * covariance + 0.0009)
        expected /= (mean_x.square() + mean_y.square() + 0.0001) * (var_x + var_y + 0.0009)
        torch.testing.assert_close(ssim(x, y)[0, 0], expected, rtol=1e-10, atol=1e-12)

    def test_ssim_valid_window_and_gradients(self):
        prediction = self.x.clone().requires_grad_()
        score = ssim(prediction, 0.7 * self.x)
        score.mean().backward()
        self.assertEqual(score.shape, (2, 4))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        with self.assertRaises(ValueError):
            ssim(self.x[:, :, :10], self.x[:, :, :10])


if __name__ == "__main__":
    unittest.main()
