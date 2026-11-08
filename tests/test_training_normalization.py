import unittest

import torch

from src.protocol import PEAK_PROTOCOL, prepare_reconstruction
from src.training_normalization import peak_normalize_surrogate


class TrainingNormalizationTests(unittest.TestCase):
    def test_forward_is_exactly_the_existing_peak_protocol_in_both_modes(self):
        torch.manual_seed(91)
        for dtype in (torch.float32, torch.float64):
            raw = torch.rand(3, 4, 11, 13, dtype=dtype) * 7
            raw[0, 0] = 0
            expected = prepare_reconstruction(raw, PEAK_PROTOCOL)
            for detached in (False, True):
                self.assertTrue(torch.equal(peak_normalize_surrogate(raw, detached), expected))

    def test_gradient_matches_exact_chain_rule_or_detached_denominator(self):
        values = torch.tensor([[[[0.2, 0.8], [0.3, 0.1]]]], dtype=torch.float64)
        weights = torch.tensor([[[[1.1, 2.0], [3.2, -0.7]]]], dtype=torch.float64)
        maximum = values.max()
        peak = (values == maximum).to(values.dtype)
        for detached in (False, True):
            raw = values.clone().requires_grad_()
            loss = (peak_normalize_surrogate(raw, detached) * weights).sum()
            gradient, = torch.autograd.grad(loss, raw)
            expected = weights / maximum
            if not detached:
                expected = expected - peak * (weights * values).sum() / maximum.square()
            torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)

    def test_forward_scale_invariance_but_detached_backward_is_biased(self):
        base = torch.tensor([[[[0.2, 0.8], [0.3, 0.1]]]], dtype=torch.float64)
        expected_value = (base / base.max()).sum()
        for detached in (False, True):
            scale = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
            value = peak_normalize_surrogate(base * scale, detached).sum()
            gradient, = torch.autograd.grad(value, scale)
            torch.testing.assert_close(value, expected_value, atol=1e-12, rtol=1e-12)
            expected_gradient = expected_value / scale.detach() if detached else torch.zeros_like(scale)
            torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)

    def test_fixed_energy_mse_surrogate_matches_amplitude_overlap_gradient(self):
        # A two-coordinate positive amplitude on a fixed-energy circle. This
        # tests a derivative identity; no optical mask is optimized or fitted.
        angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        amplitude = 1.3 * torch.stack((angle.cos(), angle.sin())).reshape(1, 1, 1, 2)
        target = torch.tensor([[[[0.2, 0.7]]]], dtype=torch.float64)
        loss = (peak_normalize_surrogate(amplitude, True) - target).square().mean()
        gradient, = torch.autograd.grad(loss, angle, retain_graph=True)
        overlap_gradient, = torch.autograd.grad((amplitude * target).sum(), angle, retain_graph=True)
        expected = -2 * overlap_gradient / (amplitude.numel() * amplitude.detach().max())
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
        quadratic = amplitude.square().mean() / amplitude.detach().max().square()
        quadratic_gradient, = torch.autograd.grad(quadratic, angle)
        torch.testing.assert_close(quadratic_gradient, torch.zeros_like(angle), atol=1e-12, rtol=1e-12)

    def test_zero_images_have_finite_zero_gradients_and_dtype_validation(self):
        for dtype in (torch.float32, torch.float64):
            for detached in (False, True):
                raw = torch.zeros(2, 4, 8, 8, dtype=dtype, requires_grad=True)
                normalized = peak_normalize_surrogate(raw, detached)
                self.assertEqual(int(torch.count_nonzero(normalized)), 0)
                normalized.sum().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertEqual(int(torch.count_nonzero(raw.grad)), 0)
        with self.assertRaises(TypeError):
            peak_normalize_surrogate(torch.zeros(1, 4, 8, 8, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
