import unittest

import torch

from src.depth_frequency_loss import depth_frequency_loss
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator


class DepthFrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_valid_filter_matches_explicit_centered_windows(self):
        torch.manual_seed(111)
        raw = torch.rand(2, 4, 8, 9, dtype=torch.float64)
        target = torch.rand_like(raw)
        pred = raw / raw.square().mean((-2, -1), keepdim=True).sqrt()
        ref = target / target.square().mean((-2, -1), keepdim=True).sqrt()
        difference = pred - ref
        difference = difference - difference.mean(1, keepdim=True)
        values = []
        for row in range(2, 6):
            for column in range(2, 7):
                values.append(difference[..., row, column] - difference[..., row - 2:row + 3, column - 2:column + 3].mean((-2, -1)))
        expected = torch.stack(values).square().mean()
        torch.testing.assert_close(depth_frequency_loss(raw, target), expected, atol=1e-13, rtol=1e-13)

    def test_independent_brightness_scaling_and_joint_plane_permutation(self):
        torch.manual_seed(112)
        raw = torch.rand(2, 4, 15, 17, dtype=torch.float64)
        target = torch.rand_like(raw)
        prediction_scales = torch.tensor([[0.1, 0.3, 2.0, 4.0], [3.0, 0.2, 0.7, 1.3]], dtype=torch.float64)[..., None, None]
        target_scales = prediction_scales.flip(1) * 0.6
        expected = depth_frequency_loss(raw, target)
        torch.testing.assert_close(depth_frequency_loss(raw * prediction_scales, target * target_scales), expected, atol=1e-13, rtol=1e-13)
        permutation = [2, 0, 3, 1]
        torch.testing.assert_close(depth_frequency_loss(raw[:, permutation], target[:, permutation]), expected, atol=1e-13, rtol=1e-13)

    def test_common_mode_cancels_and_constant_plane_differences_have_no_edge_penalty(self):
        torch.manual_seed(113)
        raw = torch.rand(2, 1, 15, 15, dtype=torch.float64).expand(-1, 4, -1, -1)
        target = torch.rand(2, 1, 15, 15, dtype=torch.float64).expand(-1, 4, -1, -1)
        self.assertLess(float(depth_frequency_loss(raw, target)), 1e-28)
        raw = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float64)[None, :, None, None].expand(2, -1, 15, 15)
        target = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float64)[None, :, None, None].expand_as(raw)
        self.assertEqual(float(depth_frequency_loss(raw, target)), 0)

    def test_zero_safety_and_detached_targets(self):
        torch.manual_seed(114)
        for both_zero in (False, True):
            raw = torch.zeros(2, 4, 12, 12, requires_grad=True)
            target = (torch.zeros_like(raw) if both_zero else torch.rand_like(raw)).requires_grad_()
            loss = depth_frequency_loss(raw, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(raw.grad).all())
            self.assertIsNone(target.grad)
            if both_zero:
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(torch.count_nonzero(raw.grad)), 0)
            else:
                self.assertGreater(float(raw.grad.abs().sum()), 0)

    def test_finite_optical_backward_into_shared_neural_weights(self):
        torch.manual_seed(115)
        model = TransportPhaseGenerator(width=4, phase_decoder="complex")
        target = torch.rand(1, 4, 128, 128)
        amplitude = FresnelPropagator().amplitude(model(target))
        loss = depth_frequency_loss(amplitude, target)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(model.transport_target_encoder.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.transport_phase_encoder.output.real.weight.grad.abs().sum()), 0)

    def test_invalid_shapes_and_window_are_rejected(self):
        raw = torch.rand(1, 4, 12, 12)
        for arguments in ({"window_size": 4}, {"window_size": 13}, {"window_size": 1}, {"eps": 0}):
            with self.assertRaises(ValueError):
                depth_frequency_loss(raw, raw, **arguments)
        with self.assertRaises(ValueError):
            depth_frequency_loss(raw[:, :1], raw[:, :1])
        with self.assertRaises(ValueError):
            depth_frequency_loss(raw, raw.to(torch.uint8))


if __name__ == "__main__":
    unittest.main()
