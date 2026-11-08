import math
import unittest

import torch
from torch import nn

from src.models.phase_bias import SharedPhaseBias


class SmallPhaseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(4, 1, 1)

    def forward(self, images, return_diagnostics=False):
        phase = math.pi * self.output(images).tanh()
        return (phase, {"base_diagnostic": phase.detach().mean()}) if return_diagnostics else phase


class SharedPhaseBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_zero_initialization_preserves_complex_field_and_diagnostics(self):
        torch.manual_seed(42)
        base = SmallPhaseGenerator()
        model = SharedPhaseBias(base)
        images = torch.rand(2, 4, 128, 128)
        original = base(images)
        result, diagnostics = model(images, return_diagnostics=True)
        self.assertEqual(result.shape, (2, 1, 128, 128))
        self.assertEqual(model.phase_bias.numel(), 128 * 128)
        self.assertTrue(torch.count_nonzero(model.phase_bias) == 0)
        self.assertTrue(torch.allclose(torch.exp(1j * original), torch.exp(1j * result), atol=3e-7, rtol=0))
        self.assertIn("base_diagnostic", diagnostics)
        self.assertEqual(float(diagnostics["shared_phase_bias_rms"]), 0)

    def test_joint_batch_training_updates_shared_bias_and_base_weights(self):
        torch.manual_seed(43)
        model = SharedPhaseBias(SmallPhaseGenerator())
        images = torch.rand(3, 4, 128, 128)
        weights_before = model.base_generator.output.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        phase = model(images)
        (torch.cos(phase) * torch.randn_like(phase)).mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer.step()
        self.assertGreater(float(model.phase_bias.detach().abs().sum()), 0)
        self.assertFalse(torch.equal(weights_before, model.base_generator.output.weight))
        self.assertEqual(model.phase_bias.shape, (1, 1, 128, 128))

    def test_large_bias_wraps_without_inference_state_mutation(self):
        torch.manual_seed(44)
        model = SharedPhaseBias(SmallPhaseGenerator()).eval()
        with torch.no_grad():
            model.phase_bias.copy_(50 * torch.randn_like(model.phase_bias))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        images = torch.rand(2, 4, 128, 128)
        with torch.no_grad():
            phase = model(images)
            repeated = model(images)
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(float(phase.abs().max()), math.pi + 1e-6)
        self.assertTrue(torch.equal(phase, repeated))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_wrapping_preserves_field_at_branch_cut(self):
        phase = torch.tensor([-math.pi, math.pi, 3 * math.pi, -3 * math.pi], requires_grad=True)
        wrapped = torch.atan2(phase.sin(), phase.cos())
        field = torch.exp(1j * wrapped)
        self.assertTrue(torch.allclose(field, torch.exp(1j * phase), atol=3e-7, rtol=0))
        field.imag.sum().backward()
        self.assertTrue(torch.isfinite(phase.grad).all())
        self.assertTrue(torch.allclose(phase.grad, phase.detach().cos(), atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
