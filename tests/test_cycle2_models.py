import math
import unittest

import torch

from src.models.networks import AttentionMixing, build_model, phase_from_logits


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_phase_forward_and_gradients(self):
        torch.manual_seed(3)
        model = build_model("attention", width=4, position="xy")
        images = torch.rand(2, 4, 128, 128, requires_grad=True)
        phase = model(images)
        self.assertEqual(phase.shape, (2, 1, 128, 128))
        self.assertLessEqual(phase.abs().max().item(), math.pi)
        phase.square().mean().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.global_mixing.position.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.global_mixing.attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_distant_tokens_influence_output(self):
        torch.manual_seed(4)
        mixing = AttentionMixing(16)
        features = torch.randn(1, 16, 32, 32, requires_grad=True)
        mixing(features)[0, 0, 0, 0].backward()
        distant_gradient = features.grad[:, :, -4:, -4:]
        self.assertGreater(distant_gradient.abs().sum().item(), 1e-6)

    def test_shared_weights_and_read_only_inference(self):
        model = build_model("attention", width=4).eval()
        baseline = build_model("cnn", width=4)
        result = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(key.startswith("global_mixing.") for key in result.missing_keys))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        images = torch.rand(1, 4, 32, 32)
        with torch.inference_mode():
            first, second = model(images), model(images)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_parameter_budget(self):
        attention = sum(p.numel() for p in build_model("attention", position="xy").parameters())
        fourier = sum(p.numel() for p in build_model("fourier", position="xy").parameters())
        self.assertLess(abs(attention - fourier) / fourier, 0.01)


class WrappedPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_periodicity_and_scalar_gradient(self):
        logits = torch.tensor([-3.2, -1.4, -0.3, 0.2, 1.3, 4.1], requires_grad=True)
        phase = phase_from_logits(logits, "wrapped")
        shifted = phase_from_logits(logits + 2, "wrapped")
        torch.testing.assert_close(torch.exp(1j * phase), torch.exp(1j * shifted), atol=3e-6, rtol=1e-6)
        phase.sum().backward()
        torch.testing.assert_close(logits.grad, torch.full_like(logits, math.pi))
        self.assertLessEqual(phase.abs().max().item(), math.pi)

    def test_complex_field_derivative_across_wrap(self):
        logits = torch.tensor([0.9999, 1.0001, -1.0001, -0.9999], requires_grad=True)
        phase = phase_from_logits(logits, "wrapped")
        field = torch.exp(1j * phase)
        field.imag.sum().backward()
        expected = math.pi * torch.cos(math.pi * logits.detach())
        torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-6)
        epsilon = 1e-3
        sides = phase_from_logits(torch.tensor([1 - epsilon, 1 + epsilon]), "wrapped")
        finite_difference = (torch.exp(1j * sides[1]) - torch.exp(1j * sides[0])) / (2 * epsilon)
        self.assertAlmostEqual(finite_difference.real.item(), 0, places=4)
        self.assertAlmostEqual(finite_difference.imag.item(), -math.pi, places=3)

    def test_parameter_parity_and_model_gradients(self):
        for name in ("fourier", "attention", "interleave"):
            with self.subTest(model=name):
                tanh = build_model(name, width=4)
                wrapped = build_model(name, width=4, phase_head="wrapped")
                wrapped.load_state_dict(tanh.state_dict(), strict=True)
                self.assertEqual(sum(p.numel() for p in tanh.parameters()), sum(p.numel() for p in wrapped.parameters()))
                phase = wrapped(torch.rand(1, 4, 32, 32))
                self.assertEqual(phase.shape, (1, 1, 32, 32))
                phase.cos().mean().backward()
                self.assertTrue(torch.isfinite(phase).all())
                for parameter in wrapped.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
