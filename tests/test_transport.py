import math
import unittest
from unittest.mock import patch

import torch

from src.models.transport import TransportPhaseGenerator, UnitaryFresnelTransport
from src.physics.fresnel import FresnelPropagator


class UnitaryTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_adjoint_and_two_sided_unitary_roundtrip(self):
        torch.manual_seed(11)
        x = torch.complex(torch.randn(2, 4, 8, 8, dtype=torch.float64), torch.randn(2, 4, 8, 8, dtype=torch.float64))
        y = torch.complex(torch.randn_like(x.real), torch.randn_like(x.real))
        for gauge in (True, False):
            operator = UnitaryFresnelTransport(size=8, include_output_chirp=gauge)
            forward = operator(x)
            adjoint = operator.adjoint(y)
            self.assertTrue(torch.allclose(torch.vdot(forward.flatten(), y.flatten()), torch.vdot(x.flatten(), adjoint.flatten()), atol=2e-11, rtol=2e-12))
            self.assertTrue(torch.allclose(operator.adjoint(forward), x, atol=2e-12, rtol=2e-12))
            self.assertTrue(torch.allclose(operator(adjoint), y, atol=2e-12, rtol=2e-12))

    def test_matches_physical_forward_after_removing_positive_prefactors(self):
        torch.manual_seed(12)
        physics = FresnelPropagator(size=8)
        phase = torch.randn(2, 8, 8, dtype=torch.float64)
        source = torch.exp(1j * phase[:, None]).expand(-1, 4, -1, -1)
        expected = physics(phase) / physics.field_rms[None, :, None, None]
        full = UnitaryFresnelTransport(size=8)(source)
        gauge = UnitaryFresnelTransport(size=8, include_output_chirp=False)(source)
        self.assertTrue(torch.allclose(full, expected, atol=2e-12, rtol=2e-12))
        self.assertTrue(torch.allclose(full.abs(), gauge.abs(), atol=2e-12, rtol=2e-12))

    def test_transport_validation(self):
        operator = UnitaryFresnelTransport(size=8)
        with self.assertRaises(TypeError):
            operator.adjoint(torch.ones(1, 4, 8, 8))
        with self.assertRaises(ValueError):
            operator.adjoint(torch.ones(1, 1, 8, 8, dtype=torch.complex64))


class TransportGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_all_heads_and_both_feature_gauges_have_finite_optical_gradients(self):
        physics = FresnelPropagator()
        for head in ("tanh", "circular", "wrapped"):
            for gauge in (True, False):
                torch.manual_seed(42)
                model = TransportPhaseGenerator(width=4, phase_head=head, include_output_chirp=gauge)
                target = torch.rand(1, 4, 128, 128, requires_grad=True)
                phase = model(target)
                self.assertEqual(phase.shape, (1, 1, 128, 128))
                self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
                amplitude = physics.amplitude(phase)
                reconstruction = amplitude / amplitude.amax((-2, -1), keepdim=True)
                (reconstruction - target).square().mean().backward()
                self.assertTrue(torch.isfinite(target.grad).all())
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(model.transport_target_bias.grad.abs().sum()), 0)
                self.assertGreater(float(model.transport_target_encoder.output.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.transport_phase_encoder.output.weight.grad.abs().sum()), 0)

    def test_inference_is_input_dependent_and_does_not_update_state(self):
        torch.manual_seed(43)
        model = TransportPhaseGenerator(width=4).eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        inputs = torch.rand(2, 4, 128, 128)
        with torch.no_grad():
            first, diagnostics = model(inputs, return_diagnostics=True)
            second = model(inputs)
            changed = model(inputs.flip(1))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.allclose(first, changed))
        self.assertIn("radius_min", diagnostics)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_generator_validation(self):
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=0)
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(phase_head="linear")
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(position="chirp")
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(phase_bias_init="lens")
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(phase_decoder="analytic")
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=4)(torch.rand(1, 3, 128, 128))

    def test_random_bias_is_seeded_shared_and_preserves_other_initial_weights(self):
        torch.manual_seed(17)
        zero = TransportPhaseGenerator(width=4)
        torch.manual_seed(17)
        random = TransportPhaseGenerator(width=4, phase_bias_init="random")
        torch.manual_seed(17)
        repeated = TransportPhaseGenerator(width=4, phase_bias_init="random")
        torch.manual_seed(18)
        different = TransportPhaseGenerator(width=4, phase_bias_init="random")
        self.assertEqual(random.transport_target_bias.shape, (1, 4, 128, 128))
        self.assertTrue(random.transport_target_bias.requires_grad)
        self.assertEqual(int(torch.count_nonzero(zero.transport_target_bias)), 0)
        self.assertGreater(float(random.transport_target_bias.detach().std()), 1)
        self.assertLessEqual(float(random.transport_target_bias.detach().abs().max()), math.pi)
        self.assertTrue(torch.equal(random.transport_target_bias, repeated.transport_target_bias))
        self.assertFalse(torch.equal(random.transport_target_bias, different.transport_target_bias))
        for name, value in zero.state_dict().items():
            if name != "transport_target_bias":
                self.assertTrue(torch.equal(value, random.state_dict()[name]), name)

    def test_random_bias_has_finite_optical_gradients_in_both_gauges(self):
        physics = FresnelPropagator()
        for gauge in (True, False):
            torch.manual_seed(19)
            model = TransportPhaseGenerator(width=4, include_output_chirp=gauge, phase_bias_init="random")
            target = torch.rand(2, 4, 128, 128, requires_grad=True)
            amplitude = physics.amplitude(model(target))
            reconstruction = amplitude / amplitude.amax((-2, -1), keepdim=True)
            (reconstruction - target).square().mean().backward()
            self.assertTrue(torch.isfinite(target.grad).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(model.transport_target_bias.grad.abs().sum()), 0)

    def test_zero_linear_skip_preserves_seeded_state_and_initial_output(self):
        for bias_init in ("zero", "random"):
            torch.manual_seed(23)
            base = TransportPhaseGenerator(width=4, phase_bias_init=bias_init).eval()
            torch.manual_seed(23)
            skipped = TransportPhaseGenerator(width=4, phase_bias_init=bias_init, linear_skip=True).eval()
            base_state = base.state_dict()
            skip_state = skipped.state_dict()
            self.assertNotIn("transport_linear_skip.weight", base_state)
            self.assertEqual(set(skip_state) - set(base_state), {"transport_linear_skip.weight"})
            for name, value in base_state.items():
                self.assertTrue(torch.equal(value, skip_state[name]), name)
            self.assertEqual(skipped.transport_linear_skip.weight.numel(), 16)
            self.assertEqual(int(torch.count_nonzero(skipped.transport_linear_skip.weight)), 0)
            target = torch.rand(2, 4, 128, 128)
            with torch.no_grad():
                self.assertTrue(torch.equal(base(target), skipped(target)))

    def test_linear_skip_learns_through_optics_while_other_branches_keep_gradients(self):
        physics = FresnelPropagator()
        for gauge in (True, False):
            torch.manual_seed(24)
            model = TransportPhaseGenerator(width=4, phase_bias_init="random", include_output_chirp=gauge, linear_skip=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            target = torch.rand(2, 4, 128, 128)
            amplitude = physics.amplitude(model(target))
            normalized = amplitude / amplitude.amax((-2, -1), keepdim=True)
            (normalized - target).square().mean().backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(model.transport_linear_skip.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.transport_target_encoder.output.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.transport_phase_encoder.output.weight.grad.abs().sum()), 0)
            optimizer.step()
            self.assertGreater(float(model.transport_linear_skip.weight.detach().abs().sum()), 0)

    def test_explicit_unet_preserves_default_checkpoint_and_initial_values(self):
        torch.manual_seed(25)
        default = TransportPhaseGenerator(width=4, phase_bias_init="random")
        torch.manual_seed(25)
        explicit = TransportPhaseGenerator(width=4, phase_bias_init="random", phase_decoder="unet")
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for name, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        explicit.load_state_dict(default.state_dict(), strict=True)

    def test_subpixel_heads_have_finite_optical_gradients_without_interpolation(self):
        physics = FresnelPropagator()
        for head in ("tanh", "circular", "wrapped"):
            torch.manual_seed(26)
            model = TransportPhaseGenerator(width=4, phase_head=head, phase_bias_init="random", phase_decoder="subpixel")
            target = torch.rand(1, 4, 128, 128, requires_grad=True)
            with patch("torch.nn.functional.interpolate", side_effect=AssertionError("interpolation is not part of this decoder")):
                phase = model(target)
            self.assertEqual(phase.shape, (1, 1, 128, 128))
            self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
            amplitude = physics.amplitude(phase)
            normalized = amplitude / amplitude.amax((-2, -1), keepdim=True)
            (normalized - target).square().mean().backward()
            self.assertTrue(torch.isfinite(target.grad).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            per_output_gradient = model.transport_phase_encoder.output.weight.grad.flatten(1).abs().sum(1)
            self.assertTrue((per_output_gradient > 0).all())
            self.assertGreater(float(model.transport_target_encoder.output.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
