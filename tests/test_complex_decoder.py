import math
import unittest

import torch

from src.models.complex_decoder import ComplexPhaseEncoder, ComplexPointwise, expand_complex_mixing_state
from src.models.networks import phase_from_logits
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator


def rotate_features(features, angle, complex_channels):
    real = features[:, :complex_channels]
    imag = features[:, complex_channels:2 * complex_channels]
    coordinates = features[:, 2 * complex_channels:]
    return torch.cat((real * math.cos(angle) - imag * math.sin(angle),
                      real * math.sin(angle) + imag * math.cos(angle), coordinates), dim=1)


class ComplexDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_complex_pointwise_matches_complex_matrix_product(self):
        torch.manual_seed(51)
        layer = ComplexPointwise(4, 3).double()
        features = torch.randn(2, 8, 7, 9, dtype=torch.float64)
        field = torch.complex(features[:, :4], features[:, 4:])
        weights = torch.complex(layer.real.weight[:, :, 0, 0], layer.imag.weight[:, :, 0, 0])
        expected = torch.einsum("oi,bihw->bohw", weights, field)
        actual = layer(features)
        self.assertTrue(torch.allclose(actual, torch.cat((expected.real, expected.imag), dim=1), atol=1e-12, rtol=1e-12))

    def test_global_phase_equivariance_with_and_without_spatial_context(self):
        for channels in (8, 10):
            for context in (False, True):
                for kernel in (1, 3):
                    torch.manual_seed(52)
                    model = ComplexPhaseEncoder(channels, 2, width=5, spatial_context=context, mix_kernel=kernel).double().eval()
                    features = torch.randn(2, channels, 17, 19, dtype=torch.float64)
                    logits = model(features)
                    for angle in (0.71, -2.32, math.pi):
                        rotated = model(rotate_features(features, angle, 4))
                        expected = rotate_features(logits, angle, 1)
                        self.assertTrue(torch.allclose(rotated, expected, atol=2e-12, rtol=2e-12))

    def test_circular_head_and_physical_amplitude_respect_global_phase(self):
        torch.manual_seed(53)
        model = ComplexPhaseEncoder(10, 2, width=5).eval()
        features = torch.randn(2, 10, 32, 32)
        phase = phase_from_logits(model(features), "circular")
        rotated = phase_from_logits(model(rotate_features(features, 0.73, 4)), "circular")
        self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
        physics = FresnelPropagator(size=32)
        self.assertTrue(torch.allclose(physics.amplitude(phase), physics.amplitude(rotated), atol=2e-6, rtol=2e-5))

    def test_finite_end_to_end_optical_gradients_and_all_weights_update(self):
        for context in (False, True):
            torch.manual_seed(54)
            model = TransportPhaseGenerator(width=4)
            model.transport_phase_encoder = ComplexPhaseEncoder(10, 2, width=4, spatial_context=context)
            target = torch.rand(1, 4, 128, 128, requires_grad=True)
            phase = model(target)
            self.assertEqual(phase.shape, (1, 1, 128, 128))
            amplitude = FresnelPropagator().amplitude(phase)
            reconstructed = amplitude / amplitude.amax((-2, -1), keepdim=True)
            loss = (reconstructed - target).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(target.grad).all())
            before = {key: value.detach().clone() for key, value in model.transport_phase_encoder.named_parameters()}
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            torch.optim.SGD(model.parameters(), lr=0.01).step()
            for key, value in model.transport_phase_encoder.named_parameters():
                self.assertFalse(torch.equal(before[key], value), key)

    def test_input_dependence_zero_stability_and_unchanged_inference_state(self):
        torch.manual_seed(55)
        model = ComplexPhaseEncoder(10, 2, width=5).eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        features = torch.randn(2, 10, 16, 16)
        with torch.no_grad():
            first = model(features)
            second = model(features)
            changed = model(features.flip(-1))
            zero = model(torch.zeros_like(features))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.allclose(first, changed))
        self.assertTrue(torch.isfinite(zero).all())
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_argument_validation(self):
        for arguments in ({"in_channels": 9}, {"out_channels": 1}, {"width": 0}, {"depth": 0},
                          {"mix_kernel": 2}, {"mix_kernel": 3.0}, {"mix_kernel": True}):
            with self.assertRaises(ValueError):
                ComplexPhaseEncoder(**arguments)
        for kernel in (0, 2, 5, 3.0, True):
            with self.assertRaises(ValueError):
                ComplexPointwise(4, 3, kernel_size=kernel)
        with self.assertRaises(ValueError):
            ComplexPhaseEncoder()(torch.rand(1, 8, 8, 8))

    def test_default_kernel_keeps_seeded_state_and_forward(self):
        torch.manual_seed(56)
        default = ComplexPhaseEncoder(width=5)
        after_default = torch.get_rng_state().clone()
        torch.manual_seed(56)
        explicit = ComplexPhaseEncoder(width=5, mix_kernel=1)
        self.assertTrue(torch.equal(after_default, torch.get_rng_state()))
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
        features = torch.randn(2, 10, 15, 17)
        self.assertTrue(torch.equal(default(features), explicit(features)))

    def test_spatial_mixing_matches_complex_convolution(self):
        torch.manual_seed(57)
        layer = ComplexPointwise(4, 3, kernel_size=3).double()
        features = torch.randn(2, 8, 9, 11, dtype=torch.float64)
        field = torch.complex(features[:, :4], features[:, 4:])
        weights = torch.complex(layer.real.weight, layer.imag.weight)
        expected = torch.nn.functional.conv2d(field, weights, padding=1)
        actual = layer(features)
        self.assertEqual(actual.shape, (2, 6, 9, 11))
        torch.testing.assert_close(actual, torch.cat((expected.real, expected.imag), dim=1), atol=2e-12, rtol=2e-12)

    def test_expanded_state_preserves_forward_exactly_and_has_no_side_effects(self):
        for channels, context, dtype in ((8, False, torch.float32), (10, True, torch.float32), (10, True, torch.float64)):
            torch.manual_seed(58)
            source = ComplexPhaseEncoder(channels, width=4, depth=2, spatial_context=context).to(dtype)
            target = ComplexPhaseEncoder(channels, width=4, depth=2, spatial_context=context, mix_kernel=3).to(dtype)
            state = {key: value.clone() for key, value in source.state_dict().items()}
            target_before = {key: value.clone() for key, value in target.state_dict().items()}
            rng_before = torch.get_rng_state().clone()
            expanded = expand_complex_mixing_state(state, target.state_dict())
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, source.state_dict()[key]))
                self.assertTrue(torch.equal(target_before[key], target.state_dict()[key]))
                self.assertNotEqual(expanded[key].data_ptr(), value.data_ptr())
                if key.startswith("mixes."):
                    self.assertTrue(torch.equal(expanded[key][:, :, 1:2, 1:2], value))
                    neighbors = expanded[key].clone()
                    neighbors[:, :, 1, 1] = 0
                    self.assertEqual(int(torch.count_nonzero(neighbors)), 0)
                else:
                    self.assertTrue(torch.equal(expanded[key], value))
            target.load_state_dict(expanded, strict=True)
            features = torch.randn(2, channels, 17, 19, dtype=dtype)
            self.assertTrue(torch.equal(source(features), target(features)), (channels, context, dtype))
            self.assertEqual(target.input.real.weight.shape[-2:], (1, 1))
            self.assertEqual(target.output.real.weight.shape[-2:], (1, 1))

    def test_zero_initialized_neighbors_receive_optical_gradients_and_updates(self):
        torch.manual_seed(59)
        model = TransportPhaseGenerator(width=4)
        original_decoder = ComplexPhaseEncoder(width=4)
        model.transport_phase_encoder = ComplexPhaseEncoder(width=4, mix_kernel=3)
        expanded = expand_complex_mixing_state(original_decoder.state_dict(), model.transport_phase_encoder.state_dict())
        model.transport_phase_encoder.load_state_dict(expanded, strict=True)
        target = torch.rand(1, 4, 128, 128)
        phase = model(target)
        amplitude = FresnelPropagator().amplitude(phase)
        reconstructed = amplitude / amplitude.amax((-2, -1), keepdim=True)
        (reconstructed - target).square().mean().backward()
        mask = torch.ones(3, 3, dtype=torch.bool)
        mask[1, 1] = False
        neighbors = [parameter for name, parameter in model.transport_phase_encoder.named_parameters() if name.startswith("mixes.")]
        for parameter in neighbors:
            self.assertEqual(int(torch.count_nonzero(parameter[:, :, mask])), 0)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue((parameter.grad[:, :, mask] != 0).all())
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        for parameter in neighbors:
            self.assertTrue((parameter[:, :, mask] != 0).all())

    def test_expansion_rejects_incompatible_or_incomplete_states(self):
        source = ComplexPhaseEncoder(width=4).state_dict()
        target = ComplexPhaseEncoder(width=4, mix_kernel=3).state_dict()
        cases = []
        missing = dict(source)
        missing.pop("mixes.0.real.weight")
        cases.append((missing, target))
        extra_source, extra_target = dict(source), dict(target)
        extra_source["unexpected"] = extra_target["unexpected"] = torch.zeros(1)
        cases.append((extra_source, extra_target))
        wrong_dtype = dict(source)
        wrong_dtype["input.real.weight"] = wrong_dtype["input.real.weight"].double()
        cases.append((wrong_dtype, target))
        nonfinite = {key: value.clone() for key, value in source.items()}
        nonfinite["mixes.0.real.weight"][0, 0, 0, 0] = float("nan")
        cases.append((nonfinite, target))
        cases.extend(((target, target), (source, source), (None, target),
                      (source, ComplexPhaseEncoder(width=5, mix_kernel=3).state_dict()),
                      (source, ComplexPhaseEncoder(width=4, mix_kernel=3, spatial_context=False).state_dict()),
                      (source, ComplexPhaseEncoder(in_channels=8, width=4, mix_kernel=3).state_dict())))
        for number, (old, new) in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError):
                expand_complex_mixing_state(old, new)


if __name__ == "__main__":
    unittest.main()
