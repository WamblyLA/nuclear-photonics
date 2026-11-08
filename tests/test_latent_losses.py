import unittest

import torch

from src.latent_losses import balanced_source_features, latent_source_consistency
from src.models.transport import TransportPhaseGenerator, UnitaryFresnelTransport


class LatentConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_balanced_features_have_unit_rms_after_unitary_transport(self):
        torch.manual_seed(81)
        target = torch.rand(2, 4, 16, 16, dtype=torch.float64)
        phase = torch.randn_like(target)
        for gauge in (True, False):
            operator = UnitaryFresnelTransport(size=16, include_output_chirp=gauge)
            source = operator.adjoint(target * torch.exp(1j * phase))
            balanced = balanced_source_features(source, target)
            rms = balanced.abs().square().mean((-2, -1)).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-13, atol=1e-13)

    def test_balanced_features_ignore_independent_plane_brightness(self):
        torch.manual_seed(82)
        target = torch.rand(2, 4, 16, 16, dtype=torch.float64)
        source = torch.randn(2, 4, 16, 16, dtype=torch.complex128)
        scales = torch.tensor([[0.01, 0.2, 2.0, 10.0], [2.5, 0.05, 7.0, 0.4]], dtype=torch.float64)[..., None, None]
        expected = balanced_source_features(source, target)
        scaled = balanced_source_features(source * scales, target * scales)
        torch.testing.assert_close(scaled, expected, rtol=1e-13, atol=1e-13)

    def test_balanced_features_have_finite_gradients_and_safe_zero_planes(self):
        torch.manual_seed(83)
        target = torch.rand(2, 4, 12, 12, dtype=torch.float64)
        target[:, 1] = 0
        target.requires_grad_()
        source = torch.randn(2, 4, 12, 12, dtype=torch.complex128, requires_grad=True)
        balanced = balanced_source_features(source, target)
        self.assertEqual(int(torch.count_nonzero(balanced[:, 1])), 0)
        weights = torch.rand_like(target)
        (balanced.real * weights + balanced.imag * weights.square()).mean().backward()
        for tensor in (source, target):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(int(torch.count_nonzero(tensor.grad[:, 1])), 0)
            self.assertGreater(float(tensor.grad[:, 0].abs().sum()), 0)
        empty_source = torch.randn(1, 4, 12, 12, dtype=torch.complex128, requires_grad=True)
        empty_target = torch.zeros(1, 4, 12, 12, dtype=torch.float64, requires_grad=True)
        empty = balanced_source_features(empty_source, empty_target)
        self.assertEqual(int(torch.count_nonzero(empty)), 0)
        empty.real.sum().backward()
        for tensor in (empty_source, empty_target):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(int(torch.count_nonzero(tensor.grad)), 0)

    def test_zero_for_consistent_random_source_and_arbitrary_target_scales(self):
        torch.manual_seed(71)
        phase = torch.randn(2, 1, 16, 16, dtype=torch.float64)
        source = torch.exp(1j * phase).expand(-1, 4, -1, -1)
        scales = torch.tensor([0.03, 0.2, 0.7, 2.0], dtype=torch.float64)[None, :, None, None]
        for gauge in (True, False):
            operator = UnitaryFresnelTransport(size=16, include_output_chirp=gauge)
            target_fields = operator(source) * scales
            latent = operator.adjoint(target_fields)
            loss = latent_source_consistency(latent, phase, target_fields.abs())
            self.assertLess(float(loss), 1e-27)
            self.assertEqual(loss, latent_source_consistency(latent, phase[:, 0], target_fields.abs()))

    def test_global_phase_covariance_and_target_brightness_invariance(self):
        torch.manual_seed(72)
        source = torch.complex(torch.randn(2, 4, 12, 12, dtype=torch.float64), torch.randn(2, 4, 12, 12, dtype=torch.float64))
        phase = torch.randn(2, 1, 12, 12, dtype=torch.float64)
        target = torch.rand(2, 4, 12, 12, dtype=torch.float64)
        expected = latent_source_consistency(source, phase, target)
        rotation = torch.exp(torch.tensor(0.83j, dtype=torch.complex128))
        rotated = latent_source_consistency(source * rotation, phase + 0.83, target)
        scales = torch.tensor([0.1, 0.4, 1.7, 3.0], dtype=torch.float64)[None, :, None, None]
        scaled = latent_source_consistency(source * scales, phase, target * scales)
        torch.testing.assert_close(rotated, expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(scaled, expected, rtol=1e-13, atol=1e-13)

    def test_finite_nonzero_gradients_to_fields_and_phase(self):
        torch.manual_seed(73)
        source = torch.randn(2, 4, 12, 12, dtype=torch.complex128, requires_grad=True)
        phase = torch.randn(2, 1, 12, 12, dtype=torch.float64, requires_grad=True)
        target = torch.rand(2, 4, 12, 12, dtype=torch.float64)
        loss = latent_source_consistency(source, phase, target)
        loss.backward()
        for tensor in (source, phase):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_zero_targets_are_excluded_and_all_zero_batch_is_safe(self):
        torch.manual_seed(74)
        source = torch.randn(2, 4, 8, 8, dtype=torch.complex128, requires_grad=True)
        phase = torch.randn(2, 1, 8, 8, dtype=torch.float64, requires_grad=True)
        target = torch.rand(2, 4, 8, 8, dtype=torch.float64)
        target[:, 1:] = 0
        loss = latent_source_consistency(source, phase, target)
        reference = latent_source_consistency(source[:, :1], phase, target[:, :1])
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(source.grad[:, 1:])), 0)
        source.grad = None
        phase.grad = None
        empty_target = torch.zeros_like(target, requires_grad=True)
        empty = latent_source_consistency(source, phase, empty_target)
        self.assertEqual(float(empty.detach()), 0)
        empty.backward()
        for tensor in (source, phase, empty_target):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(int(torch.count_nonzero(tensor.grad)), 0)

    def test_latent_forward_contract_and_end_to_end_neural_gradients(self):
        torch.manual_seed(75)
        inputs = torch.rand(1, 4, 128, 128)
        for target_encoder in ("unet", "subpixel", "residual", "envelope"):
            model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder=target_encoder)
            phase, source = model(inputs, return_latent=True)
            self.assertTrue(torch.equal(phase, model(inputs)))
            self.assertEqual(source.shape, inputs.shape)
            self.assertTrue(source.is_complex())
            self.assertTrue(source.requires_grad)
            latent_source_consistency(source, phase, inputs).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(model.transport_target_bias.grad.abs().sum()), 0)
            self.assertGreater(float(model.transport_phase_encoder.output.real.weight.grad.abs().sum()), 0)
            target_module = model.transport_target_encoder.base if target_encoder == "residual" else model.transport_target_encoder
            self.assertGreater(float(target_module.output.weight.grad.abs().sum()), 0)
            with self.assertRaises(ValueError):
                model(inputs, return_diagnostics=True, return_latent=True)
            diagnostic_phase, diagnostics = model(inputs, return_diagnostics=True)
            self.assertTrue(torch.equal(phase, diagnostic_phase))
            self.assertIn("radius_min", diagnostics)

    def test_corrected_features_retain_energy_and_phase_penalties(self):
        torch.manual_seed(76)
        target = torch.rand(2, 4, 8, 8, dtype=torch.float64) + 0.1
        phase = torch.randn(2, 1, 8, 8, dtype=torch.float64)
        reference_rms = target.square().mean((-2, -1), keepdim=True).sqrt()
        relative_rms = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)[None, :, None, None]
        phase_offset = torch.tensor([0.0, 0.2, -0.5, 1.2], dtype=torch.float64)[None, :, None, None]
        corrected = reference_rms * relative_rms * torch.exp(1j * (phase + phase_offset))
        expected = ((relative_rms - 1).square() + 2 * relative_rms * (1 - phase_offset.cos())).mean()
        torch.testing.assert_close(latent_source_consistency(corrected, phase, target), expected, rtol=1e-13, atol=1e-13)
        compatible = reference_rms * torch.exp(1j * phase)
        self.assertLess(float(latent_source_consistency(compatible, phase, target)), 1e-28)

    def test_shape_and_dtype_validation(self):
        source = torch.ones(1, 4, 8, 8, dtype=torch.complex64)
        phase = torch.zeros(1, 1, 8, 8)
        target = torch.ones(1, 4, 8, 8)
        for fields, mask, amplitude in (
            (source.real, phase, target),
            (source, phase.expand(-1, 4, -1, -1), target),
            (source, phase, target[:, :1]),
            (source, phase.to(torch.int64), target),
        ):
            with self.assertRaises(ValueError):
                latent_source_consistency(fields, mask, amplitude)


if __name__ == "__main__":
    unittest.main()
