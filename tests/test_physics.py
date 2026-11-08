import math
import unittest

import torch

from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase, quantize_phase


class FresnelTests(unittest.TestCase):
    def test_direct_fresnel_sum(self):
        # No FFT in this reference: directly integrate the sampled kernel.
        for size in (7, 8):
            model = FresnelPropagator(size=size, distances=(0.0241, 0.0277)).double()
            generator = torch.Generator().manual_seed(17)
            phi = torch.randn(1, size, size, generator=generator, dtype=torch.float64)
            indices = torch.arange(size, dtype=torch.float64) - size // 2
            yy, xx = torch.meshgrid(indices * model.pitch, indices * model.pitch, indexing="ij")
            values = []
            for distance, output_pitch in zip(model.distances, model.output_pitch):
                y, x = torch.meshgrid(indices * output_pitch, indices * output_pitch, indexing="ij")
                difference = (x[..., None, None] - xx).square() + (y[..., None, None] - yy).square()
                kernel = torch.exp(1j * math.pi * difference / (model.wavelength * distance))
                prefactor = torch.exp(1j * 2 * math.pi * distance / model.wavelength) / (1j * model.wavelength * distance)
                values.append(prefactor * model.pitch**2 * (kernel * torch.exp(1j * phi[0])).sum(dim=(-2, -1)))
            expected = torch.stack(values)[None]
            torch.testing.assert_close(model(phi), expected, rtol=1e-9, atol=1e-10)

    def test_physical_energy_and_output_sampling(self):
        model = FresnelPropagator().double()
        phi = torch.randn(2, 1, 128, 128, dtype=torch.float64)
        amplitude = model.amplitude(phi)
        expected_pitches = torch.tensor([10.0165625, 10.5153125, 11.0140625, 11.5128125], dtype=torch.float64) * 1e-6
        torch.testing.assert_close(model.output_pitch, expected_pitches)
        energy = amplitude.square().sum(dim=(-2, -1)) * model.output_pitch.square()
        torch.testing.assert_close(energy, torch.full_like(energy, (128 * model.pitch) ** 2), rtol=1e-12, atol=1e-18)
        torch.testing.assert_close(amplitude.square().mean(dim=(-2, -1)).sqrt(), model.field_rms.expand(2, -1))

    def test_global_phase_and_input_shapes(self):
        model = FresnelPropagator(size=16).double()
        phi = torch.randn(2, 16, 16, dtype=torch.float64)
        torch.testing.assert_close(model.amplitude(phi), model.amplitude(phi + 1.37), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(model(phi), model(phi[:, None]))
        self.assertEqual(model(phi.float()).dtype, torch.complex64)
        self.assertEqual(model(phi).dtype, torch.complex128)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 4, 16, 16))

    def test_gradient_against_finite_difference(self):
        model = FresnelPropagator(size=8).double()
        generator = torch.Generator().manual_seed(21)
        phi = torch.randn(1, 8, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        weights = torch.rand(1, 4, 8, 8, generator=generator, dtype=torch.float64)
        loss = (model.amplitude(phi) * weights).mean()
        gradient, = torch.autograd.grad(loss, phi)
        self.assertTrue(torch.isfinite(gradient).all())
        for row, col in ((0, 0), (3, 4), (7, 7)):
            delta = torch.zeros_like(phi)
            delta[0, row, col] = 1e-6
            plus = (model.amplitude(phi.detach() + delta) * weights).mean()
            minus = (model.amplitude(phi.detach() - delta) * weights).mean()
            reference = (plus - minus) / 2e-6
            torch.testing.assert_close(gradient[0, row, col], reference, rtol=1e-5, atol=1e-10)

    def test_phase_png_codes_and_straight_through(self):
        codes = torch.arange(256, dtype=torch.uint8)
        phi = uint8_to_phase(codes, dtype=torch.float64)
        self.assertAlmostEqual(phi[0].item(), -math.pi)
        self.assertAlmostEqual(phi[-1].item(), math.pi)
        torch.testing.assert_close(phase_to_uint8(phi), codes)
        continuous = torch.linspace(-math.pi, math.pi, 1001, dtype=torch.float64, requires_grad=True)
        quantized = quantize_phase(continuous, straight_through=True)
        torch.testing.assert_close(quantized.detach(), uint8_to_phase(phase_to_uint8(continuous), dtype=torch.float64))
        self.assertLessEqual((quantized - continuous).abs().max().item(), math.pi / 255 + 1e-12)
        quantized.sum().backward()
        torch.testing.assert_close(continuous.grad, torch.ones_like(continuous))


if __name__ == "__main__":
    unittest.main()
