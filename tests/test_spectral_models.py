import math
import unittest

import torch

from src.models.spectral import SpectralPhaseGenerator, spectral_features


class SpectralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_spectral_features_preserve_information_and_mix_globally(self):
        image = torch.zeros(1, 4, 128, 128)
        image[:, 0, 37, 51] = 1
        encoded = spectral_features(image)
        real, imag = encoded.sinh().chunk(2, dim=1)
        spectrum = torch.complex(real, imag)
        reconstructed = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        torch.testing.assert_close(reconstructed.real, image, atol=1e-6, rtol=1e-5)
        self.assertGreater(float(spectrum[:, 0].abs().min()), 0)
        self.assertTrue(torch.equal(encoded[:, 1:4], torch.zeros_like(encoded[:, 1:4])))

    def test_phase_range_and_physics_gradient_for_each_head(self):
        for head in ("tanh", "circular", "wrapped"):
            with self.subTest(head=head):
                torch.manual_seed(5)
                model = SpectralPhaseGenerator(width=4, phase_head=head)
                images = torch.rand(2, 4, 128, 128, requires_grad=True)
                phase = model(images)
                self.assertEqual(phase.shape, (2, 1, 128, 128))
                self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
                # Differentiate a complex field, not the wrapped scalar itself.
                field = torch.polar(torch.ones_like(phase), phase)
                amplitude = torch.fft.fft2(field, norm="ortho").abs()
                loss = (amplitude[..., 3:19, 7:23] - 0.5).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(float(images.grad.abs().sum()), 0)
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

    def test_inference_depends_on_input_without_mutation(self):
        for mode in ("spectral", "hybrid"):
            with self.subTest(mode=mode):
                model = SpectralPhaseGenerator(width=4, feature_mode=mode).eval()
                state = {key: value.clone() for key, value in model.state_dict().items()}
                images = torch.rand(1, 4, 128, 128)
                with torch.no_grad():
                    first = model(images)
                    repeated = model(images)
                    changed = model(images.roll(13, dims=-1))
                torch.testing.assert_close(first, repeated, rtol=0, atol=0)
                self.assertGreater(float((first - changed).abs().mean()), 1e-5)
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))

    def test_invalid_shape_and_configuration(self):
        for arguments in ({"width": 0}, {"feature_mode": "inverse"}, {"position": "chirp"}, {"phase_head": "raw"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SpectralPhaseGenerator(**arguments)
        model = SpectralPhaseGenerator(width=4)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 4, 64, 64))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 4, 128, 128, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
