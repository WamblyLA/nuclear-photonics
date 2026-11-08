import math
import unittest

import torch

from src.models.full_spectral import FullSpectralBlock, FullSpectralGenerator


class FullSpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_all_phase_heads_have_valid_forward_and_backward(self):
        for head in ("tanh", "circular", "wrapped"):
            torch.manual_seed(42)
            model = FullSpectralGenerator(width=4, phase_head=head)
            inputs = torch.rand(1, 4, 128, 128, requires_grad=True)
            phase = model(inputs)
            self.assertEqual(phase.shape, (1, 1, 128, 128))
            self.assertTrue(torch.isfinite(phase).all())
            self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
            field = torch.exp(1j * phase)
            loss = (field.real * torch.randn_like(phase)).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
            self.assertGreater(float(inputs.grad.abs().sum()), 0)
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_full_frequency_branch_retains_nyquist_information(self):
        torch.manual_seed(13)
        block = FullSpectralBlock(2, 4)
        checker = 1 - 2 * ((torch.arange(32)[:, None] + torch.arange(32)[None]) % 2)
        inputs = checker.float()[None, None].expand(1, 2, -1, -1).contiguous().requires_grad_()
        features = block.spectral_features(inputs)
        spectrum = torch.fft.rfft2(features, norm="ortho")
        self.assertGreater(float(spectrum[..., 16, 16].detach().abs().sum()), 1e-4)
        spectrum[..., 16, 16].abs().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0)

    def test_inference_depends_on_inputs_without_mutating_state(self):
        torch.manual_seed(19)
        model = FullSpectralGenerator(width=4, phase_head="wrapped").eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        inputs = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            first = model(inputs)
            repeated = model(inputs)
            changed = model(inputs.flip(1))
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.allclose(first, changed))
        for key, value in state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FullSpectralGenerator(width=0)
        with self.assertRaises(ValueError):
            FullSpectralGenerator(phase_head="linear")
        with self.assertRaises(ValueError):
            FullSpectralGenerator(position="chirp")
        with self.assertRaises(ValueError):
            FullSpectralGenerator(width=4)(torch.rand(1, 4, 64, 64))


if __name__ == "__main__":
    unittest.main()
