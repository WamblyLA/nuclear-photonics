import math
import unittest

import torch

from src.models.networks import build_model


class CrossAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_all_heads_have_valid_output_and_input_gradients(self):
        for phase_head in ("tanh", "circular", "wrapped"):
            with self.subTest(phase_head=phase_head):
                torch.manual_seed(11)
                model = build_model("cross_attention", width=4, phase_head=phase_head, position="xy")
                inputs = torch.rand(1, 4, 128, 128, requires_grad=True)
                phase = model(inputs)
                self.assertEqual(phase.shape, (1, 1, 128, 128))
                self.assertLessEqual(phase.abs().max().item(), math.pi)
                phase.sin().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue((inputs.grad.abs().sum((-2, -1)) > 0).all())
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_queries_are_input_independent_but_phase_is_conditional(self):
        model = build_model("cross_attention", width=4).eval()
        queries = []
        handle = model.cross_query.register_forward_hook(lambda module, inputs, output: queries.append(output.detach().clone()))
        with torch.no_grad():
            first = model(torch.rand(1, 4, 128, 128))
            second = model(torch.rand(1, 4, 128, 128))
        handle.remove()
        torch.testing.assert_close(queries[0], queries[1], rtol=0, atol=0)
        self.assertGreater((first - second).abs().max().item(), 1e-5)

    def test_subpixel_phase_layout(self):
        model = build_model("cross_attention", width=4)
        with torch.no_grad():
            model.cross_phase_output.weight.zero_()
            model.cross_phase_output.bias.copy_(torch.linspace(-1, 1, 16))
            phase = model(torch.rand(1, 4, 128, 128))
        expected = math.pi * torch.tanh(torch.linspace(-1, 1, 16).reshape(4, 4))
        torch.testing.assert_close(phase[0, 0, :4, :4], expected)
        torch.testing.assert_close(phase[0, 0, 100:104, 100:104], expected)

    def test_no_accidental_cnn_weight_loading(self):
        model = build_model("cross_attention", width=4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        result = model.load_state_dict(build_model("cnn", width=4).state_dict(), strict=False)
        self.assertTrue(result.unexpected_keys)
        self.assertEqual(set(result.missing_keys), set(before))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
