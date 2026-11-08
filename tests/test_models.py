import math
import unittest

import torch

from src.models.networks import build_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_phase_range_and_gradients(self):
        for name in ("cnn", "fourier", "ae", "local", "interleave"):
            with self.subTest(model=name):
                torch.manual_seed(7)
                model = build_model(name, width=4)
                images = torch.rand(2, 4, 128, 128, requires_grad=True)
                phase = model(images)
                self.assertEqual(phase.shape, (2, 1, 128, 128))
                self.assertTrue(torch.isfinite(phase).all())
                self.assertLessEqual(phase.abs().max().item(), math.pi)
                phase.square().mean().backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(images.grad.abs().sum().item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                if name == "fourier":
                    for parameter in (model.global_mixing.weight_top, model.global_mixing.weight_bottom):
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                if name == "local":
                    self.assertGreater(model.global_mixing.local.weight.grad.abs().sum().item(), 0)

    def test_local_control_parameter_count(self):
        local = build_model("local", width=16)
        fourier = build_model("fourier", width=16)
        local_count = sum(parameter.numel() for parameter in local.parameters())
        fourier_count = sum(parameter.numel() for parameter in fourier.parameters())
        self.assertEqual(local_count - fourier_count, 64)

    def test_ae_changes_only_skip_connections(self):
        cnn = build_model("cnn", width=4).eval()
        ae = build_model("ae", width=4).eval()
        ae.load_state_dict(cnn.state_dict())
        self.assertEqual(sum(p.numel() for p in cnn.parameters()), sum(p.numel() for p in ae.parameters()))
        images = torch.rand(1, 4, 32, 32)
        with torch.no_grad():
            self.assertFalse(torch.allclose(cnn(images), ae(images)))

    def test_inference_does_not_update_model_or_need_gradients(self):
        for model_name in ("fourier", "interleave"):
            model = build_model(model_name, width=4).eval()
            state_before = {name: value.clone() for name, value in model.state_dict().items()}
            images = torch.rand(1, 4, 32, 32)
            with torch.inference_mode():
                first = model(images)
                second = model(images)
            torch.testing.assert_close(first, second)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state_before[name])

    def test_circular_head_and_origin_guard(self):
        model = build_model("cnn", width=4, phase_head="circular")
        phase, diagnostics = model(torch.rand(1, 4, 32, 32), return_diagnostics=True)
        self.assertEqual(phase.shape, (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(phase.abs().max().item(), math.pi)
        self.assertIn("radius_min", diagnostics)
        phase.square().mean().backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

        logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
        phase, diagnostics = model.to_phase(logits, return_diagnostics=True)
        phase.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(diagnostics["near_origin_fraction"].item(), 1)
        self.assertEqual(phase.abs().sum().item(), 0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            build_model("unknown")
        with self.assertRaises(ValueError):
            build_model("cnn", width=0)
        with self.assertRaises(ValueError):
            build_model("cnn", phase_head="unknown")
        with self.assertRaises(ValueError):
            build_model("cnn")(torch.rand(1, 3, 32, 32))
        with self.assertRaises(ValueError):
            build_model("cnn", position="unknown")
        with self.assertRaises(ValueError):
            build_model("cnn", width=4, position="xy")(torch.rand(1, 4, 32, 32))
        with self.assertRaises(ValueError):
            build_model("interleave", position="xy")
        with self.assertRaises(ValueError):
            build_model("interleave", width=4)(torch.rand(1, 4, 31, 32))

    def test_interleave_learns_separate_subpixel_phases(self):
        model = build_model("interleave", width=4)
        head = model.subpixel_output[0]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.linspace(-1, 1, 16))
            phase = model(torch.rand(1, 4, 128, 128))
        expected = math.pi * torch.tanh(torch.linspace(-1, 1, 16).reshape(4, 4))
        torch.testing.assert_close(phase[0, 0, :4, :4], expected)
        torch.testing.assert_close(phase[0, 0, 4:8, 4:8], expected)
        circular = build_model("interleave", width=4, phase_head="circular")
        phase, diagnostics = circular(torch.rand(1, 4, 128, 128), return_diagnostics=True)
        self.assertEqual(phase.shape, (1, 1, 128, 128))
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(phase.abs().max().item(), math.pi)
        self.assertIn("near_origin_fraction", diagnostics)
        phase.square().mean().backward()
        for parameter in circular.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_position_branch_starts_as_exact_baseline(self):
        images = torch.rand(1, 4, 128, 128)
        for name in ("cnn", "fourier", "ae", "local"):
            torch.manual_seed(42)
            baseline = build_model(name, width=4).eval()
            for position in ("xy", "chirp"):
                with self.subTest(model=name, position=position):
                    torch.manual_seed(42)
                    model = build_model(name, width=4, position=position).eval()
                    for key, value in baseline.state_dict().items():
                        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(baseline(images), model(images), rtol=0, atol=0)
                    model(images).square().mean().backward()
                    gradient = model.position_projection.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)

    def test_position_feature_geometry(self):
        xy = build_model("cnn", width=4, position="xy").position_features
        chirp = build_model("cnn", width=4, position="chirp").position_features
        self.assertEqual(xy.shape, (1, 2, 128, 128))
        self.assertEqual(chirp.shape, (1, 10, 128, 128))
        torch.testing.assert_close(chirp[:, :2], xy)
        self.assertEqual(xy[0, 0, 64, 64].item(), 0)
        self.assertEqual(xy[0, 1, 0, 64].item(), -1)
        torch.testing.assert_close(chirp[:, 2:6].square() + chirp[:, 6:10].square(), torch.ones(1, 4, 128, 128))
        expected = math.pi * (10e-6) ** 2 / (532e-9 * 0.0241)
        self.assertAlmostEqual(chirp[0, 2, 64, 65].item(), math.sin(expected), places=6)


if __name__ == "__main__":
    unittest.main()
