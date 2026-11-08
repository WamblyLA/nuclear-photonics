import math
import unittest

import torch

from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator


class TransportTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_default_unet_keeps_seeded_checkpoint_and_output_unchanged(self):
        torch.manual_seed(61)
        default = TransportPhaseGenerator(width=4, phase_decoder="complex")
        torch.manual_seed(61)
        explicit = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="unet")
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for name, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
            self.assertFalse("transport_target_encoder.base." in name)
            self.assertFalse("transport_target_encoder.detail." in name)
        inputs = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            self.assertTrue(torch.equal(default(inputs), explicit(inputs)))

    def test_loading_base_preserves_exact_forward_and_all_existing_state(self):
        for decoder in ("unet", "complex"):
            torch.manual_seed(62)
            base = TransportPhaseGenerator(width=4, phase_decoder=decoder, phase_bias_init="random").eval()
            residual = TransportPhaseGenerator(width=4, phase_decoder=decoder, target_encoder="residual").eval()
            loaded = residual.load_base_state_dict(base.state_dict())
            self.assertTrue(loaded.missing_keys)
            self.assertTrue(all(key.startswith("transport_target_encoder.detail.") for key in loaded.missing_keys))
            self.assertFalse(loaded.unexpected_keys)
            for key, value in base.state_dict().items():
                target_key = key.replace("transport_target_encoder.", "transport_target_encoder.base.", 1) if key.startswith("transport_target_encoder.") else key
                self.assertTrue(torch.equal(value, residual.state_dict()[target_key]), key)
            inputs = torch.rand(2, 4, 128, 128)
            with torch.no_grad():
                self.assertTrue(torch.equal(base(inputs), residual(inputs)))

    def test_detail_output_learns_first_step_and_interior_has_second_step_gradient(self):
        torch.manual_seed(63)
        base = TransportPhaseGenerator(width=4, phase_decoder="complex")
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="residual")
        model.load_base_state_dict(base.state_dict())
        detail = model.transport_target_encoder.detail
        interior = detail.features[0].layers[0].weight
        before_interior = interior.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        target = torch.rand(1, 4, 128, 128)
        physics = FresnelPropagator()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            amplitude = physics.amplitude(model(target))
            reconstructed = amplitude / amplitude.amax((-2, -1), keepdim=True)
            (reconstructed - target).square().mean().backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            if step == 0:
                self.assertEqual(int(torch.count_nonzero(interior.grad)), 0)
                self.assertGreater(float(detail.output.weight.grad.abs().sum()), 0)
                self.assertGreater(float(detail.output.bias.grad.abs().sum()), 0)
            else:
                self.assertGreater(float(interior.grad.abs().sum()), 0)
            optimizer.step()
            if step == 0:
                self.assertGreater(float(detail.output.weight.detach().abs().sum()), 0)
                self.assertTrue(torch.equal(interior, before_interior))
        self.assertFalse(torch.equal(interior, before_interior))

    def test_scratch_subpixel_has_input_dependent_finite_optical_gradients(self):
        torch.manual_seed(64)
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="subpixel")
        inputs = torch.rand(2, 4, 128, 128, requires_grad=True)
        phase = model(inputs)
        self.assertEqual(phase.shape, (2, 1, 128, 128))
        self.assertLessEqual(float(phase.detach().abs().max()), math.pi + 1e-6)
        self.assertFalse(torch.allclose(phase[0], phase[1]))
        amplitude = FresnelPropagator().amplitude(phase)
        reconstructed = amplitude / amplitude.amax((-2, -1), keepdim=True)
        (reconstructed - inputs).square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        output_gradient = model.transport_target_encoder.output.weight.grad.flatten(1).abs().sum(1)
        self.assertTrue((output_gradient > 0).all())

    def test_incompatible_base_states_are_rejected_before_mutation(self):
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="residual")
        correct = TransportPhaseGenerator(width=4, phase_decoder="complex").state_dict()
        missing = dict(correct)
        missing.pop("transport_target_bias")
        unexpected = dict(correct)
        unexpected["unexpected.weight"] = torch.zeros(1)
        wrong_shape = dict(correct)
        wrong_shape["transport_target_bias"] = torch.zeros(1, 4, 16, 16)
        states = [
            missing, unexpected, wrong_shape,
            TransportPhaseGenerator(width=4, phase_decoder="unet").state_dict(),
            TransportPhaseGenerator(width=8, phase_decoder="complex").state_dict(),
            TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="subpixel").state_dict(),
            model.state_dict(),
        ]
        original = {key: value.clone() for key, value in model.state_dict().items()}
        for state in states:
            with self.assertRaises(ValueError):
                model.load_base_state_dict(state)
            for key, value in original.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=4).load_base_state_dict(correct)
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(target_encoder="analytic")


if __name__ == "__main__":
    unittest.main()
