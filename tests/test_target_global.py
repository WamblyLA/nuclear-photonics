import unittest

import torch

from src.models.networks import build_model
from src.models.target_global import TargetSpectralMixer
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator, phase_to_uint8


class GlobalTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **options):
        return TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="global", **options)

    def test_constructor_preserves_shared_weights_rng_and_zero_forward(self):
        torch.manual_seed(601)
        base = build_model("transport_complex", 4, "circular", "xy")
        rng = torch.get_rng_state().clone()
        torch.manual_seed(601)
        model = build_model("transport_globaltarget_complex", 4, "circular", "xy")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertTrue(all(key.startswith("transport_target_encoder.global_") for key in set(model.state_dict()) - set(base.state_dict())))
        target = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            old, new = base(target), model(target)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(phase_to_uint8(old), phase_to_uint8(new)))

    def test_strict_transfer_preserves_rng_mixer_phase_latents_and_png(self):
        torch.manual_seed(602)
        base = TransportPhaseGenerator(width=4, phase_decoder="complex")
        with torch.no_grad():
            base.transport_target_bias.normal_(std=0.03)
        model = self.model()
        with torch.no_grad():
            model.transport_target_encoder.global_gain.fill_(0.7)
        mixer = {key: value.clone() for key, value in model.transport_target_encoder.global_mixer.state_dict().items()}
        rng = torch.get_rng_state().clone()
        source = {key: value.clone() for key, value in base.state_dict().items()}
        loaded = model.load_globaltarget_state_dict(source)
        self.assertFalse(loaded.missing_keys or loaded.unexpected_keys)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(model.transport_target_encoder.global_gain.detach()), 0)
        for key, value in mixer.items():
            self.assertTrue(torch.equal(value, model.transport_target_encoder.global_mixer.state_dict()[key]))
        for key, value in source.items():
            self.assertTrue(torch.equal(value, base.state_dict()[key]))
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
            self.assertNotEqual(value.data_ptr(), model.state_dict()[key].data_ptr())
        target = torch.rand(2, 4, 128, 128)
        with torch.no_grad():
            old, old_latent = base(target, return_latent=True)
            new, new_latent = model(target, return_latent=True)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(old_latent, new_latent))
        self.assertTrue(torch.equal(phase_to_uint8(old), phase_to_uint8(new)))

    def test_gate_learns_first_then_spectral_parameters_get_optical_gradients(self):
        torch.manual_seed(603)
        model = self.model()
        target = torch.rand(2, 4, 128, 128)
        branch = model.transport_target_encoder.global_mixer
        gain = model.transport_target_encoder.global_gain
        physics = FresnelPropagator()
        optimizer = torch.optim.SGD([gain] + list(branch.parameters()), lr=0.05)
        for step in range(2):
            optimizer.zero_grad()
            amplitude = physics.amplitude(model(target))
            prediction = amplitude / amplitude.amax((-2, -1), keepdim=True)
            (prediction - target).square().mean().backward()
            self.assertTrue(torch.isfinite(gain.grad).all())
            self.assertGreater(float(gain.grad.abs()), 1e-10)
            for parameter in branch.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                if step == 0:
                    self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
            if step == 1:
                for layer in (branch.frequency[0], branch.frequency[3]):
                    self.assertGreater(float(layer.weight.grad.abs().sum()), 1e-10)
            optimizer.step()
        self.assertNotEqual(float(gain.detach()), 0)

    def test_spectral_branch_is_nonlinear_and_spatially_coupled(self):
        torch.manual_seed(604)
        mixer = TargetSpectralMixer(4)
        first, second = torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
        with torch.no_grad():
            self.assertGreater(float((mixer(first + second) - mixer(first) - mixer(second)).abs().mean()), 0.01)
            changed = first.clone()
            changed[:, 0, 2, 3] += 5
            distant = (mixer(changed) - mixer(first))[:, :, 20:, 20:]
            self.assertGreater(float(distant.abs().max()), 1e-5)

    def test_invalid_transfer_is_atomic_and_rejects_architecture_changes(self):
        source = TransportPhaseGenerator(width=4, phase_decoder="complex").state_dict()
        model = self.model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        missing, extra, shape, dtype, nonfinite = (dict(source) for _ in range(5))
        missing.pop("transport_target_bias")
        extra["unexpected"] = torch.zeros(1)
        shape["transport_target_bias"] = torch.zeros(1, 4, 32, 32)
        dtype["transport_target_bias"] = source["transport_target_bias"].double()
        nonfinite["transport_target_bias"] = torch.full_like(source["transport_target_bias"], float("nan"))
        envelope = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope").state_dict()
        for index, state in enumerate((None, {}, missing, extra, shape, dtype, nonfinite, envelope, model.state_dict())):
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.load_globaltarget_state_dict(state)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        for options in ({"energy_balanced": True}, {"complex_mix_kernel": 3}, {"linear_skip": True}, {"include_output_chirp": False}, {"phase_bias_init": "random"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options).load_globaltarget_state_dict(source)


if __name__ == "__main__":
    unittest.main()
