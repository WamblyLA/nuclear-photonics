import math
import unittest

import torch

from src.models.networks import build_model
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator, phase_to_uint8


class TargetConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **options):
        return TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="consensus", **options)

    def test_seeded_construction_preserves_base_and_envelope_rng(self):
        models, states = [], []
        for name in ("transport_complex", "transport_envelope_complex", "transport_consensus_complex"):
            torch.manual_seed(501)
            models.append(build_model(name, width=4, phase_head="circular", position="xy"))
            states.append(torch.get_rng_state().clone())
        base, envelope, consensus = models
        self.assertTrue(all(torch.equal(states[0], state) for state in states))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, consensus.state_dict()[key]), key)
        self.assertEqual(set(consensus.state_dict()) - set(envelope.state_dict()), {"transport_consensus_gain"})
        self.assertEqual(consensus.transport_consensus_gain.shape, torch.Size([]))
        self.assertEqual(float(consensus.transport_consensus_gain.detach()), 0)
        self.assertEqual(sum(parameter.numel() for parameter in consensus.parameters()) - sum(parameter.numel() for parameter in envelope.parameters()), 1)
        self.assertFalse(consensus.energy_balanced)
        target = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(target), consensus(target)))
            self.assertTrue(torch.equal(envelope(target), consensus(target)))

    def test_zero_transfer_preserves_phase_latents_png_and_rng(self):
        for position in ("none", "xy"):
            with self.subTest(position=position):
                torch.manual_seed(502)
                base = TransportPhaseGenerator(width=4, phase_decoder="complex", position=position)
                with torch.no_grad():
                    base.transport_target_bias.normal_(std=0.03)
                model = self.model(position=position)
                with torch.no_grad():
                    model.transport_target_encoder.envelope.weight.fill_(0.2)
                    model.transport_target_encoder.envelope.bias.fill_(-0.3)
                    model.transport_consensus_gain.fill_(0.8)
                state = {key: value.clone() for key, value in base.state_dict().items()}
                rng = torch.get_rng_state().clone()
                loaded = model.load_consensus_state_dict(state)
                self.assertFalse(loaded.missing_keys or loaded.unexpected_keys)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                    self.assertTrue(torch.equal(value, base.state_dict()[key]), key)
                    self.assertNotEqual(value.data_ptr(), model.state_dict()[key].data_ptr())
                for parameter in (model.transport_consensus_gain, model.transport_target_encoder.envelope.weight, model.transport_target_encoder.envelope.bias):
                    self.assertEqual(int(torch.count_nonzero(parameter)), 0)
                target = torch.rand(2, 4, 128, 128)
                with torch.no_grad():
                    old_phase, old_latent = base(target, return_latent=True)
                    new_phase, new_latent = model(target, return_latent=True)
                self.assertTrue(torch.equal(old_latent, new_latent))
                self.assertTrue(torch.equal(old_phase, new_phase))
                self.assertTrue(torch.equal(phase_to_uint8(old_phase), phase_to_uint8(new_phase)))

    def test_zero_gain_receives_nonzero_optical_gradient(self):
        torch.manual_seed(503)
        model = self.model()
        target = torch.rand(2, 4, 128, 128)
        phase = model(target)
        amplitude = FresnelPropagator().amplitude(phase)
        prediction = amplitude / amplitude.amax((-2, -1), keepdim=True)
        (prediction - target).square().mean().backward()
        for parameter in (model.transport_consensus_gain, model.transport_target_encoder.envelope.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 1e-10)
        torch.optim.SGD([model.transport_consensus_gain], lr=0.1).step()
        self.assertNotEqual(float(model.transport_consensus_gain.detach()), 0)
        with torch.no_grad():
            self.assertFalse(torch.equal(phase, model(target)))

    def test_free_gain_changes_only_internal_envelope_by_specified_formula(self):
        torch.manual_seed(504)
        model = self.model()
        target = torch.rand(1, 4, 128, 128)
        original = target.clone()
        for gain in (-0.5, 1.0, 1.5):
            with self.subTest(gain=gain), torch.no_grad():
                model.transport_consensus_gain.fill_(gain)
                logits, delta = model.transport_target_encoder(model.add_coordinates(target))
                corrected = target + delta + gain * (target.mean(dim=1, keepdim=True) - target)
                expected = corrected * torch.exp(1j * (math.pi * logits + model.transport_target_bias))
                phase, latent = model(target, return_latent=True)
                torch.testing.assert_close(model.transport_operator(latent), expected, atol=2e-6, rtol=2e-6)
                self.assertTrue(torch.equal(target, original))
                self.assertTrue(torch.isfinite(phase).all())
                self.assertLessEqual(float(phase.abs().max()), math.pi + 1e-6)

    def test_bad_source_states_are_rejected_atomically(self):
        source = TransportPhaseGenerator(width=4, phase_decoder="complex").state_dict()
        model = self.model()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        missing, extra = dict(source), dict(source)
        missing.pop("transport_target_bias")
        extra["unexpected"] = torch.zeros(1)
        shape, dtype, nonfinite, not_tensor = (dict(source) for _ in range(4))
        shape["transport_target_bias"] = torch.zeros(1, 4, 32, 32)
        dtype["transport_target_bias"] = source["transport_target_bias"].double()
        nonfinite["transport_target_bias"] = torch.full_like(source["transport_target_bias"], float("inf"))
        not_tensor["transport_target_bias"] = 0
        envelope = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope").state_dict()
        wider = TransportPhaseGenerator(width=5, phase_decoder="complex").state_dict()
        states = (None, {}, missing, extra, shape, dtype, nonfinite, not_tensor, envelope, model.state_dict(), wider)
        for index, state in enumerate(states):
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.load_consensus_state_dict(state)
            for key, value in initial.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_incompatible_destination_is_rejected(self):
        source = TransportPhaseGenerator(width=4, phase_decoder="complex").state_dict()
        for options in ({"energy_balanced": True}, {"complex_mix_kernel": 3}, {"linear_skip": True}, {"include_output_chirp": False}, {"phase_bias_init": "random"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options).load_consensus_state_dict(source)
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope").load_consensus_state_dict(source)


if __name__ == "__main__":
    unittest.main()
