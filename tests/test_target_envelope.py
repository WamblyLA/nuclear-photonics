import math
import unittest

import torch

from src.models.target_envelope import TargetEnvelopeEncoder
from src.models.transport import TransportEncoder, TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator, phase_to_uint8


class TargetEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_shared_features_and_zero_head_preserve_seeded_phase_encoder(self):
        torch.manual_seed(71)
        base = TransportEncoder(6, 4, 4)
        base_rng = torch.get_rng_state().clone()
        torch.manual_seed(71)
        envelope = TargetEnvelopeEncoder(6, 4, 4)
        self.assertTrue(torch.equal(base_rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, envelope.state_dict()[key]))
        features = torch.randn(2, 6, 32, 32)
        phase, delta = envelope(features)
        self.assertTrue(torch.equal(base(features), phase))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        self.assertEqual(sum(parameter.numel() for parameter in envelope.parameters()) - sum(parameter.numel() for parameter in base.parameters()), 4 * (4 + 1))
        with self.assertRaises(ValueError):
            TargetEnvelopeEncoder(6, 1, 4)

    def test_new_head_does_not_shift_other_model_initialization(self):
        torch.manual_seed(72)
        base = TransportPhaseGenerator(width=4, phase_decoder="complex", phase_bias_init="random")
        rng = torch.get_rng_state().clone()
        torch.manual_seed(72)
        envelope = TransportPhaseGenerator(width=4, phase_decoder="complex", phase_bias_init="random", target_encoder="envelope")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, envelope.state_dict()[key]), key)
        target = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(target), envelope(target)))

    def test_strict_transfer_preserves_phase_and_latents_exactly(self):
        for position in ("none", "xy"):
            torch.manual_seed(73)
            base = TransportPhaseGenerator(width=4, phase_decoder="complex", position=position)
            with torch.no_grad():
                base.transport_target_bias.normal_(std=0.03)
            torch.manual_seed(74)
            envelope = TransportPhaseGenerator(width=4, phase_decoder="complex", position=position, target_encoder="envelope")
            with torch.no_grad():
                envelope.transport_target_encoder.envelope.weight.fill_(0.1)
                envelope.transport_target_encoder.envelope.bias.fill_(-0.2)
            state = {key: value.clone() for key, value in base.state_dict().items()}
            rng = torch.get_rng_state().clone()
            loaded = envelope.load_envelope_state_dict(state)
            self.assertFalse(loaded.missing_keys)
            self.assertFalse(loaded.unexpected_keys)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, base.state_dict()[key]))
                self.assertTrue(torch.equal(value, envelope.state_dict()[key]))
                self.assertNotEqual(value.data_ptr(), envelope.state_dict()[key].data_ptr())
            self.assertEqual(int(torch.count_nonzero(envelope.transport_target_encoder.envelope.weight)), 0)
            self.assertEqual(int(torch.count_nonzero(envelope.transport_target_encoder.envelope.bias)), 0)
            target = torch.rand(2, 4, 128, 128)
            with torch.no_grad():
                old_phase, old_latent = base(target, return_latent=True)
                new_phase, new_latent = envelope(target, return_latent=True)
                repeated = envelope(target)
            self.assertTrue(torch.equal(old_phase, new_phase))
            self.assertTrue(torch.equal(old_latent, new_latent))
            self.assertTrue(torch.equal(phase_to_uint8(old_phase), phase_to_uint8(new_phase)))
            self.assertTrue(torch.equal(new_phase, repeated))

    def test_zero_envelope_gets_optical_gradients_and_changes_output(self):
        torch.manual_seed(75)
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope")
        target = torch.rand(2, 4, 128, 128)
        phase = model(target)
        amplitude = FresnelPropagator().amplitude(phase)
        prediction = amplitude / amplitude.amax((-2, -1), keepdim=True)
        (prediction - target).square().mean().backward()
        head = model.transport_target_encoder.envelope
        for parameter in (head.weight, head.bias, model.transport_target_encoder.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        before = phase.detach().clone()
        torch.optim.SGD(head.parameters(), lr=0.1).step()
        with torch.no_grad():
            after = model(target)
        self.assertGreater(int(torch.count_nonzero(head.weight)), 0)
        self.assertGreater(int(torch.count_nonzero(head.bias)), 0)
        self.assertTrue(torch.isfinite(after).all())
        self.assertFalse(torch.equal(before, after))

    def test_signed_envelope_remains_a_complex_latent_feature(self):
        torch.manual_seed(76)
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope")
        with torch.no_grad():
            model.transport_target_encoder.envelope.bias.fill_(-2)
            target = torch.rand(1, 4, 128, 128)
            phase_logits, delta = model.transport_target_encoder(model.add_coordinates(target))
            self.assertTrue((target + delta < 0).all())
            phase, latent = model(target, return_latent=True)
            expected = (target + delta) * torch.exp(1j * (math.pi * phase_logits + model.transport_target_bias))
            recovered = model.transport_operator(latent)
        torch.testing.assert_close(recovered, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(float(phase.abs().max()), math.pi + 1e-6)

    def test_migration_rejects_bad_states_before_any_mutation(self):
        source = TransportPhaseGenerator(width=4, phase_decoder="complex").state_dict()
        model = TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="envelope")
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        missing = dict(source)
        missing.pop("transport_target_bias")
        extra = dict(source)
        extra["unexpected"] = torch.zeros(1)
        wrong_shape = dict(source)
        wrong_shape["transport_target_bias"] = torch.zeros(1, 4, 32, 32)
        wrong_dtype = dict(source)
        wrong_dtype["transport_target_bias"] = wrong_dtype["transport_target_bias"].double()
        nonfinite = dict(source)
        nonfinite["transport_target_bias"] = torch.full_like(source["transport_target_bias"], float("nan"))
        states = (None, {}, missing, extra, wrong_shape, wrong_dtype, nonfinite, model.state_dict(),
                  TransportPhaseGenerator(width=5, phase_decoder="complex").state_dict(),
                  TransportPhaseGenerator(width=4, phase_decoder="complex", target_encoder="residual").state_dict())
        for index, state in enumerate(states):
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.load_envelope_state_dict(state)
            for key, value in initial.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=4, phase_decoder="complex").load_envelope_state_dict(source)
        with self.assertRaises(ValueError):
            TransportPhaseGenerator(width=4, target_encoder="envelope").load_envelope_state_dict(source)


if __name__ == "__main__":
    unittest.main()
