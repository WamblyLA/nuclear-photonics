import unittest

import torch

from src.train_plane_curriculum import Config, active_loss_terms, masked_plane_inputs, select_active_planes, validation_planes


class PlaneCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_each_example_has_only_its_chosen_channel(self):
        images = torch.arange(4 * 4 * 16 * 16, dtype=torch.float32).reshape(4, 4, 16, 16) + 1
        planes = torch.tensor([3, 1, 0, 2])
        masked = masked_plane_inputs(images, planes)
        selected = select_active_planes(images, planes)
        for index, plane in enumerate(planes):
            self.assertTrue(torch.equal(masked[index, plane], images[index, plane]))
            self.assertTrue(torch.equal(selected[index, 0], images[index, plane]))
            inactive = torch.arange(4) != plane
            self.assertEqual(int(torch.count_nonzero(masked[index, inactive])), 0)

    def test_inactive_amplitudes_and_targets_never_enter_loss_or_gradient(self):
        torch.manual_seed(101)
        raw = torch.rand(4, 4, 24, 24, requires_grad=True)
        targets = torch.rand_like(raw)
        planes = torch.tensor([0, 1, 2, 3])
        terms = active_loss_terms(raw, targets, planes, Config())
        loss = terms["objective"].mean()
        loss.backward()
        altered_raw = raw.detach().clone()
        altered_targets = targets.clone()
        for index, plane in enumerate(planes):
            inactive = torch.arange(4) != plane
            self.assertEqual(int(torch.count_nonzero(raw.grad[index, inactive])), 0)
            self.assertGreater(float(raw.grad[index, plane].abs().sum()), 0)
            altered_raw[index, inactive] = 17
            altered_targets[index, inactive] = 0
        altered = active_loss_terms(altered_raw, altered_targets, planes, Config())
        for key in terms:
            self.assertTrue(torch.equal(terms[key], altered[key]), key)

    def test_validation_is_balanced_and_independent_of_batch_boundaries(self):
        all_planes = validation_planes(64)
        self.assertEqual(torch.bincount(all_planes).tolist(), [16, 16, 16, 16])
        batched = torch.cat([validation_planes(min(7, 64 - start), start) for start in range(0, 64, 7)])
        self.assertTrue(torch.equal(all_planes, batched))

    def test_invalid_active_assignments_are_rejected(self):
        images = torch.rand(2, 4, 12, 12)
        for planes in (torch.tensor([0, 4]), torch.tensor([-1, 2]), torch.tensor([0]), torch.tensor([0.0, 1.0])):
            with self.assertRaises(ValueError):
                masked_plane_inputs(images, planes)


if __name__ == "__main__":
    unittest.main()
