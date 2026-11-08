import unittest

import torch

from src.losses import depth_ranking_loss


class DepthRankingLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(271)
        self.target = torch.rand(2, 4, 13, 15, generator=generator, dtype=torch.float64)

    def test_matching_targets_satisfy_margin(self):
        for margin in (0.0, 0.2, 1.0):
            self.assertAlmostEqual(float(depth_ranking_loss(self.target, self.target, margin)), 0.0, places=12)

    def test_permuting_planes_increases_loss(self):
        wrong = self.target[:, [1, 2, 3, 0]]
        self.assertGreater(float(depth_ranking_loss(wrong, self.target)), 0.1)

    def test_identical_targets_do_not_demand_separation(self):
        identical = self.target[:, :1].expand_as(self.target)
        self.assertAlmostEqual(float(depth_ranking_loss(self.target, identical)), 0.0, places=12)

    def test_finite_gradient_and_detached_target(self):
        target = self.target.clone().requires_grad_()
        prediction = self.target[:, [1, 2, 3, 0]].clone().requires_grad_()
        depth_ranking_loss(prediction, target).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        self.assertIsNone(target.grad)

    def test_constant_reconstruction_can_learn(self):
        prediction = torch.zeros_like(self.target, requires_grad=True)
        depth_ranking_loss(prediction, self.target).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_constant_targets_are_excluded(self):
        prediction = self.target.clone().requires_grad_()
        loss = depth_ranking_loss(prediction, torch.zeros_like(self.target))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_margin_and_shape_validation(self):
        for margin in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                depth_ranking_loss(self.target, self.target, margin)
        with self.assertRaises(ValueError):
            depth_ranking_loss(self.target[:, :1], self.target[:, :1])
        with self.assertRaises(ValueError):
            depth_ranking_loss(self.target, self.target[:, :, :5])


if __name__ == "__main__":
    unittest.main()
