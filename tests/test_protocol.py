import unittest

import torch

from src.protocol import LEGACY_PROTOCOL, PEAK_PROTOCOL, normalize_peak, prepare_reconstruction, prepare_target


class ProtocolTests(unittest.TestCase):
    def test_per_image_scale_invariance_before_clipping(self):
        images = torch.tensor([[[[0., 2.], [4., 8.]], [[1., 2.], [3., 4.]]]])
        expected = images / images.amax((-2, -1), keepdim=True)
        self.assertTrue(torch.equal(prepare_reconstruction(images, PEAK_PROTOCOL), expected))
        self.assertTrue(torch.equal(normalize_peak(images * 16), expected))
        self.assertFalse(torch.equal(normalize_peak(images.clamp(0, 1)), expected))

    def test_zero_and_gradients(self):
        image = torch.zeros(2, 4, 8, 8)
        self.assertTrue(torch.equal(normalize_peak(image), image))
        image = torch.rand(2, 4, 8, 8, requires_grad=True)
        normalize_peak(image).square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_legacy_input_unchanged(self):
        image = torch.rand(2, 4, 8, 8) * 0.7
        self.assertIs(prepare_target(image, LEGACY_PROTOCOL), image)
        self.assertTrue(torch.equal(prepare_reconstruction(image * 2, LEGACY_PROTOCOL), (image * 2).clamp(0, 1)))
        self.assertTrue(torch.equal(prepare_target(image, PEAK_PROTOCOL).amax((-2, -1)), torch.ones(2, 4)))


if __name__ == "__main__":
    unittest.main()
