"""Verify position-wise feature processing and trainable feed-forward layers."""

import math
import unittest

import torch

from model import FeedForward, ModelConfig


class FeedForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(1337)

    def test_cpu_shapes_and_expansion(self) -> None:
        for width in (8, 128):
            layer = FeedForward(ModelConfig(vocab_size=5, n_embd=width))
            self.assertEqual(layer.expand.out_features, 4 * width)
            for length in (1, 5):
                with self.subTest(width=width, length=length):
                    output = layer(torch.randn(2, length, width))
                    self.assertEqual(output.shape, (2, length, width))
                    self.assertEqual(output.device.type, "cpu")
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_known_values_include_gelu_and_both_linear_layers(self) -> None:
        layer = FeedForward(
            ModelConfig(vocab_size=3, n_embd=1, n_head=1, dropout=0.0)
        )
        with torch.no_grad():
            layer.expand.weight.copy_(torch.tensor([[1.], [-1.], [0.], [2.]]))
            layer.expand.bias.copy_(torch.tensor([0., 0., 0., -1.]))
            layer.project.weight.copy_(torch.tensor([[1., 2., 3., 4.]]))
            layer.project.bias.fill_(0.5)
        # Input 1 expands to [1, -1, 0, 1]. GELU(x) = x * Phi(x).
        phi_one = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
        expected = 5 * phi_one - 2 * (1 - phi_one) + 0.5
        torch.testing.assert_close(
            layer(torch.ones(1, 1, 1)), torch.tensor([[[expected]]])
        )

    def test_positions_are_independent_and_share_the_same_transform(self) -> None:
        layer = FeedForward(ModelConfig(vocab_size=5, n_embd=8)).eval()
        x = torch.randn(2, 5, 8)
        changed = x.clone()
        changed[:, 2] += 10
        output = layer(x)
        changed_output = layer(changed)
        torch.testing.assert_close(output[:, [0, 1, 3, 4]], changed_output[:, [0, 1, 3, 4]])
        # Processing a token alone must give the same result as in a sequence.
        torch.testing.assert_close(output[:, 2:3], layer(x[:, 2:3]))
        permutation = [4, 2, 0, 3, 1]
        torch.testing.assert_close(layer(x[:, permutation]), output[:, permutation])

    def test_gradients_reach_input_and_both_linear_layers(self) -> None:
        layer = FeedForward(ModelConfig(vocab_size=5, n_embd=8, dropout=0.0))
        x = torch.randn(2, 5, 8, requires_grad=True)
        layer(x).square().mean().backward()
        for tensor in (x, *layer.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_dropout_respects_training_and_evaluation_modes(self) -> None:
        layer = FeedForward(ModelConfig(vocab_size=5, n_embd=8, dropout=0.5))
        with torch.no_grad():
            layer.project.weight.zero_()
            layer.project.bias.fill_(1)
        x = torch.randn(2, 5, 8)
        output = layer(x)
        self.assertTrue(((output == 0) | (output == 2)).all().item())
        self.assertTrue((output == 0).any().item())
        self.assertTrue((output == 2).any().item())
        layer.eval()
        torch.testing.assert_close(layer(x), torch.ones_like(x))

    def test_invalid_input_shape_is_rejected(self) -> None:
        layer = FeedForward(ModelConfig(vocab_size=5, n_embd=8))
        for shape in ((2, 8), (2, 3, 7)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                layer(torch.randn(*shape))


if __name__ == "__main__":
    unittest.main()
