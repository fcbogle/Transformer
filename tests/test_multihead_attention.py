"""Check head combination, causal behaviour, and trainable parameters."""

import unittest

import torch

from model import ModelConfig, MultiHeadAttention, TokenAndPositionEmbeddings


class MultiHeadAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(1337)

    def test_cpu_shapes_for_different_head_counts_and_sequence_lengths(self) -> None:
        for heads in (1, 2, 4):
            attention = MultiHeadAttention(
                ModelConfig(vocab_size=5, n_embd=8, n_head=heads)
            ).eval()
            for length in (1, 5):
                with self.subTest(heads=heads, length=length):
                    output = attention(torch.randn(2, length, 8))
                    self.assertEqual(output.shape, (2, length, 8))
                    self.assertEqual(output.device.type, "cpu")
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_known_head_outputs_are_concatenated_then_projected(self) -> None:
        attention = MultiHeadAttention(
            ModelConfig(vocab_size=3, n_embd=2, n_head=2, dropout=0.0)
        )
        with torch.no_grad():
            attention.heads[0].value.weight.copy_(torch.tensor([[1., 0.]]))
            attention.heads[1].value.weight.copy_(torch.tensor([[0., 2.]]))
            attention.projection.weight.copy_(torch.tensor([[1., 1.], [2., -1.]]))
            attention.projection.bias.copy_(torch.tensor([1., -2.]))
        # One token: heads return 3 and 8, concatenation [3, 8], projection [12, -4].
        output = attention(torch.tensor([[[3., 4.]]]))
        torch.testing.assert_close(output, torch.tensor([[[12., -4.]]]))

    def test_future_tokens_do_not_change_earlier_outputs(self) -> None:
        attention = MultiHeadAttention(
            ModelConfig(vocab_size=5, n_embd=8, n_head=2)
        ).eval()
        x = torch.randn(2, 5, 8)
        changed = x.clone()
        changed[:, 3:] = torch.randn(2, 2, 8) * 10
        torch.testing.assert_close(attention(x)[:, :3], attention(changed)[:, :3])
        torch.testing.assert_close(attention(x)[:, :3], attention(x[:, :3]))

    def test_gradients_reach_embeddings_all_heads_and_output_projection(self) -> None:
        config = ModelConfig(vocab_size=5, n_embd=8, n_head=2, dropout=0.0)
        embeddings = TokenAndPositionEmbeddings(config)
        attention = MultiHeadAttention(config)
        tokens = torch.tensor([[0, 1, 2, 3], [1, 3, 2, 4]])
        attention(embeddings(tokens)).square().mean().backward()
        for module in (embeddings, attention):
            for name, parameter in module.named_parameters():
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Heads must have separate learned projections, registered in state_dict.
        for name in ("query", "key", "value"):
            self.assertIsNot(
                getattr(attention.heads[0], name).weight,
                getattr(attention.heads[1], name).weight,
            )
            self.assertIn(f"heads.1.{name}.weight", attention.state_dict())

    def test_output_dropout_and_eval_mode(self) -> None:
        attention = MultiHeadAttention(
            ModelConfig(vocab_size=5, n_embd=8, n_head=2, dropout=0.5)
        )
        # Constant projection isolates output dropout from attention-weight dropout.
        with torch.no_grad():
            attention.projection.weight.zero_()
            attention.projection.bias.fill_(1)
        x = torch.randn(2, 5, 8)
        output = attention(x)
        self.assertTrue(((output == 0) | (output == 2)).all().item())
        self.assertTrue((output == 0).any().item())
        self.assertTrue((output == 2).any().item())
        attention.eval()
        torch.testing.assert_close(attention(x), torch.ones_like(x))
        self.assertTrue(all(not head.training for head in attention.heads))


if __name__ == "__main__":
    unittest.main()
