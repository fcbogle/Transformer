"""Check a complete block's causal, residual, and pre-normalisation behaviour."""

import unittest

import torch

from model import ModelConfig, TokenAndPositionEmbeddings, TransformerBlock


class TransformerBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(1337)

    def test_cpu_shapes_and_stacking(self) -> None:
        for width in (8, 128):
            config = ModelConfig(vocab_size=5, n_embd=width)
            blocks = torch.nn.Sequential(TransformerBlock(config), TransformerBlock(config)).eval()
            for length in (1, 5):
                with self.subTest(width=width, length=length):
                    output = blocks(torch.randn(2, length, width))
                    self.assertEqual(output.shape, (2, length, width))
                    self.assertEqual(output.device.type, 'cpu')
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_future_tokens_do_not_affect_earlier_outputs(self) -> None:
        config = ModelConfig(vocab_size=5, n_embd=8)
        embeddings = TokenAndPositionEmbeddings(config)
        blocks = torch.nn.Sequential(TransformerBlock(config), TransformerBlock(config)).eval()
        tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        changed = tokens.clone()
        changed[:, 3:] = 2
        output = blocks(embeddings(tokens))
        torch.testing.assert_close(output[:, :3], blocks(embeddings(changed))[:, :3])
        torch.testing.assert_close(output[:, :3], blocks(embeddings(tokens[:, :3])))

    def test_zero_branches_preserve_input_and_identity_gradient(self) -> None:
        block = TransformerBlock(ModelConfig(vocab_size=5, n_embd=8))
        with torch.no_grad():
            for branch in (block.self_attention, block.feed_forward):
                for parameter in branch.parameters():
                    parameter.zero_()
        x = torch.randn(2, 5, 8, requires_grad=True)
        original = x.detach().clone()
        output = block(x)
        torch.testing.assert_close(output, original)
        torch.testing.assert_close(x, original)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_each_branch_receives_normalised_residual_stream(self) -> None:
        block = TransformerBlock(ModelConfig(vocab_size=5, n_embd=8, dropout=0.0))
        captured = {}

        def capture_attention(module, inputs, output):
            captured['attention_input'] = inputs[0].detach()
            captured['attention_output'] = output.detach()

        def capture_feed_forward(module, inputs, output):
            captured['feed_forward_input'] = inputs[0].detach()

        for module, hook in ((block.self_attention, capture_attention),
                             (block.feed_forward, capture_feed_forward)):
            handle = module.register_forward_hook(hook)
            self.addCleanup(handle.remove)
        x = torch.randn(2, 5, 8) * 3 + 7
        block(x)
        # Default LayerNorm affine weights are 1 and biases 0.
        def normalise(tensor):
            variance, mean = torch.var_mean(tensor, dim=-1, correction=0, keepdim=True)
            return (tensor - mean) / torch.sqrt(variance + 1e-5)

        torch.testing.assert_close(captured['attention_input'], normalise(x))
        torch.testing.assert_close(
            captured['feed_forward_input'], normalise(x + captured['attention_output'])
        )
        self.assertIsNot(block.layer_norm_1.weight, block.layer_norm_2.weight)

    def test_gradients_reach_embeddings_norms_and_both_branches(self) -> None:
        config = ModelConfig(vocab_size=5, n_embd=8, dropout=0.0)
        embeddings = TokenAndPositionEmbeddings(config)
        block = TransformerBlock(config)
        tokens = torch.tensor([[0, 1, 2, 3], [1, 3, 2, 4]])
        block(embeddings(tokens)).square().mean().backward()
        for module in (embeddings, block):
            for name, parameter in module.named_parameters():
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_invalid_input_shape_is_rejected(self) -> None:
        block = TransformerBlock(ModelConfig(vocab_size=5, n_embd=8))
        for shape in ((2, 8), (2, 3, 7)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'shape'):
                block(torch.randn(*shape))


if __name__ == '__main__':
    unittest.main()
