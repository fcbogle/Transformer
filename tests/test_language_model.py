"""Verify next-character predictions, causality, and learning on CPU."""

import unittest

import torch
from torch.nn import functional as F

from model import ModelConfig, ShakespeareTransformer


class ShakespeareTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(1337)
        self.model = ShakespeareTransformer(ModelConfig(
            vocab_size=6, block_size=8, n_embd=8, n_head=2,
            n_layer=2, dropout=0.0,
        ))

    def test_cpu_logits_with_and_without_targets(self) -> None:
        for length in (1, 5, 8):
            with self.subTest(length=length):
                tokens = torch.randint(6, (2, length))
                logits, loss = self.model(tokens)
                self.assertEqual(logits.shape, (2, length, 6))
                self.assertEqual(logits.device.type, 'cpu')
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertIsNone(loss)
                # Non-contiguous targets should also work when flattened.
                targets = torch.randint(6, (length, 2)).transpose(0, 1)
                training_logits, loss = self.model(tokens, targets)
                expected = torch.stack([
                    F.cross_entropy(training_logits[:, t], targets[:, t])
                    for t in range(length)
                ]).mean()
                self.assertEqual(loss.shape, torch.Size([]))
                torch.testing.assert_close(loss, expected)

    def test_future_tokens_do_not_affect_earlier_logits(self) -> None:
        self.model.eval()
        tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        changed = tokens.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 6
        with torch.no_grad():
            logits, _ = self.model(tokens)
            changed_logits, _ = self.model(changed)
            prefix_logits, _ = self.model(tokens[:, :3])
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])
        torch.testing.assert_close(logits[:, :3], prefix_logits)

    def test_loss_backpropagates_through_entire_model(self) -> None:
        tokens = torch.tensor([[0, 1, 2, 3], [1, 3, 2, 4]])
        targets = torch.tensor([[1, 2, 3, 4], [3, 2, 4, 5]])
        _, loss = self.model(tokens, targets)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_loss_decreases_on_repeated_sequence(self) -> None:
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]])
        targets = torch.tensor([[1, 2, 3, 4, 5, 0, 1, 2]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        _, initial_loss = self.model(tokens, targets)
        initial_value = initial_loss.item()
        for _ in range(30):
            optimizer.zero_grad()
            _, loss = self.model(tokens, targets)
            loss.backward()
            optimizer.step()
        self.model.eval()
        with torch.no_grad():
            _, final_loss = self.model(tokens, targets)
        self.assertLess(final_loss.item(), initial_value * 0.5)

    def test_invalid_inputs_are_rejected(self) -> None:
        tokens = torch.zeros(2, 4, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, 'same shape'):
            self.model(tokens, torch.zeros(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'torch.long'):
            self.model(tokens, torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, 'Sequence length'):
            self.model(torch.zeros(2, 9, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
