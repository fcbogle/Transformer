"""Check embedding shapes, position information, and trainable parameters."""

import unittest

import torch

from model import ModelConfig, TokenAndPositionEmbeddings


class EmbeddingTests(unittest.TestCase):
    def test_default_batch_shape_on_cpu(self) -> None:
        embeddings = TokenAndPositionEmbeddings(ModelConfig(vocab_size=65))
        token_ids = torch.zeros((32, 128), dtype=torch.long)
        output = embeddings(token_ids)
        self.assertEqual(output.shape, (32, 128, 128))
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(output.is_floating_point())
        self.assertEqual(embeddings.token_embedding.weight.shape, (65, 128))
        self.assertEqual(embeddings.position_embedding.weight.shape, (128, 128))

    def test_known_vectors_are_added_by_token_and_position(self) -> None:
        config = ModelConfig(vocab_size=3, block_size=3, n_embd=2, n_head=1)
        embeddings = TokenAndPositionEmbeddings(config)
        with torch.no_grad():
            embeddings.token_embedding.weight.copy_(
                torch.tensor([[1., 2.], [10., 20.], [100., 200.]])
            )
            embeddings.position_embedding.weight.copy_(
                torch.tensor([[0., 0.], [3., 4.], [5., 6.]])
            )
        # Repeated IDs select the same token row but receive different positions.
        output = embeddings(torch.tensor([[1, 1], [2, 0]]))
        expected = torch.tensor([[[10., 20.], [13., 24.]],
                                 [[100., 200.], [4., 6.]]])
        torch.testing.assert_close(output, expected)

    def test_gradients_reach_both_tables(self) -> None:
        config = ModelConfig(vocab_size=3, block_size=4, n_embd=2, n_head=1)
        embeddings = TokenAndPositionEmbeddings(config)
        embeddings(torch.tensor([[1, 1]])).sum().backward()
        # Token 1 occurs twice; each of positions 0 and 1 occurs once.
        torch.testing.assert_close(
            embeddings.token_embedding.weight.grad,
            torch.tensor([[0., 0.], [2., 2.], [0., 0.]]),
        )
        torch.testing.assert_close(
            embeddings.position_embedding.weight.grad,
            torch.tensor([[1., 1.], [1., 1.], [0., 0.], [0., 0.]]),
        )

    def test_single_token_sequence(self) -> None:
        embeddings = TokenAndPositionEmbeddings(ModelConfig(vocab_size=3))
        self.assertEqual(embeddings(torch.tensor([[2]])).shape, (1, 1, 128))

    def test_invalid_shapes_types_and_lengths_are_rejected(self) -> None:
        embeddings = TokenAndPositionEmbeddings(ModelConfig(vocab_size=3, block_size=4))
        for token_ids in (
            torch.tensor([1, 2]),
            torch.zeros((1, 2, 3), dtype=torch.long),
            torch.zeros((1, 2)),
            torch.zeros((1, 0), dtype=torch.long),
            torch.zeros((1, 5), dtype=torch.long),
        ):
            with self.subTest(shape=token_ids.shape, dtype=token_ids.dtype):
                with self.assertRaises(ValueError):
                    embeddings(token_ids)


if __name__ == "__main__":
    unittest.main()
