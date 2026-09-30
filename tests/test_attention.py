"""Inspect attention stages and verify the complete causal head."""

import unittest

import torch

from model import AttentionHead, ModelConfig, TokenAndPositionEmbeddings


class AttentionHeadTests(unittest.TestCase):
    def test_combine_values_known_weighted_sum(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3, n_embd=4, n_head=2))
        weights = torch.tensor([[[1., 0.], [0.25, 0.75]]])
        value = torch.tensor([[[2., 4.], [10., 8.]]])
        torch.testing.assert_close(
            head.combine_values(weights, value), torch.tensor([[[2., 4.], [8., 7.]]])
        )

    def test_forward_is_causal_and_gradients_reach_embeddings(self) -> None:
        config = ModelConfig(vocab_size=5, block_size=4, n_embd=8, n_head=2)
        embeddings = TokenAndPositionEmbeddings(config)
        head = AttentionHead(config).eval()
        tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        output = head(embeddings(tokens))
        self.assertEqual(output.shape, (2, 4, 4))
        self.assertEqual(output.device.type, "cpu")
        changed = tokens.clone()
        changed[:, 2:] = 0
        torch.testing.assert_close(output[:, :2], head(embeddings(changed))[:, :2])
        # With only one visible token, its output must equal its own Value.
        torch.testing.assert_close(output[:, :1], head.value(embeddings(tokens))[:, :1])
        output.square().mean().backward()
        for parameter in list(head.parameters()) + list(embeddings.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_dropout_respects_training_mode_and_preserves_causal_zeros(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3, dropout=0.5))
        weights = head.attention_weights(head.apply_causal_mask(torch.zeros(2, 8, 8)))
        original = weights.clone()
        head.train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1337)
            dropped = head.apply_dropout(weights)
        self.assertEqual(dropped.shape, weights.shape)
        self.assertTrue(((dropped == 0) | (dropped == weights * 2)).all().item())
        self.assertTrue(((weights > 0) & (dropped == 0)).any().item())
        self.assertTrue((dropped > 0).any().item())
        self.assertEqual(torch.count_nonzero(dropped.triu(diagonal=1)).item(), 0)
        torch.testing.assert_close(weights, original)

        head.eval()
        torch.testing.assert_close(head.apply_dropout(weights), weights)

    def test_zero_dropout_preserves_weights_during_training(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3, dropout=0.0))
        weights = torch.tensor([[[1., 0.], [0.25, 0.75]]])
        head.train()
        torch.testing.assert_close(head.apply_dropout(weights), weights)

    def test_causal_mask_preserves_visible_scores_and_blocks_future(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3))
        scores = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        original = scores.clone()
        masked = head.apply_causal_mask(scores)
        expected = torch.tensor([
            [[0., -torch.inf, -torch.inf], [3., 4., -torch.inf], [6., 7., 8.]],
            [[9., -torch.inf, -torch.inf], [12., 13., -torch.inf], [15., 16., 17.]],
        ])
        torch.testing.assert_close(masked, expected)
        torch.testing.assert_close(scores, original)

        weights = head.attention_weights(masked)
        self.assertEqual(weights[0, 0].tolist(), [1., 0., 0.])
        self.assertEqual(torch.count_nonzero(weights.triu(diagonal=1)).item(), 0)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))

    def test_causal_mask_allows_single_token_to_attend_to_itself(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3))
        scores = torch.tensor([[[2.]]])
        torch.testing.assert_close(head.apply_causal_mask(scores), scores)

    def test_known_query_key_value_projections(self) -> None:
        head = AttentionHead(ModelConfig(vocab_size=3, n_embd=2, n_head=1))
        with torch.no_grad():
            head.query.weight.copy_(torch.tensor([[1., 0.], [0., 2.]]))
            head.key.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
            head.value.weight.copy_(torch.tensor([[1., 1.], [1., -1.]]))

        query, key, value = head.project_qkv(torch.tensor([[[2., 3.]]]))
        torch.testing.assert_close(query, torch.tensor([[[2., 6.]]]))
        torch.testing.assert_close(key, torch.tensor([[[3., 2.]]]))
        torch.testing.assert_close(value, torch.tensor([[[5., -1.]]]))
        self.assertIsNot(head.query.weight, head.key.weight)
        self.assertIsNot(head.key.weight, head.value.weight)

    def test_known_scores_compare_each_query_with_each_key(self) -> None:
        query = torch.tensor([[[1., 2.], [3., 0.]]])
        key = torch.tensor([[[2., 0.], [1., 3.]]])
        scores = AttentionHead.attention_scores(query, key)
        # Query [1, 2] scores 2 against Key A, and 7 against Key B.
        expected = torch.tensor([[[2., 7.], [6., 3.]]])
        torch.testing.assert_close(scores, expected)
        print("\nQueries:", query.tolist())
        print("Keys:", key.tolist())
        print("Raw scores (rows = Queries, columns = Keys):", scores.tolist())

    def test_embeddings_feed_projections_and_gradients_reach_parameters(self) -> None:
        config = ModelConfig(vocab_size=5, block_size=4, n_embd=8, n_head=2)
        embeddings = TokenAndPositionEmbeddings(config)
        head = AttentionHead(config)
        x = embeddings(torch.tensor([[0, 1, 2], [2, 3, 4]]))
        query, key, value = head.project_qkv(x)
        for projection in (query, key, value):
            self.assertEqual(projection.shape, (2, 3, 4))
        scores = head.attention_scores(query, key)
        self.assertEqual(scores.shape, (2, 3, 3))
        self.assertEqual(scores.device.type, "cpu")

        # Synthetic objective checks connectivity only, not language modelling.
        (scores.sum() + value.sum()).backward()
        for parameter in list(head.parameters()) + list(embeddings.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
