"""Embeddings, causal attention, and blocks for a small Transformer.

Next implement ShakespeareTransformer.

Tensor notation: B = batch size, T = sequence length, C = embedding dimension,
V = vocabulary size, H = number of attention heads.
The model will map token IDs (B, T) to vocabulary logits (B, T, V).
"""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Model dimensions; derive vocab_size from the training corpus."""

    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class TokenAndPositionEmbeddings(nn.Module):
    """Convert character IDs into learned token-plus-position vectors.

    Both tables start with random values and are updated during training.
    This is the input component of the future Transformer, not a language model.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.block_size = config.block_size
        # Each row represents one character (V, C) or one position (block_size, C).
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs (B, T) to floating-point embeddings (B, T, C)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, T)")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must contain torch.long integer IDs")
        sequence_length = token_ids.shape[1]
        if not 1 <= sequence_length <= self.block_size:
            raise ValueError(f"Sequence length must be between 1 and {self.block_size}")

        token_vectors = self.token_embedding(token_ids)  # (B, T, C)
        positions = torch.arange(sequence_length, device=token_ids.device)  # (T,)
        position_vectors = self.position_embedding(positions)  # (T, C)
        # Broadcasting adds the same position vectors to every sequence in B.
        return token_vectors + position_vectors  # (B, T, C)


class AttentionHead(nn.Module):
    """One attention head, built incrementally for inspection.

    Projects Q/K/V, scales and masks scores, applies softmax and dropout,
    then combines Values. Use the individual methods to inspect each stage.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        # Three independent learned matrices; the same input enters each one.
        # nn.Linear stores weights as (head_size, C) and computes x @ weight.T.
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map embeddings (B, T, C) to three tensors (B, T, head_size)."""
        if x.ndim != 3 or x.shape[-1] != self.query.in_features:
            raise ValueError("x must have shape (B, T, n_embd)")
        return self.query(x), self.key(x), self.value(x)

    @staticmethod
    def attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Return raw dot products (B, T, T), before scaling or masking.

        Entry [b, i, j] compares Query i with Key j in batch sequence b.
        These are scores, not probabilities; future positions are still visible.
        """
        if query.ndim != 3 or key.shape != query.shape:
            raise ValueError("query and key must share shape (B, T, head_size)")
        return query @ key.transpose(-2, -1)

    def scale_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Divide raw scores (B, T, T) by sqrt(head_size), preserving shape.

        Dot products tend to grow with head_size. Scaling keeps the later
        softmax from becoming overly peaked just because the head is wider.
        The returned scores still need causal masking and softmax.
        """
        if scores.ndim != 3 or scores.shape[-2] != scores.shape[-1]:
            raise ValueError("scores must have shape (B, T, T)")
        return scores / sqrt(self.head_size)

    def apply_causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Mask future positions in scores (B, T, T), preserving shape.

        Row i is Query i; column j is Key j. Replace scores where j > i
        with negative infinity so softmax will give those positions zero weight.
        Current and earlier positions remain visible; the input is unchanged.
        """
        if scores.ndim != 3 or scores.shape[-2] != scores.shape[-1]:
            raise ValueError("scores must have shape (B, T, T)")
        sequence_length = scores.shape[-1]
        # True above the diagonal marks future tokens. Broadcast across B.
        future_positions = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=scores.device
        ).triu(diagonal=1)  # (T, T)
        return scores.masked_fill(future_positions, float("-inf"))

    def attention_weights(self, masked_scores: torch.Tensor) -> torch.Tensor:
        """Convert masked scores (B, T, T) to probabilities of the same shape.

        Softmax across keys (the last dimension) makes each Query's row sum
        to one. Positions masked with -inf receive exactly zero weight.
        Apply scaling and causal masking before calling this method.
        """
        if masked_scores.ndim != 3 or masked_scores.shape[-2] != masked_scores.shape[-1]:
            raise ValueError("masked_scores must have shape (B, T, T)")
        return torch.softmax(masked_scores, dim=-1)

    def apply_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply training dropout to attention weights (B, T, T).

        Randomly zero weights with probability config.dropout and scale the
        survivors by 1 / (1 - dropout). Rows no longer necessarily sum to one.
        Masked positions stay zero. In eval() mode, weights pass through unchanged.
        """
        if weights.ndim != 3 or weights.shape[-2] != weights.shape[-1]:
            raise ValueError("weights must have shape (B, T, T)")
        return self.dropout(weights)

    def combine_values(self, weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Mix Values (B, T, head_size) using attention weights (B, T, T).

        Each Query's output is a weighted sum of the Value vectors across
        key positions. Causal weights prevent future Values from contributing.
        """
        if weights.ndim != 3 or weights.shape[-2] != weights.shape[-1]:
            raise ValueError("weights must have shape (B, T, T)")
        if value.ndim != 3 or value.shape != (*weights.shape[:2], self.head_size):
            raise ValueError("value must have shape (B, T, head_size) matching weights")
        return weights @ value  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map embeddings (B, T, C) to causal attention output (B, T, head_size)."""
        query, key, value = self.project_qkv(x)
        scores = self.attention_scores(query, key)
        scaled_scores = self.scale_scores(scores)
        masked_scores = self.apply_causal_mask(scaled_scores)
        weights = self.attention_weights(masked_scores)
        dropped_weights = self.apply_dropout(weights)
        return self.combine_values(dropped_weights, value)


class MultiHeadAttention(nn.Module):
    """Run independent causal heads, then learn how to mix their outputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # ModuleList registers every head's parameters for training and saving.
        self.heads = nn.ModuleList(
            AttentionHead(config) for _ in range(config.n_head)
        )
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, T, C) to (B, T, C), preserving causal visibility."""
        # Each head receives the same input and returns (B, T, C / H).
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)  # (B, T, C)
        # Mix features within each token; do not mix sequence positions here.
        projected = self.projection(combined)  # (B, T, C)
        return self.dropout(projected)


class FeedForward(nn.Module):
    """Process each token independently with shared learned feature transforms."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, T, C) to (B, T, C) without mixing token positions."""
        if x.ndim != 3 or x.shape[-1] != self.expand.in_features:
            raise ValueError("x must have shape (B, T, n_embd)")
        expanded = self.expand(x)  # (B, T, 4 * C)
        activated = self.activation(expanded)  # Nonlinear feature processing.
        projected = self.project(activated)  # (B, T, C)
        return self.dropout(projected)


class TransformerBlock(nn.Module):
    """A pre-normalised causal attention and feed-forward block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Each LayerNorm normalises features within one token, never across time.
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine token features (B, T, C), preserving shape and causality."""
        if x.ndim != 3 or x.shape[-1] != self.layer_norm_1.normalized_shape[0]:
            raise ValueError("x must have shape (B, T, n_embd)")
        # Residual additions preserve a direct path for features and gradients.
        x = x + self.self_attention(self.layer_norm_1(x))  # (B, T, C)
        x = x + self.feed_forward(self.layer_norm_2(x))  # (B, T, C)
        return x
