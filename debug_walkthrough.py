"""Inspect implemented Transformer stages on CPU; use --step for breakpoints."""

import argparse

import torch

from dataset import CharacterTokenizer, encode_text, get_batch, split_data
from model import ModelConfig, TokenAndPositionEmbeddings, TransformerBlock


def pause(stage: str, step: bool) -> None:
    print(stage)
    if step:
        # In pdb, type 'up' to inspect main()'s tensors, then 'continue'.
        breakpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", action="store_true", help="Pause after each stage")
    args = parser.parse_args()
    torch.manual_seed(1337)
    torch.set_printoptions(precision=3, sci_mode=False)

    # Repetition ensures even the 10% validation split has enough tokens.
    text = "To be or " * 20
    tokenizer = CharacterTokenizer(text)
    example_ids = tokenizer.encode("To be or")
    assert tokenizer.decode(example_ids) == "To be or"
    pause("1. Characters → vocabulary IDs: inspect tokenizer.stoi and example_ids", args.step)

    tokens = encode_text(text, tokenizer)
    train_tokens, validation_tokens = split_data(tokens)
    x, y = get_batch(train_tokens, batch_size=1, block_size=8)
    input_text = tokenizer.decode(x[0].tolist())
    target_text = tokenizer.decode(y[0].tolist())
    assert torch.equal(x[:, 1:], y[:, :-1])
    pause(f"2. Shifted batch: input={input_text!r}, target={target_text!r}", args.step)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=8,
                         n_embd=8, n_head=2, n_layer=1, dropout=0.1)
    embeddings = TokenAndPositionEmbeddings(config).eval()
    block = TransformerBlock(config).eval()
    # Evaluation disables dropout for repeatable inspection; nothing is trained.
    with torch.no_grad():
        token_vectors = embeddings.token_embedding(x)
        positions = torch.arange(x.shape[1])
        position_vectors = embeddings.position_embedding(positions)
        embedded = embeddings(x)
        torch.testing.assert_close(embedded, token_vectors + position_vectors)
        pause("3. Token + position vectors: inspect embedded[0, 0] (shape 1, 8, 8)", args.step)

        normalised = block.layer_norm_1(embedded)
        pause("4. First LayerNorm: inspect normalised[0, 0]", args.step)

        head = block.self_attention.heads[0]
        query, key, value = head.project_qkv(normalised)
        pause("5. Q/K/V projections: each has shape (1, 8, 4)", args.step)

        scores = head.attention_scores(query, key)
        scaled_scores = head.scale_scores(scores)
        pause("6. Dot products / sqrt(4): inspect scores[0] and scaled_scores[0]", args.step)

        masked_scores = head.apply_causal_mask(scaled_scores)
        pause("7. Causal mask: inspect masked_scores[0] for -inf above diagonal", args.step)

        weights = head.attention_weights(masked_scores)
        dropped_weights = head.apply_dropout(weights)
        assert weights.triu(diagonal=1).count_nonzero().item() == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 8))
        pause("8. Softmax and dropout: inspect weights[0]; eval disables dropout", args.step)

        head_output = head.combine_values(dropped_weights, value)
        torch.testing.assert_close(head_output[:, 0], value[:, 0])
        pause("9. Weighted Values: inspect head_output (shape 1, 8, 4)", args.step)

        attention = block.self_attention
        head_outputs = [item(normalised) for item in attention.heads]
        combined = torch.cat(head_outputs, dim=-1)
        projected = attention.projection(combined)
        attention_output = attention.dropout(projected)
        torch.testing.assert_close(attention_output, attention(normalised))
        pause("10. Two heads → concatenate → project: inspect combined and projected", args.step)

        residual = embedded + attention_output
        normalised_residual = block.layer_norm_2(residual)
        pause("11. First residual + second LayerNorm: inspect residual", args.step)

        feed_forward = block.feed_forward
        expanded = feed_forward.expand(normalised_residual)
        activated = feed_forward.activation(expanded)
        contracted = feed_forward.project(activated)
        feed_forward_output = feed_forward.dropout(contracted)
        pause("12. MLP: expanded (1, 8, 32) → GELU → contracted (1, 8, 8)", args.step)

        output = residual + feed_forward_output
        torch.testing.assert_close(output, block(embedded))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % config.vocab_size
        changed_output = block(embeddings(changed))
        torch.testing.assert_close(output[:, :4], changed_output[:, :4])
        pause("13. Second residual: output matches block.forward(); causal check passed", args.step)

    print("Walkthrough complete. These are untrained features, not character predictions.")


if __name__ == "__main__":
    main()
