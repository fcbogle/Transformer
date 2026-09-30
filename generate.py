"""Generation scaffold: reload a checkpoint and sample new characters.

Restore the model configuration, weights, and character mappings. Encode a
prompt, crop to the most recent block_size tokens, and sample from the final
position's logits using a configurable temperature. Repeat without retraining.
"""


def main() -> None:
    """Placeholder for checkpoint loading and autoregressive generation."""
    raise SystemExit(
        "Generation is not implemented yet. This is the project scaffold. "
        "A trained checkpoint will be required; see README.md."
    )


if __name__ == "__main__":
    main()
