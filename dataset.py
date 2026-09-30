"""Load Shakespeare text, tokenise it, and prepare next-character batches."""

from collections.abc import Iterable
from pathlib import Path

import torch


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "input.txt"


def load_raw_text(path: str | Path = DEFAULT_DATA_PATH) -> str:
    """Return the UTF-8 corpus as text, preserving spaces and line breaks."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8", newline="") as source:
            text = source.read()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Save the Shakespeare corpus there; "
            "see README.md for download instructions."
        ) from error
    if not text:
        raise ValueError(f"Dataset is empty: {path}")
    return text


class CharacterTokenizer:
    """Assign a stable integer ID to each unique character in the corpus.

    Sorting the vocabulary makes the mappings reproducible for the same set
    of characters. Unknown characters or IDs raise KeyError; no text is dropped.
    """

    def __init__(self, text: str) -> None:
        characters = sorted(set(text))
        if not characters:
            raise ValueError("Cannot build a vocabulary from empty text")
        self.stoi = {character: index for index, character in enumerate(characters)}
        self.itos = {index: character for character, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        """Number of unique characters, for ModelConfig.vocab_size."""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Convert characters to their vocabulary IDs."""
        return [self.stoi[character] for character in text]

    def decode(self, tokens: Iterable[int]) -> str:
        """Convert vocabulary IDs back to characters."""
        return "".join(self.itos[token] for token in tokens)


def encode_text(text: str, tokenizer: CharacterTokenizer) -> torch.Tensor:
    """Encode the corpus as a one-dimensional CPU tensor of integer IDs.

    torch.long stores integers suitable for embedding lookup and loss targets.
    Use tensor.tolist() to convert IDs back to a Python list for decode().
    """
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(len(tokens), "tokens; first 10:", tokens[:10].tolist())
    return tokens


def split_data(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first 90% for training and the remaining 10% for validation.

    Preserve the original order. Each split is sampled separately so a training
    sequence never includes validation tokens. Round the training length down.
    """
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("tokens must be a one-dimensional torch.long tensor")
    if len(tokens) < 2:
        raise ValueError("At least two tokens are needed for a train/validation split")
    split_index = int(0.9 * len(tokens))
    return tokens[:split_index], tokens[split_index:]


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input/target sequences from one split, with replacement.

    Keep the source split on CPU; transfer only the sampled batch to device.
    B = batch_size, T = block_size. Both x and y have shape (B, T).
    Set torch.manual_seed(...) before sampling for repeatable CPU batches.
    """
    if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.device.type != "cpu":
        raise ValueError("tokens must be a one-dimensional CPU torch.long tensor")
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive")
    if len(tokens) <= block_size:
        raise ValueError(
            f"This split has {len(tokens)} tokens; at least {block_size + 1} "
            "are needed for an input sequence and its next-character target"
        )

    # randint's upper bound is exclusive. Reserve one extra token for y.
    starts = torch.randint(len(tokens) - block_size, (batch_size,)).tolist()
    x = torch.stack([tokens[start : start + block_size] for start in starts])
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)
