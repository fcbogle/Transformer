"""Reload a trained checkpoint and sample characters without retraining."""

import argparse
import math
from pathlib import Path

import torch

from model import ModelConfig, ShakespeareTransformer


@torch.no_grad()
def sample_tokens(model: ShakespeareTransformer, tokens: torch.Tensor,
                  max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    """Append sampled IDs, using only the most recent block_size characters."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = tokens[:, -model.config.block_size:]
            logits, _ = model(context)
            probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    finally:
        model.train(was_training)
    return tokens


def select_device(requested: str = "auto") -> str:
    """Prefer CUDA, then Apple Metal, with CPU as the fallback."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = list((Path(__file__).resolve().parent / "checkpoints").glob("*/model.pt"))
        if not candidates:
            parser.error("No checkpoint found. Run python train.py first.")
        checkpoint_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ShakespeareTransformer(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    if not args.prompt:
        parser.error("The prompt must contain at least one character.")
    unknown = sorted(set(args.prompt) - stoi.keys())
    if unknown:
        parser.error(f"Prompt contains characters absent from the corpus: {unknown!r}")
    tokens = torch.tensor([[stoi[char] for char in args.prompt]], device=device)
    result = sample_tokens(model, tokens, args.max_new_tokens, args.temperature)
    print("".join(itos[index] for index in result[0].tolist()))


if __name__ == "__main__":
    main()
