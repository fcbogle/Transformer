"""Train on Shakespeare and report training/validation loss to the terminal and CSV."""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
import time

import torch

from dataset import (
    DEFAULT_DATA_PATH, CharacterTokenizer, encode_text, get_batch,
    load_raw_text, split_data,
)
from generate import sample_tokens, select_device
from model import ModelConfig, ShakespeareTransformer


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    """Starting settings for training; model dimensions live in model.py."""

    data_path: Path = DEFAULT_DATA_PATH
    checkpoint_dir: Path = PROJECT_DIR / "checkpoints"
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100
    seed: int = 1337


@torch.no_grad()
def estimate_loss(model: ShakespeareTransformer, splits: dict[str, torch.Tensor],
                  batch_size: int, eval_iters: int, device: str) -> dict[str, float]:
    """Average random batches with dropout disabled; restore the previous mode."""
    was_training = model.training
    model.eval()
    try:
        losses = {}
        for name, tokens in splits.items():
            total = 0.0
            for _ in range(eval_iters):
                x, y = get_batch(tokens, batch_size, model.config.block_size, device)
                _, loss = model(x, y)
                total += loss.item()
            losses[name] = total / eval_iters
        return losses
    finally:
        model.train(was_training)


def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    for name in ("batch_size", "max_steps", "eval_interval", "eval_iters", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    for name, value in (("block_size", 128), ("n_embd", 128), ("n_head", 4), ("n_layer", 4)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.max_steps, args.eval_interval, args.eval_iters) <= 0:
        parser.error("Batch size, steps, evaluation interval and evaluation iterations must be positive.")
    if not 0 < args.learning_rate < float("inf") or args.sample_tokens < 0:
        parser.error("Learning rate must be finite and positive; sample tokens must be non-negative.")
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    text = load_raw_text(args.data_path)
    tokenizer = CharacterTokenizer(text)
    train_tokens, validation_tokens = split_data(encode_text(text, tokenizer))
    config = ModelConfig(tokenizer.vocab_size, args.block_size, args.n_embd,
                         args.n_head, args.n_layer, args.dropout)
    splits = {"train": train_tokens, "validation": validation_tokens}
    for name, tokens in splits.items():
        if len(tokens) <= config.block_size:
            parser.error(f"The {name} split needs at least {config.block_size + 1} characters.")
    model = ShakespeareTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A fresh directory for every run keeps previous checkpoints intact.
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.checkpoint_dir))
    checkpoint_path = run_dir / "model.pt"
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Run files: {run_dir}", flush=True)
    start = time.monotonic()
    completed_steps = 0

    def save_checkpoint() -> None:
        # Replace only this run's checkpoint, after a complete temporary write.
        temporary = run_dir / "model.tmp"
        torch.save({
            "model_state_dict": model.state_dict(), "config": asdict(config),
            "stoi": tokenizer.stoi, "itos": tokenizer.itos,
            "optimizer_state_dict": optimizer.state_dict(), "step": completed_steps,
            "training_config": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
        }, temporary)
        temporary.replace(checkpoint_path)

    with (run_dir / "losses.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["step", "train_loss", "validation_loss", "elapsed_seconds"])

        def report() -> None:
            losses = estimate_loss(model, splits, args.batch_size, args.eval_iters, device)
            elapsed = time.monotonic() - start
            print(f"step {completed_steps}/{args.max_steps} | train {losses['train']:.4f} | "
                  f"validation {losses['validation']:.4f} | elapsed {elapsed:.1f}s", flush=True)
            writer.writerow([completed_steps, losses["train"], losses["validation"], round(elapsed, 2)])
            log.flush()

        try:
            report()  # Untrained baseline for comparison.
            model.train()
            for step in range(1, args.max_steps + 1):
                x, y = get_batch(train_tokens, args.batch_size, config.block_size, device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                completed_steps = step
                if step % args.eval_interval == 0 or step == args.max_steps:
                    report()
                    save_checkpoint()
        except KeyboardInterrupt:
            print("\nTraining interrupted; saving current weights.", flush=True)
        finally:
            save_checkpoint()
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    if args.sample_tokens:
        first_token = tokenizer.stoi['\n'] if '\n' in tokenizer.stoi else 0
        prompt = torch.tensor([[first_token]], device=device)
        generated = sample_tokens(model, prompt, args.sample_tokens)
        print("\nSample:\n" + tokenizer.decode(generated[0].tolist()), flush=True)


if __name__ == "__main__":
    main()
