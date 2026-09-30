"""Training scaffold: data → token IDs → batches → next-token prediction.

Corpus loading, tokenisation, the sequential 90/10 split, and shifted batches
are available in dataset.py. Next implement the model, then the AdamW loop.
Finally add validation, checkpoint saving, and a generated sample.
"""

from dataclasses import dataclass
from pathlib import Path

from dataset import DEFAULT_DATA_PATH


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


def main() -> None:
    """Placeholder for the incremental training implementation."""
    raise SystemExit(
        "Training is not implemented yet. This is the project scaffold. "
        "See README.md for dataset preparation and implementation milestones."
    )


if __name__ == "__main__":
    main()
