# Shakespeare Transformer

An educational decoder-only Transformer language model in Python 3.11+ and
PyTorch. The aim is to make the path from characters to generated text easy to
follow. See [AGENTS.MD](AGENTS.MD) for the implementation requirements.

## Current status

Raw dataset loading, character tokenisation, token tensors, a sequential 90/10
split, next-character batches, and token/positional embeddings are implemented.
The complete single causal attention head is also implemented, from Q/K/V
projections through weighted Value mixing.
The full language model, training loop, and generation
are not implemented yet. The dataset and trained checkpoints
are not tracked in version control.

## Project layout

```text
.
├── AGENTS.MD
├── README.md
├── requirements.txt
├── data/                 # Put the corpus in input.txt
├── dataset.py            # Text loading, tokenisation, tensors, and batches
├── model.py              # Configuration, embeddings, and a complete attention head
├── train.py              # Training entry-point placeholder
├── generate.py           # Generation entry-point placeholder
├── checkpoints/          # Saved model weights and vocabulary
└── tests/                # Dataset and tokenizer checks
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.11 or newer. PyTorch is the only runtime dependency.

## Prepare the dataset

Save a UTF-8 Shakespeare corpus as `data/input.txt`. For example, download
[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
from the char-rnn repository, running this command from the project root:

```bash
curl --fail --location https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --output data/input.txt
```

Dataset preparation is separate from model code. The corpus and generated
checkpoints are excluded from version control.

The tokenizer assigns an integer to each unique character in the corpus,
with `stoi` and `itos` mappings and `encode`/`decode` methods. Data is split
sequentially into 90% training and 10% validation. Each input sequence's target
is the same sequence shifted forward by one character.

After preparing the dataset, access its raw text and token IDs in Python:

```python
from dataset import CharacterTokenizer, load_raw_text
from model import ModelConfig

text = load_raw_text()  # Reads data/input.txt relative to this project
print(text[:200])

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)
assert tokenizer.decode(tokens) == text
print(tokenizer.stoi)  # Character → integer ID
print(tokenizer.itos)  # Integer ID → character
config = ModelConfig(vocab_size=tokenizer.vocab_size)
```

You can also pass a custom path to `load_raw_text("path/to/corpus.txt")`.
Missing or empty files produce an explanatory error. Vocabulary IDs follow
sorted character order; encoding an unknown character or decoding an unknown
ID raises `KeyError`.

## Prepare training batches

```python
import torch
from dataset import CharacterTokenizer, encode_text, get_batch, load_raw_text, split_data
from model import ModelConfig
from train import TrainingConfig

training = TrainingConfig()
torch.manual_seed(training.seed)
text = load_raw_text(training.data_path)
tokenizer = CharacterTokenizer(text)
model = ModelConfig(vocab_size=tokenizer.vocab_size)

tokens = encode_text(text, tokenizer)  # One-dimensional torch.long tensor
train_tokens, validation_tokens = split_data(tokens)
x, y = get_batch(train_tokens, training.batch_size, model.block_size)

print(x.shape, y.shape)  # Both torch.Size([32, 128])
print(tokenizer.decode(x[0].tolist()))  # First input sequence
print(tokenizer.decode(y[0].tolist()))  # Its next-character targets
```

`tokenizer.encode()` still returns a Python list. `encode_text()` converts that
list into a tensor of integer IDs. `split_data()` preserves order and rounds
the training length down. `get_batch()` samples random starting positions within
one split, with replacement, then stacks the sequences into rows. Both returned
tensors have shape `(B, T)`, where `B = batch_size` and `T = block_size`.
The target at each position is the next character after the corresponding input.
Each split needs at least `block_size + 1` tokens to produce a batch.

Pass `validation_tokens` instead of `train_tokens` to sample validation batches.
The source tokens stay on CPU; the optional `device` argument to `get_batch()`
selects where to put the returned batch.

## Token and positional embeddings

`TokenAndPositionEmbeddings` in `model.py` is a PyTorch `nn.Module` containing
two learnable lookup tables. The token table has shape `(vocab_size, n_embd)`;
the position table has shape `(block_size, n_embd)`. They are randomly initialised
and will learn their values when training is implemented.

Continuing the batch example above:

```python
from model import TokenAndPositionEmbeddings

embeddings = TokenAndPositionEmbeddings(model)
vectors = embeddings(x)
print(vectors.shape)  # torch.Size([32, 128, 128]): B, T, C
```

Calling the module runs its `forward()` method. Each character ID selects a
token vector; each position from zero to `T - 1` selects a position vector.
The two are added together, with the same position vectors used for each
sequence in the batch. Input IDs have shape `(B, T)`; the resulting floating-point
tensor has shape `(B, T, C)`, where `C = n_embd`. Sequences may be shorter than
`block_size`, but cannot be empty or longer than it. The module and input must
be on the same device (both use CPU in this example).

These vectors are ready for attention. They are not yet vocabulary logits or
next-character predictions; the remaining Transformer layers come next.

## First attention calculations

`AttentionHead` implements seven inspectable stages:

```python
from model import AttentionHead

head = AttentionHead(model)
query, key, value = head.project_qkv(vectors)
scores = head.attention_scores(query, key)
scaled_scores = head.scale_scores(scores)
masked_scores = head.apply_causal_mask(scaled_scores)
weights = head.attention_weights(masked_scores)
dropped_weights = head.apply_dropout(weights)
output = head.combine_values(dropped_weights, value)

print(query.shape)   # (32, 128, 32): B, T, head_size
print(scores.shape)  # (32, 128, 128): B, Query position, Key position
print(output.shape)  # (32, 128, 32): B, T, head_size
```

Here `head_size = n_embd // n_head`. Three independent linear layers project the
same embeddings into Queries, Keys, and Values. The weights are learned
parameters; the resulting vectors depend on the current input. A dot product
compares every Query with every Key within the same sequence.

`scores` contains raw dot products. `scale_scores()` divides them by
`sqrt(head_size)` so larger heads do not make the later softmax overly peaked.
Both tensors have shape `(B, T, T)`; scaling leaves the raw scores unchanged.
`apply_causal_mask()` replaces future-position scores (above the diagonal)
with `-inf`, preserving current and earlier scores. It returns a new tensor
on the same device with shape `(B, T, T)`. `attention_weights()` applies softmax
across key positions (`dim=-1`), converting these scores to probabilities.
Each row sums to one, and masked positions have zero probability.
`apply_dropout()` randomly zeros attention weights during training using
`config.dropout` (default 0.1), helping reduce overfitting. Surviving weights
are scaled by `1 / (1 - dropout)`, so rows may no longer sum to one. Causal
zeros stay zero. Calling `head.eval()` disables dropout; `head.train()`
enables it again. New modules start in training mode.
`combine_values()` multiplies the weights by the Value vectors. Each Query
receives a weighted sum of visible Values, producing shape `(B, T, head_size)`.
Calling `output = head(vectors)` runs all seven stages through `forward()`.
Separate calls in training mode can differ because dropout samples a fresh mask;
use `head.eval()` to compare outputs. Next comes multi-head attention.

Run the small numerical example, which prints scores of 2 and 7 for the first
Query:

```bash
python -m pytest tests/test_attention.py -v -s
```

## Tests and experiments

After installing the requirements and activating `.venv`, run all checks:

```bash
python -m unittest discover -s tests -v
```

Alternatively, install pytest in the same environment and run:

```bash
python -m pip install pytest
python -m pytest -v -s
```

For VS Code testing, select this project's `.venv/bin/python` as the Python
interpreter so both PyTorch and pytest are available.

For printed character-to-integer examples, run:

```bash
python -m unittest discover -s tests -p 'test_tokenizer_examples.py' -v
```

`ShakespeareExamplesTests` prints the first line of `data/input.txt`, including
its line ending, and the corresponding IDs from the full corpus vocabulary.
It skips if the dataset is missing. `TokenizerExperimentTests` uses a small
editable string in `tests/test_tokenizer_examples.py`; change `text` there to
experiment without the dataset. Its vocabulary is built from that string alone,
so the IDs can differ from those in the Shakespeare vocabulary. Both examples
check that encoding preserves character count and decoding restores the text.

## Planned architecture

```text
Shakespeare
    ↓
Character Tokenisation
    ↓
Token IDs
    ↓
Token + Position Embeddings
    ↓
┌────────────────────────┐
│ Transformer Block × N  │
│ Causal Self-Attention  │
│ Feed-Forward Network   │
└────────────────────────┘
    ↓
Language Model Head
    ↓
Next Character Probability
    ↓
Generated Shakespeare
```

Causal attention lets each character use only its preceding context and itself.
Each block will use pre-normalisation, residual connections, explicit multi-head
scaled dot-product attention, and a GELU feed-forward network. The output logits
will have shape `(B, T, V)`: batch size, sequence length, vocabulary size.
Cross-entropy will train the model to predict the next character.

## Configuration defaults

Defaults live in `ModelConfig` in `model.py` and `TrainingConfig` in `train.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `vocab_size` | From corpus | Number of unique characters |
| `block_size` | 128 | Maximum context length |
| `n_embd` | 128 | Embedding dimension |
| `n_head` | 4 | Number of attention heads; must divide `n_embd` |
| `n_layer` | 4 | Number of Transformer blocks |
| `dropout` | 0.1 | Dropout probability |
| `batch_size` | 32 | Sequences per training batch |
| `learning_rate` | 0.0003 | AdamW learning rate |
| `max_steps` | 5000 | Optimiser updates |
| `eval_interval` | 250 | Updates between loss evaluations |
| `eval_iters` | 100 | Batches per evaluation split |
| `seed` | 1337 | Random seed |

## Training and generation

Once implemented, the entry points will be:

```bash
python train.py
python generate.py
```

Currently these commands exit with an explanatory scaffold message. Training
will select CUDA, then Apple Metal (MPS), then CPU according to availability,
report training and validation loss, save a checkpoint, and generate a sample.
CPU will support small correctness checks; full training may take substantially
longer than on a GPU. Runtime will depend on hardware and configuration, and
useful text will require training beyond a brief smoke test.

Checkpoints will contain `model_state_dict`, `config`, `stoi`, and `itos`, under
`checkpoints/`, without silently overwriting unrelated files. Generation will
reload these values and sample one character at a time from the most recent
`block_size` tokens, with configurable temperature.

## Implementation milestones

1. Dataset loading, character tokenisation, tensor conversion, the sequential
   split, and shifted batches are implemented and tested.
2. Token and positional embeddings are implemented and tested, including
   gradient flow to both tables. The complete `AttentionHead` is tested for
   weighted Value mixing, output shape, gradients, and causality.
   Next implement `MultiHeadAttention`, `FeedForward`,
   `TransformerBlock`, and `ShakespeareTransformer` incrementally; check CPU
   execution, tensor shapes, and causal masking as each component is added.
3. Implement AdamW training and validation; verify decreasing loss on a small
   sample before attempting a full training run.
4. Implement checkpoint saving/loading and temperature-based generation;
   verify checkpoint round-tripping and generated length.
5. Train on Shakespeare and inspect samples for improvement.
