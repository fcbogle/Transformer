# Shakespeare Transformer

An educational decoder-only Transformer language model in Python 3.11+ and
PyTorch. The aim is to make the path from characters to generated text easy to
follow. See [AGENTS.MD](AGENTS.MD) for the implementation requirements.

## Current status

For a worked example of every stage and a CPU script with optional breakpoints,
see [MODEL_WALKTHROUGH.md](MODEL_WALKTHROUGH.md). Run
`python debug_walkthrough.py --step` to inspect the implemented stages interactively.

Raw dataset loading, character tokenisation, token tensors, a sequential 90/10
split, next-character batches, and token/positional embeddings are implemented.
The complete single causal attention head is also implemented, from Q/K/V
projections through weighted Value mixing. Multi-head attention combines independent
heads, projects their outputs back to the embedding size, and applies dropout.
The position-wise feed-forward network and complete pre-normalised
`TransformerBlock` are also implemented.
`ShakespeareTransformer` now combines embeddings, a configurable stack of blocks,
final LayerNorm, and a vocabulary head, with optional next-character loss.
AdamW training, validation, terminal/CSV monitoring, checkpoint saving/loading,
and temperature-based generation are implemented. The dataset and trained checkpoints
are not tracked in version control.

## Project layout

```text
.
├── AGENTS.MD
├── README.md
├── requirements.txt
├── data/                 # Put the corpus in input.txt
├── dataset.py            # Text loading, tokenisation, tensors, and batches
├── model.py              # Configuration, embeddings, and Transformer blocks
├── train.py              # Training, validation, loss logging, and checkpoints
├── generate.py           # Load a checkpoint and sample characters
├── checkpoints/          # Saved model weights and vocabulary
└── tests/                # Dataset, tokenizer, and model component checks
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
and learn their values during training.

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
use `head.eval()` to compare outputs.

Run the small numerical example, which prints scores of 2 and 7 for the first
Query:

```bash
python -m pytest tests/test_attention.py -v -s
```

## Multi-head attention

`MultiHeadAttention` gives the same input to several independent `AttentionHead`
instances. Each head has its own learned Q/K/V projections. Their outputs are
concatenated along the feature dimension, then a learned linear projection mixes
those features within each token. Output dropout follows the projection.

Continuing the embeddings example:

```python
from model import MultiHeadAttention

attention = MultiHeadAttention(model)
attention.eval()  # Disable dropout in all heads and at the output for inspection.
output = attention(vectors)
print(output.shape)  # torch.Size([32, 128, 128]): B, T, C
```

Calling `attention(vectors)` runs its `forward()`, which calls each head's existing
`forward()`. With `n_embd=128` and `n_head=4`, each head returns 32 features per
token; concatenating four heads restores 128. Input and output both have shape
`(B, T, C)`, ready for a later residual connection. The output projection does
not mix sequence positions, so causal visibility is preserved.

Use `attention.train()` to enable dropout again. These weights are still randomly
initialised; they have not learned language patterns.

Run the multi-head checks:

```bash
python -m unittest discover -s tests -p 'test_multihead_attention.py' -v
```

## Feed-forward network

`FeedForward` processes each token independently, using the same learned layers
at every position. Attention gathers information from visible tokens; the
feed-forward network processes the resulting features within each token.

The calculation is `Linear(C, 4*C) → GELU → Linear(4*C, C) → Dropout`.
With the default configuration, features expand from 128 to 512 and return to
128. GELU adds a nonlinear transformation so the network can learn more than a
single linear mapping. Input and output both have shape `(B, T, C)`.

Continuing the multi-head example:

```python
from model import FeedForward

feed_forward = FeedForward(model)
feed_forward.eval()  # Disable dropout for inspection.
processed = feed_forward(output)
print(processed.shape)  # torch.Size([32, 128, 128]): B, T, C
```

Calling `feed_forward(output)` runs its `forward()`. Changing one input token's
vector cannot affect another position's output. Call `feed_forward.train()` to
enable dropout during training. This example demonstrates the components;
`TransformerBlock` combines them with layer normalisation and residual connections.

Run the feed-forward checks:

```bash
python -m unittest discover -s tests -p 'test_feedforward.py' -v
```

## Transformer block

`TransformerBlock` combines the existing components in this order:

```python
x = x + self_attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

Each LayerNorm normalises the features within one token and has its own learned
scale and bias. Normalisation happens before each branch (pre-normalisation).
Each residual addition keeps the incoming representation and adds the branch's
learned update, providing a direct path for both features and gradients.
All stages preserve shape `(B, T, C)`. Normalisation and feed-forward processing
do not mix token positions, so the attention mask preserves causality throughout
the block, including when multiple blocks are stacked.

Continuing the embeddings example, pass the original embedding vectors to the
block; it runs both attention and feed-forward processing internally:

```python
from model import TransformerBlock

block = TransformerBlock(model)
block.eval()  # Disable dropout throughout the block for inspection.
refined = block(vectors)
print(refined.shape)  # torch.Size([32, 128, 128]): B, T, C
```

Use `block.train()` to enable dropout for training. A block returns refined token
features, not vocabulary predictions. `ShakespeareTransformer`
combines embeddings, a stack of blocks, final normalisation, and a vocabulary
output layer. The `n_layer` setting controls that stack; a `TransformerBlock`
instance always represents one block.

Run the block checks:

```bash
python -m unittest discover -s tests -p 'test_transformer_block.py' -v
```

## Complete language model

`ShakespeareTransformer` maps character IDs `(B, T)` to vocabulary logits
`(B, T, V)`. Each position predicts the character that follows it, using only
the current and earlier input characters. The final LayerNorm normalises each
token's features, and the linear language model head maps those features to
one score for every vocabulary character.

```python
import torch
from dataset import CharacterTokenizer, encode_text, get_batch
from model import ModelConfig, ShakespeareTransformer

tokenizer = CharacterTokenizer("To be or " * 20)
tokens = encode_text("To be or " * 20, tokenizer)
x, y = get_batch(tokens, batch_size=1, block_size=8)
config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=8,
                     n_embd=8, n_head=2, n_layer=2)
language_model = ShakespeareTransformer(config)
logits, loss = language_model(x, y)
print(logits.shape)  # torch.Size([1, 8, 6]): B, T, V
print(loss.item())   # Scalar cross-entropy against shifted targets y.
loss.backward()     # Gradients reach embeddings and every Transformer block.

language_model.eval()
with torch.no_grad():
    logits, loss = language_model(x)  # Without targets, loss is None.
    probabilities = torch.softmax(logits[:, -1, :], dim=-1)
```

The model returns `(logits, loss)` in both cases. Pass raw logits directly to
cross-entropy; it performs log-softmax internally. Apply softmax when you want
probabilities for inspection or sampling. These predictions are untrained until
an optimiser updates the model weights.

Run the language model checks:

```bash
python -m unittest discover -s tests -p 'test_language_model.py' -v
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
Each block uses pre-normalisation, residual connections, explicit multi-head
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

With `data/input.txt` prepared, run in the VS Code terminal with `.venv` active:

```bash
python train.py
python generate.py
```

Training selects CUDA, then Apple Metal (MPS), then CPU according to availability.
It prints the device, parameter count, and run directory, then evaluates an
untrained baseline. It reports training/validation loss after each evaluation
interval and at the final step. For a smaller first experiment with frequent
updates, use:

```bash
python train.py --max-steps 1000 --batch-size 16 --block-size 64 --n-embd 64 --n-head 4 --n-layer 2 --eval-interval 50 --eval-iters 10
```

Loss should trend downward over many evaluations; individual readings fluctuate
because evaluation samples random batches. Validation loss measures prediction on
the held-out final 10% of the corpus. If training loss keeps falling while
validation loss rises persistently, the model may be overfitting.

Each run creates a fresh `checkpoints/run-.../` directory. Its `losses.csv` records
step, training loss, validation loss, and elapsed seconds; open it in VS Code or
a spreadsheet to compare readings or chart both losses against step. The file
is flushed after each evaluation. Use a smaller `--eval-interval` for more frequent
updates and a smaller `--eval-iters` to spend less time evaluating.

`model.pt` is saved after evaluations and when training exits. Press **Ctrl+C**
once to stop and save current weights. Checkpoints contain `model_state_dict`,
`config`, `stoi`, `itos`, optimiser state, training settings, and completed step.
The CLI starts a new training run; resuming optimiser state is not implemented.

`python generate.py` loads the most recently saved checkpoint under `checkpoints/`.
Choose a particular run, prompt, and sampling temperature with:

```bash
python generate.py --checkpoint checkpoints/run-YOUR_RUN/model.pt --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8
```

Generation appends the requested number of characters to the prompt, using only
the most recent `block_size` tokens. Lower temperatures favour likely characters;
higher temperatures increase variety. Prompts must use characters in the corpus.
Training prints a short generated sample at the end; early samples will be noisy.

CPU supports training and small correctness checks, but full training can take
substantially longer than on a GPU. Runtime depends on hardware and configuration.
Use `--device cpu` to force CPU, and `python train.py --help` for all settings.
Useful Shakespeare-like text requires more learning than a brief smoke test.

## Implementation milestones

1. Dataset loading, character tokenisation, tensor conversion, the sequential
   split, and shifted batches are implemented and tested.
2. Token and positional embeddings are implemented and tested, including
   gradient flow to both tables. The complete `AttentionHead` is tested for
   weighted Value mixing, output shape, gradients, and causality.
   `MultiHeadAttention` is implemented and tested for head combination, output
   projection, dropout, gradients, CPU shapes, and causality.
   `FeedForward` is implemented and tested for the GELU calculation, shapes,
   position independence, gradients, and dropout.
   `TransformerBlock` is implemented and tested for pre-normalisation, residual
   connections, gradients, CPU shapes, and causality through stacked blocks.
   `ShakespeareTransformer` adds final normalisation and vocabulary logits,
   tested for shapes and causal masking end to end.
3. AdamW training and validation are implemented, with tests for decreasing
   training and validation loss on a small repeated text sample.
4. Checkpoint saving/loading and temperature-based generation are implemented,
   with tests for checkpoint round-tripping and generated length.
5. A full Shakespeare training run and qualitative sample comparison remain
   experiments for the learner; short correctness checks do not establish text quality.
