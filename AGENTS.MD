# AGENTS.md

## Editor preference

The user uses **VS Code**, not PyCharm. Use VS Code for editor, debugger,
breakpoint, and environment setup instructions.

## Project Purpose

Build a small, educational **decoder-only Transformer language model in PyTorch** and train it on Shakespeare text.

The project is intended to demonstrate the core ideas behind GPT-style language models:

1. Text tokenisation
2. Token and positional embeddings
3. Causal self-attention
4. Multi-head attention
5. Feed-forward networks
6. Residual connections and layer normalisation
7. Next-token prediction
8. Training and validation
9. Autoregressive text generation

Keep the implementation simple enough to read and understand. Prefer clarity over optimisation.

---

## Target Project Structure

```text
shakespeare-transformer/
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
│   └── input.txt
├── model.py
├── train.py
├── generate.py
└── checkpoints/
```

Create missing directories/files when required.

---

## Technology

Use:

- Python 3.11+
- PyTorch
- Standard Python libraries

Avoid unnecessary frameworks.

Do not use Hugging Face Transformers, Lightning, TensorFlow, Keras, or pre-trained models for the core implementation.

The purpose is to implement the Transformer components directly with PyTorch so the architecture remains visible.

---

## Dataset

Use Shakespeare text as the training corpus.

A suitable small educational corpus is the public-domain Shakespeare text commonly used for character-level language-model demonstrations, for example Tiny Shakespeare.

Store the downloaded/prepared corpus at:

```text
data/input.txt
```

Do not commit very large datasets or generated checkpoints unless explicitly requested.

If automatic downloading is implemented, keep it separate from model code and document the source in `README.md`.

---

## Tokenisation

Start with a **character-level tokenizer**.

Build the vocabulary directly from the unique characters in:

```text
data/input.txt
```

Implement:

```python
encode(text) -> list[int]
decode(tokens) -> str
```

Maintain mappings equivalent to:

```python
stoi = {character: integer_id}
itos = {integer_id: character}
```

Do not introduce BPE or an external tokenizer unless specifically requested.

Character-level tokenisation is intentional because it makes the relationship between text, token IDs, embeddings, and model predictions easy to inspect.

---

## Training Data

Encode the complete corpus into token IDs.

Split sequentially:

```text
90% training
10% validation
```

Create batches containing:

```text
x = sequence of tokens
y = same sequence shifted one token forward
```

For example:

```text
Input:   "To be or"
Target:  "o be or "
```

Use configurable:

```python
batch_size
block_size
```

`block_size` represents the maximum context length used during training.

---

# Model Architecture

Implement a small **decoder-only Transformer**.

Conceptually:

```text
Token IDs
    ↓
Token Embeddings
    +
Positional Embeddings
    ↓
Transformer Block
    ↓
Transformer Block
    ↓
...
    ↓
LayerNorm
    ↓
Linear Language Model Head
    ↓
Vocabulary Logits
```

The model predicts the next token.

---

## Transformer Block

Each block should contain:

```text
Input
  │
  ├── LayerNorm
  │
  └── Multi-Head Causal Self-Attention
          │
          + residual connection
          ↓
      LayerNorm
          │
          └── Feed-Forward Network
                  │
                  + residual connection
                  ↓
                Output
```

Prefer pre-normalisation:

```python
x = x + self_attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

---

# Self-Attention

Implement scaled dot-product attention explicitly.

For input representation `X`, calculate:

```text
Q = XWq
K = XWk
V = XWv
```

Then:

```text
attention_scores = Q @ K.transpose(-2, -1)
attention_scores /= sqrt(head_size)
```

Apply a causal mask before softmax so a token cannot attend to future tokens.

Conceptually:

```text
Token 1 → Token 1
Token 2 → Token 1, Token 2
Token 3 → Token 1, Token 2, Token 3
Token 4 → Token 1, Token 2, Token 3, Token 4
```

Then:

```text
weights = softmax(attention_scores)
output = weights @ V
```

Include dropout where appropriate.

---

# Multi-Head Attention

Use multiple attention heads in parallel.

Each head should learn its own Query, Key, and Value projections.

Concatenate the head outputs and project them back to the model embedding dimension.

Ensure:

```python
n_embd % n_head == 0
```

A reasonable starting configuration is:

```python
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
block_size = 128
batch_size = 32
```

These values should be configurable rather than hard-coded throughout the model.

---

# Feed-Forward Network

Each Transformer block should contain a position-wise MLP.

Use approximately:

```text
n_embd
  ↓
4 × n_embd
  ↓
activation
  ↓
n_embd
```

Use GELU or ReLU.

GELU is preferred.

Include dropout.

---

# Output and Loss

The final model output should have shape:

```text
(batch, sequence, vocabulary_size)
```

Use cross-entropy loss for next-token prediction.

Reshape logits and targets as necessary for:

```python
torch.nn.functional.cross_entropy(...)
```

---

# Training

`train.py` should:

1. Load `data/input.txt`
2. Create the character vocabulary
3. Encode the dataset
4. Create training and validation splits
5. Instantiate the Transformer
6. Train using AdamW
7. Periodically calculate training and validation loss
8. Print training progress
9. Save a checkpoint
10. Generate a short sample periodically or at the end

Recommended initial settings:

```python
learning_rate = 3e-4
max_steps = 5000
eval_interval = 250
eval_iters = 100
```

These are educational defaults, not fixed requirements.

Automatically use CUDA or Apple Metal (MPS) when available, otherwise CPU.

For example:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

Make training reproducible where practical by setting a random seed.

---

# Text Generation

Implement autoregressive generation.

Given initial tokens such as:

```text
"ROMEO:"
```

repeat:

```text
1. Take the most recent block_size tokens
2. Run them through the Transformer
3. Take logits for the final position
4. Convert logits to probabilities
5. Sample the next token
6. Append it to the sequence
```

Support a configurable temperature.

Optionally support top-k sampling, but keep the first implementation simple.

`generate.py` should load a saved checkpoint and generate text without retraining.

---

# Checkpoints

Save enough information to reconstruct the model:

```python
{
    "model_state_dict": ...,
    "config": ...,
    "stoi": ...,
    "itos": ...,
}
```

Save checkpoints under:

```text
checkpoints/
```

Do not silently overwrite unrelated checkpoints.

---

# Code Quality

Prioritise educational clarity.

Use descriptive class names such as:

```python
AttentionHead
MultiHeadAttention
FeedForward
TransformerBlock
ShakespeareTransformer
```

Use type hints where they improve readability.

Add concise comments explaining tensor shapes and important Transformer concepts.

Do not over-engineer the project.

Avoid abstractions that obscure how attention works.

---

# Tensor Shapes

Document important tensor shapes in the implementation.

Use these symbols consistently:

```text
B = batch size
T = sequence length
C = embedding dimension
V = vocabulary size
H = number of attention heads
```

For example:

```python
# x: (B, T, C)
# logits: (B, T, V)
```

Shape correctness is particularly important in the attention implementation.

---

# Validation

Before considering the implementation complete, verify:

1. The model runs on CPU.
2. A forward pass produces `(B, T, V)` logits.
3. Causal masking prevents future-token attention.
4. Training loss decreases on a small sample.
5. Validation loss can be calculated.
6. A checkpoint can be saved and reloaded.
7. Generated text has the expected length.
8. Generated text becomes more Shakespeare-like after training.

Where practical, add small tests for tokenizer round-tripping and model tensor shapes.

---

# README Requirements

`README.md` should explain:

- What the project demonstrates
- How a Transformer works at a high level
- How Shakespeare is tokenised
- How to install dependencies
- How to obtain/prepare the dataset
- How to train the model
- How to generate text
- Expected hardware/runtime considerations
- The major model configuration parameters

Include a simple architecture diagram such as:

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
│                        │
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

---

# Development Approach

Build incrementally.

Recommended order:

1. Dataset loading
2. Tokenizer
3. Batch generation
4. Single attention head
5. Multi-head attention
6. Feed-forward network
7. Transformer block
8. Complete language model
9. Training loop
10. Generation
11. Checkpointing
12. Documentation/tests

After each major stage, run the code and correct errors before proceeding.

Do not implement the entire project without testing intermediate components.

---

# Definition of Done

The project is complete when a user can run:

```bash
pip install -r requirements.txt
python train.py
```

and observe decreasing loss, then run:

```bash
python generate.py
```

and receive newly generated Shakespeare-like text from the trained Transformer.

The implementation should make it possible for a learner to trace the complete path:

```text
characters
→ token IDs
→ embeddings
→ attention
→ Transformer blocks
→ logits
→ probabilities
→ generated characters
```

The primary goal is **understanding the Transformer architecture**, not achieving state-of-the-art language-model performance.
