# Following characters through the Transformer

The current code implements characters → token IDs → embeddings → Transformer
block outputs. The final language model, loss, training loop, checkpointing and
generation are still pending. Those later stages are explained below so the
whole design is visible, but the debug script stops at the implemented boundary.

## Run and pause the example

From the project directory:

```bash
.venv/bin/python debug_walkthrough.py
.venv/bin/python debug_walkthrough.py --step
```

The second command activates Python breakpoints after each of the 13 implemented
stages. At each `(Pdb)` prompt, type `up` once to select the `main()` frame, then
inspect its variables. For example:

```text
up
p tokenizer.stoi
p example_ids
continue
```

At later pauses, use `p x.shape`, `p embedded[0, 0]`, `p masked_scores[0]`,
`p weights[0]`, or `p output.shape` as appropriate. `continue` advances to the
next pause; `quit` exits. Later-stage variables only exist after their computation.
The debugger initially stops inside `pause()`, which is why `up` is needed.

For VS Code, enable Microsoft's Python and Python Debugger extensions. Open
`debug_walkthrough.py` and click the gutter beside the desired `pause(...)`
calls to set IDE breakpoints. In Run and Debug select **Transformer walkthrough**
and press F5. The supplied `.vscode/launch.json` uses the project's `.venv`
interpreter and runs without `--step`. Each marked call is after the computation
you want to inspect. Expand tensors in Variables or add expressions from the
table below to Watch. Use Step Over (F10) to advance, Continue (F5) to reach the
next marked call, and Step Into (F11) to enter a model method.
See the [VS Code Python debugging guide](https://code.visualstudio.com/docs/python/debugging).

The script provides executable Python breakpoints. VS Code gutter breakpoints
are set in your local IDE using the marked calls; they are not preconfigured here.
No extra packages, corpus download, training or checkpoint writes are needed.

## The worked example

We repeat `"To be or "` twenty times to create a small corpus. Vocabulary IDs are
assigned in sorted character order:

```text
character: space  T  b  e  o  r
ID:           0  1  2  3  4  5

"To be or" → [1, 4, 0, 2, 3, 0, 4, 5]
```

The script uses CPU, seed 1337, one sequence per batch, eight positions, eight
embedding features and two heads. Thus B=1, T=8, C=8, V=6, H=2 and head_size=4.
The batch starts at a random reproducible offset, so its text can differ from
`"To be or"`. Dropout is disabled by `.eval()` and no gradients are recorded.
Parameters remain randomly initialised: their numbers have no learned language
meaning yet. Repetition is for inspection, not a realistic training dataset.

## Implemented stages and breakpoint watches

| Pause | Operation and example | Useful watch |
| --- | --- | --- |
| 1 | Build `stoi`/`itos`; turn each character into its ID and decode it again. | `tokenizer.stoi`, `example_ids` |
| 2 | Encode the corpus; split the first 90% for training and the last 10% for validation. Sample shifted batches: an input `"To be or"` would target `"o be or "`. Both have shape `(B,T)`. | `train_tokens.shape`, `validation_tokens.shape`, `x`, `y`, `input_text`, `target_text` |
| 3 | Look up eight features for each character and add eight features for its position. Illustratively `[0.2, -0.1] + [0.3, 0.4] = [0.5, 0.3]`; actual vectors have eight entries. | `token_vectors[0,0]`, `position_vectors[0]`, `embedded.shape` |
| 4 | LayerNorm normalises features separately within each position, with learned scale and bias. It never combines different positions. | `normalised[0,0]`, `normalised.mean(-1)` |
| 5 | Three learned projections create Query, Key and Value vectors. Q/K determine relevance; V supplies the information to mix. Each head uses its own weights. | `query.shape`, `key[0,0]`, `value[0,0]` |
| 6 | Compare every Query to every Key: `Q @ K.transpose(-2,-1)`. Divide by `sqrt(head_size)=2`; a raw score 6 becomes 3. Rows are query positions, columns are key positions. | `scores[0]`, `scaled_scores[0]` |
| 7 | Replace future scores with `-inf`. Row 0 can see only column 0; row 2 can see columns 0, 1 and 2. | `masked_scores[0]` |
| 8 | Softmax each row into weights. `[0,0,-inf]` becomes `[0.5,0.5,0]`. Training dropout randomly zeros weights and rescales survivors; evaluation leaves them unchanged. | `weights[0]`, `weights.sum(-1)`, `dropped_weights` |
| 9 | Multiply weights by Values. Weights `[0.25,0.75]` and Values `[2,4]`, `[10,8]` give `[8,7]`. Position 0's output equals its own Value because it sees only itself. | `head_output[0,0]`, `value[0,0]` |
| 10 | Run two heads, each producing `(1,8,4)`. Concatenate to `(1,8,8)`, then project to mix head features within each position. | `head_outputs`, `combined.shape`, `projected[0,0]` |
| 11 | Add attention's output to the original embeddings, then normalise this updated residual stream for the MLP. For example `2 + 0.3 = 2.3`. | `embedded[0,0]`, `attention_output[0,0]`, `residual[0,0]`, `normalised_residual[0,0]` |
| 12 | The position-wise MLP expands 8 features to 32, applies GELU, and contracts to 8. The same network processes each position independently. | `expanded.shape`, `activated[0,0]`, `contracted.shape` |
| 13 | Add the MLP output to the residual stream. The script compares this staged calculation with `block(embedded)` and verifies that changing positions 4 onward leaves positions 0–3 unchanged. | `output.shape`, `output[0,0]`, `changed_output[:,:4]` |

Small numerical examples in this table illustrate the operations; they are not
claimed to be the randomly initialised tensors printed by the script. Its
assertions check the actual tensors.

## Remaining stages of the complete language model

**Stack blocks.** Feed the `(B,T,C)` output into the next Transformer block.
Each block refines the features while preserving shape and causal visibility.
The default configuration calls for four blocks; this script inspects one.

**Final LayerNorm and vocabulary head.** Normalise the final features and apply
`Linear(C,V)`. With six characters this produces `(1,8,6)` logits: six scores at
each position. A logit is a score, not a probability or a token ID.

**Next-character loss.** Compare each position's logits with its shifted target.
For an input position containing `T`, the target in `"To"` is `o`. Cross-entropy
penalises low probability for that target: probability 0.5 gives loss about
0.693, whereas 0.1 gives about 2.303. Flatten `(B,T,V)` to `(B*T,V)` and targets
to `(B*T)` when calculating loss. Targets may use the next character even though
attention cannot see it; that prediction is exactly what the model learns.

**Backpropagation and AdamW.** Clear old gradients, calculate loss, call
`loss.backward()`, and let the optimiser update the learned weights. Repeated
training should reduce average loss. During validation use evaluation mode and
disable gradient recording; do not update weights using validation data.
Evaluation mode and disabling gradients are separate operations.

**Checkpointing.** Save weights, model configuration and vocabulary mappings.
Reload them to reconstruct the same trained model and character IDs.

**Autoregressive generation.** Encode a prompt such as `"ROMEO:"`, take its most
recent `block_size` characters, and calculate the final position's logits.
Divide by temperature, apply softmax, sample one character and append it.
Repeat for the requested number of new characters. Lower positive temperature
sharpens probabilities; higher temperature spreads them out. The full Shakespeare
vocabulary is needed for this prompt; the tiny demo vocabulary cannot encode it.

Once implemented, useful extra breakpoints will be after vocabulary logits,
after loss calculation, after `backward()`, after the optimiser update, and after
sampling each new character. TensorBoard can then show loss and text over many
steps, while the debugger shows a single calculation in detail.
