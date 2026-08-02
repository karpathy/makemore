# Neural Networks: Zero to Hero — worked through

My working repo for Andrej Karpathy's [**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) series — building language models from scratch in PyTorch, starting from a bigram count table and ending at a decoder-only Transformer.

The point of the series is that nothing is imported as a black box. `nn.Transformer` never appears. Attention is written out as three linear projections and a masked softmax, so you can see exactly what "attention" computes and why every tensor has the shape it does. This repo is my pass through that material: the follow-along notebook where each idea is built up one cell at a time, and the finished script it converges on.

## Layout

```
├── nanogpt-lecture/
│   ├── gpt_dev.ipynb    # step-by-step development notebook
│   ├── build_GPT.py     # the finished ~10.8M-param GPT
│   └── input.txt        # tiny Shakespeare, 1.1 MB / ~1.1M characters
├── makemore/            # makemore series, parts 1-5, one notebook each
│   ├── build_makemore_yay_bigram.ipynb
│   ├── build_makemore_yay_MLP.ipynb
│   ├── build_makemore_yay_RNN.ipynb
│   ├── build_makemore_manual_backprop.ipynb
│   ├── build_makemore_yay_Wavenet.ipynb
│   ├── makemore.py      # Karpathy's reference implementation
│   └── names.txt
└── names.txt            # 32,032 names — the makemore dataset
```

---

## Building a GPT from scratch

> Lecture: [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)

### `gpt_dev.ipynb` — the development path

The notebook is the incremental half, built in the order the ideas actually need to arrive:

1. **Get the data.** Tiny Shakespeare — 1,115,394 characters of concatenated plays.
2. **Build the vocabulary.** 65 unique characters, sorted. Character-level, so no tokenizer library: `stoi`/`itos` dicts and two lambdas are the entire encode/decode stack. The trade-off is explicit — tiny vocab, very long sequences.
3. **Encode to a tensor** and split 90/10 into train and validation. The split is positional, not random, because the data is one continuous stream.
4. **Understand the training signal.** A single block of 9 characters is really *8 separate training examples* — `[24] → 43`, `[24,43] → 58`, `[24,43,58] → 5`, and so on. The Transformer learns to predict from every prefix length at once, which is also what lets it generate from a cold start with only one token of context.
5. **Add the batch dimension.** `get_batch` samples random offsets into the stream and stacks them into `(B, T)`, with `y` being `x` shifted one position. Every model from here on speaks in `(B, T, C)` — batch, time, channels.
6. **The bigram baseline.** An `nn.Embedding(vocab_size, vocab_size)` used as a lookup table: each token reads off the logits for the next one directly, with no context at all beyond the current character.

The bigram model's untrained loss comes out at **4.8786**. The floor for a uniform guess over 65 characters is `ln(65) ≈ 4.174`, and the gap is the cost of random initialization — the model starts out worse than knowing nothing. Sampling from it produces exactly the noise you'd expect:

```
SKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp
```

That string is the baseline the rest of the lecture exists to beat.

### `build_GPT.py` — the finished model

The full decoder-only Transformer, following the lecture's reference implementation. **10,788,929 parameters** in this configuration:

| Hyperparameter | Value |
| --- | --- |
| `n_embd` | 384 |
| `n_head` | 6 (head size 64) |
| `n_layer` | 6 |
| `block_size` | 256 |
| `batch_size` | 64 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 (AdamW) |
| `max_iters` | 5,000 |

Where those parameters live:

| Component | Parameters |
| --- | --- |
| Token embedding (65 × 384) | 24,960 |
| Position embedding (256 × 384) | 98,304 |
| 6 × Transformer block | 10,639,872 |
| Final LayerNorm | 768 |
| LM head (384 → 65) | 25,025 |

The blocks are 98.6% of the model — embeddings and the output head are rounding errors. That ratio is the whole reason depth and width are the knobs that matter when scaling up.

### What each piece is doing

**Attention is a data-dependent weighted average.** Every token emits a *query* ("what am I looking for?"), a *key* ("what do I contain?"), and a *value* ("what do I contribute?"). The affinity matrix `q @ k.transpose(-2,-1)` is `(B, T, T)` — how much each position wants to hear from each other position — and after softmax it weights the values. Attention itself has no notion of order; it's set operations on vectors.

**The `* k.shape[-1]**-0.5` scaling is not cosmetic.** Dot products of two random vectors of dimension `head_size` have variance proportional to `head_size`. Without dividing by `sqrt(head_size)`, the logits going into the softmax start out wide, the softmax saturates toward one-hot, and gradients through it vanish. Dividing keeps the initial distribution diffuse so every position gets gradient early in training.

**Causality is one masked_fill.** `tril` is registered as a buffer, not a parameter — it's constant, but it needs to follow the model onto the GPU. Filling the upper triangle with `-inf` *before* the softmax makes those weights exactly zero after it, so position `t` can only attend to positions `≤ t`. Remove that one line and you have BERT-style bidirectional attention, which would leak the answer.

**Multi-head means several smaller attentions in parallel.** Six heads of size 64 concatenate back to 384. Each head is free to specialize — one on the previous character, another on matching quotes or line structure — and `self.proj` mixes them back together.

**Residual connections plus pre-LayerNorm are what make 6 layers trainable.** `x = x + self.sa(self.ln1(x))` gives gradients a clean identity path from the loss all the way back to the embeddings. Note the norm is applied *inside* the residual branch — pre-norm, which differs from the original 2017 paper and is far more stable to train.

**The feed-forward layer is where per-token computation happens.** Attention moves information between positions; the 4× expansion (384 → 1536 → 384) is the model thinking about what it gathered. Karpathy's framing — "communication followed by computation" — is the cleanest way to hold the block in your head.

**Position embeddings are needed precisely because attention is order-blind.** A learned `nn.Embedding(block_size, n_embd)` added to the token embeddings supplies the ordering. It also hard-caps the context: nothing beyond `block_size` has a position vector, which is why `generate` crops with `idx[:, -block_size:]` on every step.

**`estimate_loss` averages over 200 batches under `@torch.no_grad()`**, with `model.eval()` / `model.train()` around it so dropout is off while measuring. A single batch's loss is far too noisy to tell whether training is working.

## Running it

Only PyTorch is needed.

```bash
cd nanogpt-lecture
python build_GPT.py
```

It prints train and validation loss every 500 iterations and samples 500 characters at the end. Loss starts near `ln(65) ≈ 4.17` and should fall well below 2. This configuration wants a GPU — `device` is set automatically, but on CPU the 5,000 iterations at `block_size=256` will take a very long time. Drop `n_layer`, `n_embd` and `block_size` for a CPU-sized run.

---

## The makemore series

> Lectures: [makemore part 1](https://www.youtube.com/watch?v=PaCmpygFfXo) through part 5

`names.txt` — 32,032 of the most common US baby names — is the dataset for the earlier half of the series, which arrives at the Transformer from the other direction. Each notebook in [`makemore/`](makemore/) is one lecture, and every one of them ends at roughly the same loss:

| Notebook | Lecture | What it builds | Result |
| --- | --- | --- | --- |
| `build_makemore_yay_bigram.ipynb` | Part 1 | Bigram counts, then the *same* model re-derived as a one-hot input into a single linear layer trained by gradient descent | loss **2.4622** |
| `build_makemore_yay_MLP.ipynb` | Part 2 | Bengio et al. 2003 MLP — character embeddings, a 3-character context window, minibatching, learning-rate search, and a proper 80/10/10 train/dev/test split | |
| `build_makemore_yay_RNN.ipynb` | Part 3 | Activations and gradients: hand-rolled `Linear`, `Tanh` and `BatchNorm1d` modules, watching saturation and initialization scale decide whether the net trains at all | train **2.4342** / val **2.4390** |
| `build_makemore_manual_backprop.ipynb` | Part 4 | Backprop by hand through the entire MLP + BatchNorm — 29 `cmp()` gradient checks against autograd, no `loss.backward()` | |
| `build_makemore_yay_Wavenet.ipynb` | Part 5 | A WaveNet-style hierarchical model: `FlattenConsecutive` fusing characters in pairs across layers, wrapped in a homemade `Sequential` | train **2.4243** / val **2.4378** |

That flat loss curve across five increasingly sophisticated models is the actual lesson of the series. A count-based bigram table gets **2.4622**; a hierarchical convolutional network with BatchNorm gets **2.4243**. The architecture barely moved the number, because a 3-character context is the binding constraint — which is exactly the problem attention was invented to solve, and why the series ends at a Transformer.

Part 1 re-deriving the count table as a neural net is the pivot the whole series rests on: once you've seen that the trained weights converge to the log-counts, "training a network" stops being magic and becomes something you can predict the answer to.

The folder also carries Karpathy's `makemore.py` and `names.txt` from [karpathy/makemore](https://github.com/karpathy/makemore) for reference, plus an empty `build_GPT.ipynb` placeholder.

---

## Notes

* `build_GPT.py` follows the lecture's reference implementation closely — this is a learning repo, and the value is in the annotations above rather than in any claim to a novel architecture.
* `gpt_dev.ipynb` currently stops at the bigram model. The self-attention derivation from the second half of the lecture — the running-average trick and the four progressively better versions of it — goes here next, and it's the most instructive part of the whole video.
* The last line of the notebook's bigram cell won't run as written (unbalanced parentheses, and `torch.zeros(idx, ...)` where `idx` was meant to be passed straight to `generate`). The saved output is from an earlier working version of the cell.

## Credits

All lecture material by [Andrej Karpathy](https://github.com/karpathy) — [Zero to Hero](https://karpathy.ai/zero-to-hero.html), [nanoGPT](https://github.com/karpathy/nanoGPT), [makemore](https://github.com/karpathy/makemore). Tiny Shakespeare comes from [char-rnn](https://github.com/karpathy/char-rnn).
