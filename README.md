# transition-embed

Train a **fixed, reusable, tied binary codebook** whose bits encode short-range
next-token **transition** structure — not similarity.

One learned `(V, D)` table serves both the input (raw values) and the output
(signed, 2-bit interface). The library is **backbone-agnostic**: you provide any
`nn.Module` mapping `(B, T, D) -> (B, T, D)`; the library trains the tied binary
embedding with a fixed recipe (weighted CE + stratified VISReg + online unigram
centering) and exports a compact, portable artifact (1 bit per code).

```python
from transition_embed import train, Codebook

# any (B,T,D) -> (B,T,D) module works as the backbone
train(backbone, data_iter, vocab_size=151669, code_dim=512, out="codebook.pt")

cb = Codebook.from_artifact("codebook.pt")
codes = cb.codes()          # (V, D) signed, unigram-centered (the 2-bit interface)
```

---

## Why "Transition Embeddings"

The name names the *distinctive* property of the learned codes. Every embedding
does the "address" part (token id -> vector). What makes these different is what
the bits *carry*: they encode **transition structure** — *where a token goes
next, over a short window* — not **similarity structure** — *what a token is
like*.

The codes are the input to a **transition kernel**: a (nonlinear) map from the
code to a distribution over the next token. That is the precise sense of the
name. (We say "kernel" in the body for the Markov-kernel meaning; we keep the
*name* "Transition Embeddings" because "kernel" in ML strongly evokes kernel
methods / feature maps, a different concept.)

A true first-order Markov kernel is memoryless; ours has a bounded, short-range
reach, so it is a *higher-order* (context-dependent) transition kernel — the code
bundles the short window so the map over the code is Markov-like, while the
underlying process is not first-order.

---

## Why transition-style, not semantic-style

This is the core of the library, and it generalizes across backbones. Three
things, in order, make the codes transition-style rather than semantic-style:

1. **The objective is next-token prediction, not similarity.** The codebook is
   trained to minimize (weighted) next-token cross-entropy through a backbone.
   The gradient signal is *"which bits help predict the next token."* So the bits
   are shaped to carry **predictive / transition** information, not **similarity**
   information. Semantic embeddings (word2vec, sentence embeddings, SHADOW-2) are
   trained with a similarity / co-occurrence objective, so they encode
   similarity. Same table, different objective, different structure.

2. **The backbone provides a short-range inductive bias.** You provide a
   short-range, causal operator (a conv-like backbone). The bias is the *cause*,
   the structure is the *effect*: a short-range operator forces the codes to carry
   **bounded, local transition structure**. Crucially, the *reach* — how many
   tokens the structure extends over — is a property of **the backbone you
   provide**, not of this library. A different backbone gives a different reach.
   (This is why we do not quote a specific token count: the reach is
   backbone-dependent, not a fixed constant of the recipe.)

3. **The tied binary interface + anti-collapse regularization.** The same `(V, D)`
   table serves both the input (raw) and the output (signed). VISReg keeps *all*
   dimensions active (anti-collapse, not packing), so the codes fill the
   hypercube **coordinate-wise**, not in clusters. The result is a full-rank,
   coordinate-wise-spread code — organized by *transition*, not by *similarity*.

**The contrast, measured:**

| | objective | WordSim-353 Spearman | structure |
|---|---|---|---|
| **Semantic** (word2vec, SHADOW-2) | similarity / co-occurrence | ~0.6 | near = related |
| **Transition** (this library) | next-token prediction, short-range backbone | ~0.08 (≈ random) | bits predict *next* |

The "meaning" of the bits is **task-relative**: they are meaningful for
*predicting the next token over a short window*, not for *measuring similarity*.
That is not a defect — it is the point. A semantic clustering is one way to encode
meaningful information; a bounded transition structure is another.

---

## The recipe

- **Tied binary interface.** One learned `(V, D)` table. Input = raw values.
  Output = `sign(raw)`, unigram-centered *by construction* (the unigram-weighted
  mean of the codes is exactly zero, so the model cannot use the code dimensions
  to predict token frequency; frequency is delegated to the output bias). Gradients
  reach the raw values through a **sign-STE** with a tanh surrogate
  (`grad * (1 - tanh(x)^2)`).
- **Weighted CE.** Inverse-frequency weighting (`w = (τ/max(τ, p))^γ`) for the
  long tail.
- **Stratified VISReg.** Anti-collapse (all dimensions active), applied to the raw
  values, per frequency stratum (cumulative-mass boundaries). It is *anti-collapse*,
  not *packing*: it forces every dimension on, so the codes fill the hypercube
  coordinate-wise. It does **not** maximize pairwise distance (the codes are not an
  error-correcting code).
- **Online unigram centering.** Naive running count, single pass, no prepass.
  Unseen token -> `p = 0` -> tail stratum, max weight. Strata are rebuilt from the
  running count every step (cheap; boundaries are stable once the head is counted).

---

## API

```python
train(backbone, data, vocab_size, code_dim, out,
      lr=1e-3, warmup=100, lam_v=0.1, tau=1e-4, gamma=0.5,
      n_slices=64, subsample=2048, cuts=(0.01, 0.10, 0.50),
      ce_chunk=16384, log_every=100, save_every=500_000,
      max_tokens=None, resume=None)
```

- `backbone`: `nn.Module` mapping `(B,T,D) -> (B,T,D)`; receives the **raw**
  embedding (signing is only for the unembed interface).
- `data`: iterator of `(B, T)` blocks (token ids); done by `StopIteration`.
- Multi-GPU: run under `torchrun --nproc_per_node=N`; the count is all-reduced
  per step, grads via DDP, and a done-handshake keeps the ranks in lockstep.

```python
cb = Codebook.from_artifact("codebook.pt")
cb.codes()      # (V, D) signed, unigram-centered
cb.signed()     # (V, D) raw sign
cb.vocab_size, cb.code_dim
```

The artifact is the portable contract: bit-packed sign + centering vector +
per-token scalars + global scalars. See `ARTIFACT.md` for the bit layout.

---

## Examples

- `examples/train_wikitext.py` — hello-world (small MLP backbone, wikitext-103,
  Qwen3 tokenizer). No external backbone dependency.
- `examples/conv_backbone.py` — a self-contained **conv-like (short-range)**
  backbone (causal depthwise convs + mixing MLP). Shows the inductive bias that
  shapes the codes into transition-style embeddings.

Both are a few lines of torch; swap in any backbone you like.

---

## Codebooks

`codebooks/` holds trained, ready-to-load artifacts. `codebooks/wikitext103_kda.pt`
is the reference 512-bit codebook (wikitext-103, KDA backbone) whose transition-style
character was characterized in the project. See `codebooks/README.md` for provenance
and metrics.

```python
from transition_embed import Codebook
cb = Codebook.from_artifact("codebooks/wikitext103_kda.pt")
codes = cb.codes()   # (151669, 512) signed, unigram-centered
```

---

## Install

```bash
uv pip install -e .            # core (torch only)
uv pip install -e ".[examples]"  # + the examples (transformers, datasets, fire)
```
