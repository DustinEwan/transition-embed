# transition-embed

Train a **fixed, reusable, tied binary codebook** whose bits encode short-range
next-token **transition** structure — not similarity.

"Tied": one learned `(V, D)` table serves both the input (signed codes) and
the output (signed, 2-bit interface). The library is **transition-function-
agnostic**: you provide any `nn.Module` mapping `(B, T, D) -> (B, T, D)`; the
library trains the tied binary embedding with a fixed recipe (weighted CE +
stratified VISReg + online unigram centering) and exports the two **
foundational** artifacts in one file — "foundational" meaning the base stage
that downstream tasks build on: the compact bitpacked transition embedding
(1 bit per code) and the trained transition function (state_dict).

```python
from transition_embed import train, Codebook

# any (B,T,D) -> (B,T,D) module works as the transition function
train(transition_fn, data_iter, vocab_size=151669, code_dim=512, out="codebook.pt")

cb = Codebook.from_artifact("codebook.pt")
codes = cb.codes()                    # (V, D) signed, unigram-centered (the 2-bit interface)
h = cb.transition_encoding(transition_fn)  # (V, D) marginal transition encoding
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

This is the core of the library, and it generalizes across transition functions. Three
things, in order, make the codes transition-style rather than semantic-style:

1. **The objective is next-token prediction, not similarity.** The codebook is
   trained to minimize (weighted) next-token cross-entropy through a transition function.
   The gradient signal is *"which bits help predict the next token."* So the bits
   are shaped to carry **predictive / transition** information, not **similarity**
   information. Semantic embeddings (word2vec, sentence embeddings) are
   trained with a similarity / co-occurrence objective, so they encode
   similarity. Same table, different objective, different structure.

2. **The transition function provides a short-range inductive bias.** You
   provide a short-range, causal operator (a conv-like transition function). The
   bias is the *cause*, the structure is the *effect*: a short-range operator
   forces the codes to carry **bounded, local transition structure**. Crucially,
   the *reach* — how many tokens the structure extends over — is a property of
   **the transition function you provide**, not of this library. A different
   transition function gives a different reach. (This is why we do not quote a
   specific token count: the reach is transition-function-dependent, not a fixed
   constant of the recipe.)

3. **The tied binary interface + anti-collapse regularization.** The same `(V, D)`
   table serves both the input (signed) and the output (signed). VISReg keeps *all*
   dimensions active (anti-collapse, not packing), so the codes fill the
   hypercube **coordinate-wise**, not in clusters. The result is a full-rank,
   coordinate-wise-spread code — organized by *transition*, not by *similarity*.

**The contrast, measured:**

| | objective | WordSim-353 Spearman | structure |
|---|---|---|---|
| **Semantic** (word2vec, sentence embeddings) | similarity / co-occurrence | ~0.6 | near = related |
| **Transition** (this library) | next-token prediction, short-range transition function | ~0.08 (≈ random) | bits predict *next* |

The "meaning" of the bits is **task-relative**: they are meaningful for
*predicting the next token over a short window*, not for *measuring similarity*.
That is not a defect — it is the point. A semantic clustering is one way to encode
meaningful information; a bounded transition structure is another.

---

## The recipe

- **Tied binary interface.** One learned `(V, D)` table. Input = signed codes
  (`sign(raw) - c + b_in`); output = `sign(raw)`, unigram-centered *by construction* (the unigram-weighted
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
train(transition_fn, data, vocab_size, code_dim, out,
      lr=1e-3, warmup=100, lam_v=0.1, tau=1e-4, gamma=0.5,
      n_slices=64, subsample=2048, cuts=(0.01, 0.10, 0.50),
      ce_chunk=16384, log_every=100, save_every=500_000,
      max_tokens=None, resume=None)
```

- `transition_fn`: `nn.Module` mapping `(B,T,D) -> (B,T,D)`; receives the
  **signed** codes (`sign(raw) - c + b_in`), matching the trained model.
- `data`: iterator of `(B, T)` blocks (token ids); done by `StopIteration`.
- Multi-GPU: run under `torchrun --nproc_per_node=N`; the count is all-reduced
  per step, grads via DDP, and a done-handshake keeps the ranks in lockstep.

```python
cb = Codebook.from_artifact("codebook.pt")
cb.codes()      # (V, D) signed, unigram-centered
cb.signed()     # (V, D) raw sign
cb.vocab_size, cb.code_dim
cb.has_transition_fn           # True for foundational-stage artifacts
h = cb.transition_encoding(arch)  # (V, D) marginal transition encoding
```

The artifact is the portable contract: bit-packed sign + centering vector +
per-token scalars + global scalars + the trained transition function. See
`ARTIFACT.md` for the bit layout.

---

## Downstream: from the codebook to transition families

The two artifacts are a **general learned transition structure**. The most
useful thing you can compute from them is the **marginal transition encoding**:
for each token, what the transition function says about *where that token goes
next* with no context. (The full transition function is a sequence operator
over `(B, T, D)`; the encoding uses its `T=1`, no-context slice.)

```python
h = cb.transition_encoding(arch)   # (V, D) — one "next-token behavior" signature per token
```

Tokens whose signatures are close **behave the same**: given the token, they
lead to (nearly) the same next-token distribution. That is a *behavioral*
equivalence — a forward property (*where the token goes*), not a semantic one
(*what the token means*). It is intrinsic to the transition encoding; a
semantic embedding (a *backward* property: what tends to come *before* the
token) does not carry it.

### Coarsening: V → V′ (transition families)

An **n-gram table** — as in Engram, DeepSeek's context module: a large,
CPU-resident lookup that maps recent token n-grams to vectors — is indexed by
raw token IDs. But many tokens behave identically, so a row per token ID
wastes space and splits behavior that should be shared. **Coarsening** fixes
that: map each token to one of K **transition families** (behavioral
equivalence classes), and let tokens in the same family share a table row. The
table's keys are re-expressed through the learned mapping — a change of basis
that requires no hand-rolled rules.

```python
fam, C = cb.discover_families(arch, K)   # K-means over the encoding
# fam: (V,) int — token id -> family id (the Dict[V, V'])
# C:   (K, D) — the family centroids
```

K-means over the encoding is the practical, rule-free partition. Its primary
value is **discovery**: it finds which tokens lead to the same next-token
distribution — the overlaps that hand-rolled rules (casefolding, whitespace,
inflectional endings) find in English, and the ones hand-rolled rules *cannot*
find, e.g. in agglutinative languages (Korean, Japanese) where the
normalization rules do not exist. Compressing an n-gram table's keys is one
consequence; tokenizer design and morphology discovery are others.

### Persisting and using the mapping

```python
save_families("families.json", fam, C, K)   # or discover_families(..., out=...)
fm = FamilyMap("families.json")             # static integer lookup, lives on the CPU
fam_ids = fm.map(token_ids)                # beside the (CPU) n-gram table
fm.to("cuda")                              # for training, beside the model
```

The mapping is the deployed artifact: V integers (~1.2 MB at V=151,669), one
gather, no model at runtime. The library stays general; a specific consumer
(e.g. an Engram-style n-gram table) is a downstream task, not part of the core.

---

## Examples

- `examples/train_wikitext.py` — hello-world (small MLP transition function,
  wikitext-103, Qwen3 tokenizer). No external transition-function dependency.
- `examples/conv_transition_fn.py` — a self-contained **conv-like (short-range)**
  transition function (causal depthwise convs + mixing MLP). Shows the
  inductive bias that shapes the codes into transition-style embeddings.
- `examples/discover_families.py` — level-1 transition family discovery on a
  trained artifact (self-contained demo without one).

Both are a few lines of torch; swap in any transition function you like.

---

## Codebooks

`codebooks/` holds trained, ready-to-load artifacts. `codebooks/wikitext103_kda.pt`
is the reference 512-bit codebook (wikitext-103, KDA transition function).
See `codebooks/README.md` for its characterization metrics.

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
