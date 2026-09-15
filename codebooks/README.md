# Codebooks

Trained transition-embedding artifacts, ready to load.

## `wikitext103_kda.pt` (11.5 MB)

The reference artifact: a full-rank, coordinate-wise-spread 512-bit codebook
trained on **wikitext-103** with a **KDA** transition function (the conv-like, short-range
inductive bias). This is the codebook whose "transition-style" character was
characterized in the project.

| | |
|---|---|
| **vocab** | 151,669 (Qwen3 tokenizer, frozen) |
| **code_dim** | 512 |
| **transition function** | KDA (single layer, hidden = code_dim) |
| **corpus** | wikitext-103-v1, 124.3M tokens |
| **recipe** | weighted CE (inverse-freq) + stratified VISReg + online unigram centering |
| **interface** | tied binary: input = signed codes, output = sign (2-bit), unigram-centered by construction |

### Characterization (what the bits encode)

| probe | result | reading |
|---|---|---|
| WordSim-353 Spearman | 0.085 (≈ random 0.041) | **not semantic** (SHADOW-2: 0.619) |
| MLP bigram (frozen emb) | wCE 7.2 (90% of the full-model gain) | **static / per-token**, not contextual |
| 5-gram conv | past 4 tokens → ~2.1 nats | **short-range** n-gram info |
| Linear probe (free W) | wCE 8.92 ≫ MLP 7.2 | the transition is **nonlinear** in the code |
| MTP reach (t+1..t+8) | gain over unigram decays smoothly, floors at t+7 | **bounded multi-step reach ~6–7 tokens** (KDA + T=256 flavor) |
| effective rank | 504 / 512 | full-rank |
| mean pairwise Hamming | 205 / 512 (random: 256) | coordinate-wise spread, not an ECC |

The *transition-style* character (not semantic) is fixed by the recipe; the
~6–7 token *reach* is a property of the KDA transition function, not of the library.

### Load

```python
from transition_embed import Codebook
cb = Codebook.from_artifact("codebooks/wikitext103_kda.pt")
codes = cb.codes()   # (151669, 512) signed, unigram-centered
```
