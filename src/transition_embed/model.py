"""
Transition-embedding model: binary tied embedding -> transition function -> tied
unembedding.

    idx -> embed(idx)             (signed (B,T,D): sign(raw) - c + b_in)
        -> transition_fn(x)       (B,T,D)   [user-provided, receives the SIGNED codes]
        -> alpha * h @ sign^T + b_out       (tied unembed, 2-bit interface)

The transition function is a contract: any nn.Module mapping (B,T,D) -> (B,T,D).
KDA (quanta) is one example; the library is transition-function-agnostic. The
"tie" is that the input interface and the output projection are the SAME signed
table — one learned (V,D) table serves both. The raw values are the gradient
carrier; the sign is the interface, used for both the transition-function input
and the unembed.

The transition function receives the SIGNED interface (the portable 1-bit-per-dim
codes), matching the trained model. This is what makes the two foundational
artifacts (the bitpacked embedding + the transition function) self-consistent:
feed the function the signed codes from artifact 1, get the h it was trained on.
"""
import torch
import torch.nn as nn

from .embedding import BinaryTiedEmbedding


class CodebookModel(nn.Module):
    """Binary tied embedding + user transition function + tied unembedding.

    Args:
        vocab_size: V
        code_dim: D (e.g. 512)
        transition_fn: nn.Module mapping (B,T,D) -> (B,T,D); receives the SIGNED
            codes (sign(raw) - c + b_in), matching the trained model
        unigram_p: (V,) corpus unigram probabilities (buffer, stop-grad); None -> zeros
    """

    def __init__(self, vocab_size: int, code_dim: int, transition_fn: nn.Module,
                 unigram_p: torch.Tensor | None = None):
        super().__init__()
        if unigram_p is None:
            unigram_p = torch.zeros(vocab_size)
        self.embedding = BinaryTiedEmbedding(vocab_size, code_dim, unigram_p)
        self.transition_fn = transition_fn

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) ids -> (B, T, code_dim) hidden states. The transition function
        receives the SIGNED interface (sign(raw) - c + b_in), matching the trained
        model (the raw values are the gradient carrier; the sign is the interface)."""
        x = self.embedding.embed(idx)  # signed (B, T, D)
        return self.transition_fn(x)   # (B, T, D)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Full (B, T, V) logits — for inference/export only (1 token: trivial)."""
        return self.embedding.unembed(self.forward(idx))


def kmeans(X: torch.Tensor, K: int, iters: int = 10, chunk: int = 4096,
           seed: int = 0):
    """GPU K-means on a (V, D) tensor. Lloyd's algorithm; the assignment step
    (the (V, K) distance matrix) is chunked over V so only a (chunk, K) block is
    materialized. Returns (assign (V,) int64, centroids (K, D)). K >= V ->
    identity (every row is its own cluster)."""
    Vn, Dn = X.shape
    if K >= Vn:
        return torch.arange(Vn, device=X.device), X.clone()
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(Vn, generator=g, device=X.device)[:K]].clone()
    x2 = (X ** 2).sum(1)
    assign = torch.empty(Vn, dtype=torch.long, device=X.device)
    for _ in range(iters):
        c2 = (C ** 2).sum(1)
        for s in range(0, Vn, chunk):
            e = min(s + chunk, Vn)
            d = x2[s:e][:, None] + c2[None, :] - 2.0 * (X[s:e] @ C.T)
            assign[s:e] = d.argmin(1)
        Cnew = torch.zeros_like(C)
        Cnew.index_add_(0, assign, X)
        cnt = torch.bincount(assign, minlength=K).to(C.dtype)
        empty = (cnt == 0).nonzero().flatten()
        if empty.numel():
            Cnew[empty] = X[torch.randint(0, Vn, (empty.numel(),),
                                         device=X.device, generator=g)]
            cnt[empty] = 1.0
        C = Cnew / cnt[:, None]
    return assign, C


class Codebook:
    """A trained transition-embedding artifact: the bitpacked tied table (the
    transition embedding) + the transition-function weights.

    The two artifacts are self-consistent: the transition function was trained on
    the signed codes, and the artifact carries the signed codes. Feed the function
    the signed codes (via transition_encoding) to get the h it was trained on —
    the substrate for downstream tasks (e.g. the V->V' coarsening).
    """

    def __init__(self, bits, c, in_bias, out_bias, alpha, unigram_p,
                 transition_fn_weights=None):
        self.bits = bits            # (V, D//8) uint8
        self.c = c                 # (D,) centering vector
        self.in_bias = in_bias     # (V,)
        self.out_bias = out_bias   # (V,)
        self.alpha = alpha         # scalar (log-temperature, exponentiated)
        self.unigram_p = unigram_p  # (V,)
        self.transition_fn_weights = transition_fn_weights  # state_dict or None

    @classmethod
    def from_artifact(cls, path):
        a = torch.load(path, map_location="cpu")
        return cls(a["bits"], a["c"], a["in_bias"], a["out_bias"],
                   a["alpha"], a["unigram_p"], a.get("transition_fn"))

    @property
    def vocab_size(self):
        return self.bits.shape[0]

    @property
    def code_dim(self):
        return self.bits.shape[1] * 8

    @property
    def has_transition_fn(self):
        return self.transition_fn_weights is not None

    def signed(self, device=None, dtype=torch.float32):
        """(V, D) raw signed codes (before centering)."""
        return (BinaryTiedEmbedding.unpack_bits(self.bits)
                .to(device, dtype=dtype) * 2.0 - 1.0)

    def codes(self, device=None, dtype=torch.float32):
        """(V, D) signed codes, unigram-centered (the 2-bit interface)."""
        return self.signed(device, dtype) - self.c.to(device, dtype=dtype)

    def transition_fn(self, arch, device=None):
        """Load the saved transition-function weights into a user-provided
        instance. `arch` must be an nn.Module with the same architecture as the
        one used in the foundational stage (matching param shapes). Returns it in
        eval mode on `device`."""
        if self.transition_fn_weights is None:
            raise ValueError("artifact has no transition-function weights "
                             "(pre-rename artifact); retrain with the current recipe")
        if device is not None:
            arch = arch.to(device)
        arch.load_state_dict(self.transition_fn_weights)
        arch.eval()
        return arch

    def transition_encoding(self, arch, chunk=4096, device=None):
        """(V, D) marginal transition encoding: the transition function applied to
        each token's signed code (T=1, no context), chunked (a sequence
        transition function like KDA crashes on B=V). `arch` is an instance of the
        foundational architecture. This is the h you K-means for the V->V'
        coarsening."""
        tfn = self.transition_fn(arch, device=device)
        dev = next(tfn.parameters()).device
        codes = self.codes(device=dev)  # (V, D)
        ib = self.in_bias.to(dev)
        b = ib - (ib * self.unigram_p.to(dev)).sum()  # centered input bias (V,)
        x = codes + b[:, None]  # (V, D) the transition function's input
        V = codes.shape[0]
        h = torch.empty(V, codes.shape[1], device=dev)
        with torch.no_grad():
            for s in range(0, V, chunk):
                e = min(s + chunk, V)
                h[s:e] = tfn(x[s:e].unsqueeze(1)).squeeze(1)
        return h

    def discover_families(self, arch, K: int, iters: int = 10,
                         chunk: int = 4096, seed: int = 0, device=None,
                         out: str | None = None):
        """Level-1 transition family discovery: K-means on the marginal
        transition encoding. Returns (families (V,) int64, centroids (K, D)) —
        `families[i]` is the transition family (behavioral equivalence class)
        token i belongs to; it is the Dict[V, V'] coarsening (e.g. for Engram
        token normalization). `arch` is an instance of the foundational
        transition-function architecture. `out` persists the result
        (save_families)."""
        h = self.transition_encoding(arch, chunk=chunk, device=device)
        fam, C = kmeans(h, K, iters=iters, chunk=chunk, seed=seed)
        if out is not None:
            save_families(out, fam, C, K, seed=seed, iters=iters)
        return fam, C

    def save(self, path):
        d = {"bits": self.bits, "c": self.c, "in_bias": self.in_bias,
             "out_bias": self.out_bias, "alpha": self.alpha,
             "unigram_p": self.unigram_p}
        if self.transition_fn_weights is not None:
            d["transition_fn"] = self.transition_fn_weights
        torch.save(d, path)


def save_families(path: str, families: torch.Tensor, centroids: torch.Tensor,
                  K: int, seed: int = 0, iters: int = 10):
    """Persist a family discovery. `.pt` (default): the Dict[V, V'] mapping
    (~1.2 MB at V=151,669) + the (K, D) centroids + the run params. `.json`:
    the mapping + run params as a plain object (no torch needed to read it;
    the mapping is just V integers — load into a tensor if you want it on
    GPU). Centroids are pt-only."""
    if path.endswith(".json"):
        import json
        with open(path, "w") as f:
            json.dump({"K": K, "seed": seed, "iters": iters,
                       "families": families.detach().cpu().tolist()}, f)
        return
    torch.save({"families": families.detach().cpu(),
                "centroids": centroids.detach().cpu(),
                "K": K, "seed": seed, "iters": iters}, path)


def load_families(path: str):
    """Load a save_families file. Returns (families (V,), centroids (K, D) or
    None for .json, meta)."""
    if path.endswith(".json"):
        import json
        with open(path) as f:
            d = json.load(f)
        return torch.tensor(d["families"], dtype=torch.long), None, d
    d = torch.load(path)
    return d["families"], d["centroids"], d


class FamilyMap:
    """A loaded Dict[V, V'] coarsening (from a save_families file). The
    deployed artifact: a static integer lookup, zero model at runtime.
    Lives on the CPU next to the (CPU) n-gram table: `map(ids)` gathers on
    the mapping's device (moving the key if needed — a few ints), so the
    family ids stay beside the table for the hash + lookup. For Engram:
    key GPU->CPU, map + hash + table read on CPU, n-gram vector CPU->GPU."""

    def __init__(self, path=None, families=None, K=None):
        if path is not None:
            if path.endswith(".json"):
                import json
                with open(path) as f:
                    d = json.load(f)
                families, K = d["families"], d.get("K")
            else:
                d = torch.load(path)
                families, K = d["families"].tolist(), d.get("K")
        if families is None:
            raise ValueError("path or families required")
        self.families = torch.as_tensor(families, dtype=torch.long)
        self.K = K if K is not None else int(self.families.max().item() + 1)

    def map(self, ids):
        return self.families[ids.to(self.families.device)]

    def to(self, device):
        """Move the mapping (e.g. to the GPU for training)."""
        self.families = self.families.to(device)
        return self

    def __call__(self, ids):
        return self.map(ids)


def _self_check():
    torch.manual_seed(0)
    V, D = 1024, 64
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition_fn = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D))
    m = CodebookModel(V, code_dim=D, transition_fn=transition_fn).to(dev)
    idx = torch.randint(0, V, (2, 64), device=dev)
    h = m(idx)
    assert h.shape == (2, 64, D), f"shape {h.shape}"
    logits = m.logits(idx)
    assert logits.shape == (2, 64, V), f"shape {logits.shape}"
    logits.sum().backward()
    assert m.embedding.emb.weight.grad is not None, "no grad to raw values"
    assert m.transition_fn[0].weight.grad is not None, "no grad to transition fn"
    # round-trip through the artifact (with the transition-function weights)
    with torch.no_grad():
        s = torch.sign(m.embedding.emb.weight)
        p = torch.softmax(torch.randn(V, device=dev), dim=0)
        c = (s * p[:, None]).sum(0)
        torch.save({"bits": BinaryTiedEmbedding.pack_bits(s > 0), "c": c,
                    "in_bias": m.embedding.in_bias, "out_bias": m.embedding.out_bias,
                    "alpha": m.embedding.log_alpha.exp(), "unigram_p": p,
                    "transition_fn": m.transition_fn.state_dict()},
                   "/tmp/_te_selfcheck.pt")
    cb = Codebook.from_artifact("/tmp/_te_selfcheck.pt")
    assert cb.vocab_size == V and cb.code_dim == D
    assert cb.codes().shape == (V, D)
    assert cb.has_transition_fn, "transition-fn weights not saved"
    # the marginal transition encoding runs (chunked) and has the right shape
    h = cb.transition_encoding(nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D)))
    assert h.shape == (V, D), f"transition encoding shape {h.shape}"
    assert torch.isfinite(h).all(), "transition encoding not finite"
    # kmeans: recovers 3 well-separated clusters; K >= V -> identity
    torch.manual_seed(0)
    mu = torch.tensor([[0., 0.], [10., 10.], [-10., 10.]])
    lab = torch.randint(0, 3, (300,))
    X = mu[lab] + 0.1 * torch.randn(300, 2)
    a, C = kmeans(X, 3, iters=10)
    assert a.max().item() == 2 and a.min().item() == 0, "3 clusters not all used"
    assert C.shape == (3, 2)
    for t in range(3):
        sub = torch.arange(300)[lab == t]
        assert a[sub].unique().numel() == 1, f"true cluster {t} split"
    a_id, C_id = kmeans(X, 300)
    assert (a_id == torch.arange(300, device=X.device)).all(), "K>=V not identity"
    # discover_families end-to-end (level 1) + persistence round-trip
    fam, Cb = cb.discover_families(nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D)), 16)
    assert fam.shape == (V,) and Cb.shape == (16, D)
    assert fam.max().item() < 16 and torch.isfinite(Cb).all()
    save_families("/tmp/_te_fams.pt", fam, Cb, 16)
    f2, c2, meta = load_families("/tmp/_te_fams.pt")
    assert (f2 == fam).all() and (c2 == Cb).all() and meta["K"] == 16
    save_families("/tmp/_te_fams.json", fam, Cb, 16)
    f3, c3, meta3 = load_families("/tmp/_te_fams.json")
    assert (f3 == fam).all() and c3 is None and meta3["K"] == 16
    # FamilyMap: load from json, stays on the mapping's device (CPU next to
    # the table) even when the key is on GPU
    fm = FamilyMap("/tmp/_te_fams.json")
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert (fm.map(ids) == fm.families[ids]).all() and fm.K == 16
    if torch.cuda.is_available():
        r = fm(ids.cuda())
        assert r.device.type == "cpu" and (r == fm.families[ids]).all()
        fm.to("cuda")
        r2 = fm.map(ids.cuda())
        assert r2.device.type == "cuda" and (r2 == fm.families[ids]).all()
        fm.to("cpu")
    print(f"model self-check: OK ({sum(p.numel() for p in m.parameters()):,} params at V={V}, {dev})")


if __name__ == "__main__":
    _self_check()
