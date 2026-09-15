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

    def save(self, path):
        d = {"bits": self.bits, "c": self.c, "in_bias": self.in_bias,
             "out_bias": self.out_bias, "alpha": self.alpha,
             "unigram_p": self.unigram_p}
        if self.transition_fn_weights is not None:
            d["transition_fn"] = self.transition_fn_weights
        torch.save(d, path)


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
    print(f"model self-check: OK ({sum(p.numel() for p in m.parameters()):,} params at V={V}, {dev})")


if __name__ == "__main__":
    _self_check()
