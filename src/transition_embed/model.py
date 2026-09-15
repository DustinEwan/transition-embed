"""
Transition-embedding model: binary tied embedding -> user backbone -> tied unembedding.

    idx -> embed_raw(idx)          (raw (B,T,D))
        -> backbone(x)             (B,T,D)   [user-provided, receives the RAW embedding]
        -> alpha * h @ sign^T + b_out       (tied unembed, 2-bit interface)

The backbone is a contract: any nn.Module mapping (B,T,D) -> (B,T,D). KDA (quanta)
is one example; the library is backbone-agnostic. The "tie" is that the output
projection is a signed version of the input table — one learned (V,D) table serves
both the input (raw) and the output (signed).

Why the backbone sees the RAW values, not the sign: the sign is a 1-bit-per-dim
compression (the portable interface). The backbone needs the full-capacity
representation to do the transition; the sign is only for the unembed interface.
"""
import torch
import torch.nn as nn

from .embedding import BinaryTiedEmbedding


class CodebookModel(nn.Module):
    """Binary tied embedding + user backbone + tied unembedding.

    Args:
        vocab_size: V
        code_dim: D (e.g. 512)
        backbone: nn.Module mapping (B,T,D) -> (B,T,D); receives the RAW embedding
        unigram_p: (V,) corpus unigram probabilities (buffer, stop-grad); None -> zeros
    """

    def __init__(self, vocab_size: int, code_dim: int, backbone: nn.Module,
                 unigram_p: torch.Tensor | None = None):
        super().__init__()
        if unigram_p is None:
            unigram_p = torch.zeros(vocab_size)
        self.embedding = BinaryTiedEmbedding(vocab_size, code_dim, unigram_p)
        self.backbone = backbone

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) ids -> (B, T, code_dim) hidden states. The backbone receives the
        RAW embedding (signing is only for the unembed interface)."""
        x = self.embedding.emb(idx)  # raw (B, T, D)
        return self.backbone(x)     # (B, T, D)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Full (B, T, V) logits — for inference/export only (1 token: trivial)."""
        return self.embedding.unembed(self.forward(idx))


class Codebook:
    """A trained transition-embedding artifact (the portable tied table)."""

    def __init__(self, bits, c, in_bias, out_bias, alpha, unigram_p):
        self.bits = bits            # (V, D//8) uint8
        self.c = c                 # (D,) centering vector
        self.in_bias = in_bias     # (V,)
        self.out_bias = out_bias   # (V,)
        self.alpha = alpha         # scalar (log-temperature, exponentiated)
        self.unigram_p = unigram_p  # (V,)

    @classmethod
    def from_artifact(cls, path):
        a = torch.load(path, map_location="cpu")
        return cls(a["bits"], a["c"], a["in_bias"], a["out_bias"],
                   a["alpha"], a["unigram_p"])

    @property
    def vocab_size(self):
        return self.bits.shape[0]

    @property
    def code_dim(self):
        return self.bits.shape[1] * 8

    def signed(self, device=None, dtype=torch.float32):
        """(V, D) raw signed codes (before centering)."""
        return (BinaryTiedEmbedding.unpack_bits(self.bits)
                .to(device, dtype=dtype) * 2.0 - 1.0)

    def codes(self, device=None, dtype=torch.float32):
        """(V, D) signed codes, unigram-centered (the 2-bit interface)."""
        return self.signed(device, dtype) - self.c.to(device, dtype=dtype)

    def save(self, path):
        torch.save({"bits": self.bits, "c": self.c, "in_bias": self.in_bias,
                    "out_bias": self.out_bias, "alpha": self.alpha,
                    "unigram_p": self.unigram_p}, path)


def _self_check():
    torch.manual_seed(0)
    V, D = 1024, 64
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D))
    m = CodebookModel(V, code_dim=D, backbone=backbone).to(dev)
    idx = torch.randint(0, V, (2, 64), device=dev)
    h = m(idx)
    assert h.shape == (2, 64, D), f"shape {h.shape}"
    logits = m.logits(idx)
    assert logits.shape == (2, 64, V), f"shape {logits.shape}"
    logits.sum().backward()
    assert m.embedding.emb.weight.grad is not None, "no grad to raw values"
    assert m.backbone[0].weight.grad is not None, "no grad to backbone"
    # round-trip through the artifact
    with torch.no_grad():
        s = torch.sign(m.embedding.emb.weight)
        p = torch.softmax(torch.randn(V, device=dev), dim=0)
        c = (s * p[:, None]).sum(0)
        torch.save({"bits": BinaryTiedEmbedding.pack_bits(s > 0), "c": c,
                    "in_bias": m.embedding.in_bias, "out_bias": m.embedding.out_bias,
                    "alpha": m.embedding.log_alpha.exp(), "unigram_p": p},
                   "/tmp/_te_selfcheck.pt")
    cb = Codebook.from_artifact("/tmp/_te_selfcheck.pt")
    assert cb.vocab_size == V and cb.code_dim == D
    assert cb.codes().shape == (V, D)
    print(f"model self-check: OK ({sum(p.numel() for p in m.parameters()):,} params at V={V}, {dev})")


if __name__ == "__main__":
    _self_check()
