"""
Binary tied embedding with sign-STE and unigram centering by construction.

Design (locked):
- raw values: nn.Embedding(V, D)  (the only learned table)
- interface:  sign(raw), unigram-centered by construction:
             b_i = sign(raw_i) - sum_j p_j sign(raw_j)
- input:     b_i + (b_in_i - sum_j p_j b_in_j)   (per-token scalar, unigram-centered)
- output:    alpha * h @ B^T + b_out             (tied, alpha = learned log-temperature)
- backward:  STE with tanh surrogate (grad * (1 - tanh(x)^2))

The unigram-weighted mean of the codes is exactly zero by construction, so the
sequence model cannot use the code dimensions to predict token frequency.
Frequency lives in b_out (uncentered, by delegation).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """Forward: hard sign. Backward: tanh surrogate (1 - tanh(x)^2)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        t = torch.tanh(x)
        return grad_output * (1.0 - t * t)


class BinaryTiedEmbedding(nn.Module):
    """
    Args:
        vocab_size: V
        code_dim: D (512)
        unigram_p: (V,) corpus unigram probabilities (buffer, stop-grad)
    """

    def __init__(self, vocab_size: int, code_dim: int, unigram_p: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, code_dim)
        # init away from 0: sign(0) = 0 would give a dead interface at step 0
        nn.init.uniform_(self.emb.weight, -1.0, 1.0)
        self.in_bias = nn.Parameter(torch.zeros(vocab_size))
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        # log-temperature; init so alpha ~ 1/sqrt(D) gives O(1) logits for O(1) h
        self.log_alpha = nn.Parameter(torch.tensor(-0.5 * math.log(float(code_dim))))
        self.register_buffer("p", unigram_p)

    @staticmethod
    def pack_bits(b: torch.Tensor) -> torch.Tensor:
        """(V, D) bool -> (V, D//8) uint8. 1 bit per code."""
        shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=b.device)
        return (b.to(torch.uint8).view(b.shape[0], b.shape[1] // 8, 8) * shifts).sum(2).to(torch.uint8)

    @staticmethod
    def unpack_bits(p: torch.Tensor) -> torch.Tensor:
        """(V, D//8) uint8 -> (V, D) bool."""
        shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=p.device)
        return (p[:, :, None] & shifts).ne(0).view(p.shape[0], p.shape[1] * 8)

    @property
    def centered_codes(self) -> torch.Tensor:
        """(V, D) signed codes, unigram-centered by construction. Exact: sum p_i b_i = 0."""
        s = SignSTE.apply(self.emb.weight)
        c = (s * self.p[:, None]).sum(dim=0)
        return s - c

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, D) input representations."""
        codes = self.centered_codes
        b = self.in_bias - (self.in_bias * self.p).sum()
        return codes[idx] + b[idx, None]

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, D) hidden states -> (B, T, V) logits. D must equal code_dim."""
        codes = self.centered_codes
        return self.log_alpha.exp() * (h @ codes.T) + self.out_bias


def load_codebook(path):
    """Load a compact artifact -> (codes (V,D) fp32, in_bias, out_bias, alpha, unigram_p)."""
    a = torch.load(path, map_location="cpu")
    s = BinaryTiedEmbedding.unpack_bits(a["bits"]).float() * 2.0 - 1.0
    return s - a["c"], a["in_bias"], a["out_bias"], a["alpha"], a["unigram_p"]


def _self_check():
    torch.manual_seed(0)
    V, D = 1024, 64
    p = torch.softmax(torch.randn(V), dim=0)
    m = BinaryTiedEmbedding(V, D, p)

    # 1. forward interface is exactly +/-1 (before centering)
    s = SignSTE.apply(m.emb.weight)
    assert torch.all((s == -1) | (s == 1)), "sign must be +/-1"

    # 2. centering is exact under the unigram prior
    codes = m.centered_codes
    mean = (codes * p[:, None]).sum(dim=0)
    assert mean.abs().max() < 1e-5, f"unigram mean not zero: {mean.abs().max()}"

    # 3. STE gradient reaches the raw values, scaled by (1 - tanh^2)
    m.emb.weight.requires_grad_(True)
    x = m.emb.weight
    y = SignSTE.apply(x)
    y.sum().backward()
    g = x.grad
    expected = 1.0 - torch.tanh(x.detach()) ** 2
    assert torch.allclose(g, expected, atol=1e-5), "STE grad mismatch"

    # 4. gradient vanishes for saturated raw values (|x| large)
    x2 = torch.tensor([[10.0, -10.0, 0.0]], requires_grad=True)
    y2 = SignSTE.apply(x2)
    y2.sum().backward()
    assert x2.grad[0, 2] > x2.grad[0, 0] + x2.grad[0, 1] + 0.9, "saturated grad should vanish"

    # 5. embed/unembed shapes and the input-bias centering
    idx = torch.randint(0, V, (2, 8))
    e = m.embed(idx)
    assert e.shape == (2, 8, D)
    in_bias_mean = (m.in_bias * p).sum()
    assert abs(in_bias_mean) < 1e-6, "in_bias is zero-init, mean must be 0"
    h = torch.randn(2, 8, D)
    logits = m.unembed(h)
    assert logits.shape == (2, 8, V)

    # 6. gradients flow to all four learnable pieces
    (e.sum() + logits.sum()).backward()
    for name, param in [("emb", m.emb.weight), ("in_bias", m.in_bias),
                        ("out_bias", m.out_bias), ("log_alpha", m.log_alpha)]:
        assert param.grad is not None, f"no grad for {name}"

    # 7. bitpack round-trip
    b = torch.randint(0, 2, (17, 64), dtype=torch.bool)
    assert torch.equal(BinaryTiedEmbedding.unpack_bits(BinaryTiedEmbedding.pack_bits(b)), b), "bitpack round-trip"

    print("binary_embedding self-check: OK")


if __name__ == "__main__":
    _self_check()
