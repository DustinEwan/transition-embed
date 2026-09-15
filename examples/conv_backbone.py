"""
Example: train a transition-embedding codebook with a self-contained conv-like
(short-range) backbone. No external backbone dependency — the whole thing is a
few lines of torch.

The point: the *short-range, causal* inductive bias is what shapes the codes
into transition-style (not semantic-style) embeddings. Any nn.Module mapping
(B,T,D) -> (B,T,D) works as the backbone; this is a minimal one.

Requires: transition-embed[examples]
Run:
    uv run python examples/conv_backbone.py
    MAX_TOKENS=1000000 uv run python examples/conv_backbone.py   # smoke
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

from transition_embed import train

CODE_DIM = 512
VOCAB_SIZE = 151669  # Qwen3


class CausalConvBackbone(nn.Module):
    """A minimal conv-like (short-range) backbone: causal depthwise convs + a
    mixing MLP. The small, causal kernel is the short-range inductive bias."""

    def __init__(self, code_dim, kernel=4, layers=2):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(code_dim, code_dim, kernel, groups=code_dim, padding=kernel - 1)
            for _ in range(layers)
        ])
        self.mix = nn.Sequential(
            nn.Linear(code_dim, code_dim), nn.SiLU(),
            nn.Linear(code_dim, code_dim), nn.SiLU(),
        )

    def forward(self, x):  # (B, T, D)
        for c in self.conv:
            y = c(x.transpose(1, 2))        # (B, D, T+kernel-1)
            y = y[:, :, :x.shape[1]]       # causal: trim back to length T
            x = F.silu(y.transpose(1, 2))  # (B, T, D)
        return self.mix(x)


def wikitext_blocks(n_docs=100_000, batch_size=24, seq_len=256):
    """Yield (B, T) blocks from wikitext-103 (Qwen3 tokenizer)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    ids = []
    for i in range(0, n_docs, 20000):
        for e in tok(ds[i:i + 20000]["text"])["input_ids"]:
            ids.extend(e)
    n = (len(ids) // (batch_size * seq_len)) * (batch_size * seq_len)
    t = torch.tensor(ids[:n], dtype=torch.long).view(-1, batch_size, seq_len)
    for i in range(t.shape[0]):
        yield t[i]


def main():
    max_tokens = int(os.environ.get("MAX_TOKENS", 0)) or None
    backbone = CausalConvBackbone(CODE_DIM, kernel=4, layers=2)
    train(backbone, wikitext_blocks(), VOCAB_SIZE, CODE_DIM,
          out="codebook_conv.pt", max_tokens=max_tokens)


if __name__ == "__main__":
    main()
