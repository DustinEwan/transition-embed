"""
Hello-world example: train a transition-embedding codebook on wikitext-103 with
a small MLP transition function (no quanta dependency). Verifies the library
end-to-end.

For the real usage (a conv-like operator such as KDA), see conv_transition_fn.py.

Requires: transition-embed[examples]
Run:
    uv run python examples/train_wikitext.py                     # full wikitext
    MAX_TOKENS=1000000 uv run python examples/train_wikitext.py  # smoke
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn

from transition_embed import train

CODE_DIM = 512
VOCAB_SIZE = 151669  # Qwen3


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
    # A small MLP transition function (the "weakest" non-identity one). Swap in a
    # conv-like operator (KDA, see conv_transition_fn.py) for the real usage.
    transition_fn = nn.Sequential(
        nn.Linear(CODE_DIM, CODE_DIM), nn.SiLU(),
        nn.Linear(CODE_DIM, CODE_DIM), nn.SiLU(),
    )
    train(transition_fn, wikitext_blocks(), VOCAB_SIZE, CODE_DIM,
          out="codebook_wikitext.pt", max_tokens=max_tokens)


if __name__ == "__main__":
    main()
