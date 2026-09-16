"""Level-1 transition family discovery on a trained artifact.

    python examples/discover_families.py [artifact.pt] [K]

Without an artifact, runs a small self-contained demo (random weights, V=4096,
MLP transition function). For a real artifact, `arch` must be an instance of
the foundational transition-function architecture (the state_dict shapes must
match the saved weights; e.g. KDA for codebooks/wikitext103_kda.pt).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transition_embed import Codebook, BinaryTiedEmbedding


def arch_mlp(d):
    return nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))


def main(artifact=None, K=64):
    if artifact is None:  # self-contained demo
        torch.manual_seed(0)
        V, D = 4096, 64
        p = torch.softmax(torch.randn(V), dim=0)
        s = torch.sign(torch.randn(V, D))
        cb = Codebook(
            bits=BinaryTiedEmbedding.pack_bits(s > 0),
            c=(s * p[:, None]).sum(0),
            in_bias=torch.zeros(V), out_bias=torch.zeros(V),
            alpha=1.0, unigram_p=p,
            transition_fn_weights=arch_mlp(D).state_dict())
        print(f"[demo] random artifact, V={V}, D={D}")
    else:
        cb = Codebook.from_artifact(artifact)
        print(f"[artifact] {artifact}: V={cb.vocab_size}, D={cb.code_dim}")

    fam, C = cb.discover_families(arch_mlp(cb.code_dim), K)
    sizes = torch.bincount(fam, minlength=fam.max().item() + 1)
    print(f"K={K}  families used={sizes.numel()}  "
          f"size min/median/max={sizes.min().item()}/{sizes.median().item()}/{sizes.max().item()}")
    # the Dict[V, V'] is just `fam`: token id -> family id
    return fam, C


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else None
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    main(a, k)
