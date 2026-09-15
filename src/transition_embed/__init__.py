"""
transition-embed: train a fixed, reusable, tied binary codebook whose bits encode
short-range next-token *transition* structure (not similarity).

The library is backbone-agnostic: you provide any nn.Module mapping (B,T,D) ->
(B,T,D) (e.g. a conv-like operator such as KDA); the library trains the tied
binary embedding + the recipe (weighted CE + stratified VISReg + online unigram
centering) and exports a compact, portable artifact (1 bit per code).

Public API:
    train(backbone, data, vocab_size, code_dim, out, ...)  -> trains + exports
    Codebook.from_artifact(path)                            -> loads the artifact
    BinaryTiedEmbedding, StratifiedVISReg, fused_weighted_ce  (the building blocks)
"""
from .model import CodebookModel, Codebook
from .embedding import BinaryTiedEmbedding, SignSTE
from .visreg import StratifiedVISReg
from .fused_ce import FusedWeightedCE, fused_weighted_ce
from .trainer import train, make_strata

__all__ = [
    "train",
    "CodebookModel",
    "Codebook",
    "BinaryTiedEmbedding",
    "SignSTE",
    "StratifiedVISReg",
    "FusedWeightedCE",
    "fused_weighted_ce",
    "make_strata",
]
