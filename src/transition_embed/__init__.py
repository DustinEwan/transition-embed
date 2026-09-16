"""
transition-embed: train a fixed, reusable, tied binary codebook whose bits encode
short-range next-token *transition* structure (not similarity).

The library is transition-function-agnostic: you provide any nn.Module mapping
(B,T,D) -> (B,T,D) (e.g. a conv-like operator such as KDA); the library trains
the tied binary embedding + the recipe (weighted CE + stratified VISReg + online
unigram centering) and exports the two foundational artifacts in one file: the
compact bitpacked transition embedding (1 bit per code) and the trained
transition function (state_dict). Downstream tasks (e.g. the V->V' coarsening)
load both and compute the transition encoding.

Public API:
    train(transition_fn, data, vocab_size, code_dim, out, ...)  -> trains + exports
    Codebook.from_artifact(path)                                -> loads the artifact
    Codebook.transition_encoding(arch)                          -> (V, D) marginal h
    Codebook.discover_families(arch, K)                        -> (families (V,), centroids (K, D))
    save_families / load_families                              -> persist the Dict[V, V']
    FamilyMap(path).map(ids)                                  -> token ids -> family ids (gather)
    BinaryTiedEmbedding, StratifiedVISReg, fused_weighted_ce  (the building blocks)
"""
from .model import CodebookModel, Codebook, kmeans, save_families, load_families, FamilyMap
from .embedding import BinaryTiedEmbedding, SignSTE
from .visreg import StratifiedVISReg
from .fused_ce import FusedWeightedCE, fused_weighted_ce
from .trainer import train, make_strata

__all__ = [
    "train",
    "CodebookModel",
    "Codebook",
    "kmeans",
    "save_families",
    "load_families",
    "FamilyMap",
    "BinaryTiedEmbedding",
    "SignSTE",
    "StratifiedVISReg",
    "FusedWeightedCE",
    "fused_weighted_ce",
    "make_strata",
]
