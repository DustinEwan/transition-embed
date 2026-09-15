"""
Transition-embedding training (the foundational stage): binary tied embedding +
user transition function.

    L = weighted_CE (inverse-frequency) + lam_v * VISReg (stratified, on raw values)

The transition function is a contract: any nn.Module mapping (B,T,D) -> (B,T,D).
KDA (quanta) is one example; the library is transition-function-agnostic. It
receives the SIGNED codes (sign(raw) - c + b_in), matching the trained model.

The foundational stage exports two self-consistent artifacts in one file:
  1. the bitpacked transition embedding (1 bit per code)
  2. the trained transition function (state_dict)
Downstream tasks (e.g. the V->V' coarsening) load both and compute the
transition encoding (Codebook.transition_encoding).

Unigram statistics accumulate naively online: a running count, single pass over the
corpus. Unseen token -> p = 0 -> tail stratum, max weight (long-tail assumption).
No prepass, no refresh schedule: strata are rebuilt from the running count every
step (cheap; cumulative-mass boundaries are stable once the head is counted).

Usage (single GPU):
    from transition_embed import train
    train(transition_fn, data_iter, vocab_size, code_dim, out="codebook.pt")

Usage (2 GPUs, data-parallel; count all-reduced per step, grads via DDP):
    torchrun --nproc_per_node=2 -m transition_embed.examples.train_wikitext
"""
import os
import math
import time

import torch
import torch.distributed as dist

from .model import CodebookModel
from .visreg import StratifiedVISReg
from .fused_ce import fused_weighted_ce
from .embedding import BinaryTiedEmbedding


def make_strata(p: torch.Tensor, cuts) -> list:
    """Rank tokens by p, cut at cumulative-mass boundaries."""
    _, order = p.sort(descending=True)
    cum = p[order].cumsum(0)
    edges = ([0] + [min(int(torch.searchsorted(cum, c)) + 1, int(p.numel())) for c in cuts]
             + [int(p.numel())])
    return [order[edges[i]:edges[i + 1]] for i in range(len(cuts) + 1)]


def train(transition_fn, data, vocab_size, code_dim, out,
          lr=1e-3, warmup=100, lam_v=0.1, tau=1e-4, gamma=0.5,
          n_slices=64, subsample=2048, cuts=(0.01, 0.10, 0.50),
          ce_chunk=16384, log_every=100, save_every=500_000,
          max_tokens=None, resume=None):
    """Train a transition-embedding codebook (the foundational stage).

    Args:
        transition_fn: nn.Module mapping (B,T,D) -> (B,T,D); receives the SIGNED
            codes (sign(raw) - c + b_in), matching the trained model.
        data: iterator of (B,T) blocks (torch.Tensor of token ids); signals done by
            raising StopIteration. All blocks must share one shape.
        vocab_size: V
        code_dim: D (e.g. 512)
        out: output artifact path (holds both the bitpacked embedding and the
            transition-function weights)
        lr, warmup, lam_v, tau, gamma, n_slices, subsample, cuts, ce_chunk,
        log_every, save_every, max_tokens, resume: training config (defaults match
        the reference recipe).
    """
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"vocab={vocab_size:,}  rank={rank}/{world}  device={dev}")
    transition_fn = transition_fn.to(dev)
    model = CodebookModel(vocab_size, code_dim, transition_fn).to(dev)
    if world > 1:
        # broadcast_buffers=False: embedding.p is updated manually from the synced count
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False)
    mm = model.module if world > 1 else model  # bare model for param/buffer access
    visreg = StratifiedVISReg(code_dim, [torch.arange(vocab_size, device=dev)],
                              n_slices=n_slices, subsample=subsample)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    count = torch.zeros(vocab_size, device=dev, dtype=torch.int64)  # global cumulative count
    step = 0
    n_tok = 0
    batch_size = seq_len = None
    if resume:
        ck = torch.load(resume, map_location=dev)
        mm.load_state_dict(ck["state"])
        count.copy_(ck["count"])
        step = ck["step"]
        n_tok = ck["n_tok"]
        log(f"resumed from {resume}: step={step} n_tok={n_tok}")
    log(f"params={sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    last_ckpt = 0
    model.train()
    it = iter(data)
    while True:
        blk = next(it, None)
        dummy = blk is None
        if blk is not None and batch_size is None:
            batch_size, seq_len = blk.shape
        idx = (torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
               if dummy else blk.to(dev, non_blocking=True))

        # ---- online unigram statistics (naive count) ----
        # all-reduce only the increment: all_reduce(count) would double-count the
        # accumulated history (both ranks hold the full sum) -> geometric growth.
        inc = (torch.zeros(vocab_size, device=dev, dtype=torch.int64)
               if dummy else torch.bincount(idx.view(-1), minlength=vocab_size))
        if world > 1:
            dist.all_reduce(inc)  # SUM: global increment this step
        count.add_(inc)
        p = count.float() / count.sum().clamp(min=1)
        mm.embedding.p.copy_(p)  # centering + input-bias centering
        visreg.strata = [s for s in make_strata(mm.embedding.p, cuts)]
        w = (tau / mm.embedding.p.clamp(min=tau)) ** gamma  # (V,) per-token weights

        # ---- loss (bf16) ----
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(idx)  # (B, T, D)
            if dummy:
                # zero-gradient step: keeps DDP + collectives in lockstep while the
                # other rank drains its remaining blocks. All params get a (zero)
                # grad so DDP (find_unused_parameters=False) stays happy.
                loss = (h.sum() + mm.embedding.out_bias.sum()
                       + mm.embedding.log_alpha) * 0.0
                loss_ce = loss_v = lc = ls = lh = None
            else:
                target = idx[:, 1:]
                loss_ce = fused_weighted_ce(
                    h[:, :-1].reshape(-1, code_dim),
                    mm.embedding.centered_codes,
                    mm.embedding.out_bias,
                    mm.embedding.log_alpha,
                    target.reshape(-1),
                    w[idx[:, 1:]].reshape(-1),
                    ce_chunk)
                lc, ls, lh = visreg(mm.embedding.emb.weight)
                loss_v = lc + ls + lh
                loss = loss_ce + lam_v * loss_v

        opt.zero_grad()
        loss.backward()  # DDP all-reduces (averages) grads
        if not dummy:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            opt.step()
            n_tok += idx.numel()
            step += 1

        # ---- done handshake: exit only when BOTH ranks are exhausted ----
        # (one rank's half can end ~200 steps before the other's; exiting early
        # desyncs the collectives -> RCCL watchdog timeout -> SIGABRT)
        if world > 1:
            done = torch.tensor([1.0 if dummy else 0.0], device=dev)
            dist.all_reduce(done)
            both_done = done.item() >= world
        else:
            both_done = dummy
        if both_done:
            break
        if max_tokens and n_tok >= max_tokens:
            break

        if not dummy and step % log_every == 0 and rank == 0:
            sizes = [len(s) for s in visreg.strata]
            dt = time.time() - t0
            log(f"s{step:6d} tok={n_tok/1e6:8.2f}M/rank  ce={loss_ce.item():.4f} "
                f"ppl={math.exp(loss_ce.item()):10.1f}  visreg={loss_v.item():.5f} "
                f"(c {lc.item():.2e} s {ls.item():.2e} h {lh.item():.2e})  "
                f"strata={sizes}  {2*n_tok/1e6/(dt/60):.2f}M tok/min (cluster)")

        # periodic checkpoint (rank 0; count is already global after all_reduce)
        if (not dummy and rank == 0 and n_tok - last_ckpt >= save_every
                and not (max_tokens and n_tok >= max_tokens)):
            last_ckpt = n_tok
            torch.save({"state": mm.state_dict(), "count": count.clone(),
                       "step": step, "n_tok": n_tok}, "transition_embed_ckpt.pt")
            log(f"checkpoint -> transition_embed_ckpt.pt (step {step})")

    if world > 1:
        # one-time check: params identical across ranks (DDP avg grads + deterministic AdamW)
        local = torch.stack([p.detach().float().norm() for p in mm.parameters()])
        chk = local.clone()
        dist.all_reduce(chk)  # SUM across ranks
        maxdiff = (chk - local * world).abs().max().item()
        log(f"param sync check: max |sum - world*local| = {maxdiff:.3e}")
        dist.barrier()
    if rank == 0:
        # ---- export artifact: the portable tied table (compact: 1 bit per code) ----
        # raw values are a gradient carrier only; never needed at inference.
        with torch.no_grad():
            s = torch.sign(mm.embedding.emb.weight)
            p = count.float() / count.sum().clamp(min=1)
            c = (s * p[:, None]).sum(0)  # centering vector (D,)
            artifact = {
                "bits": BinaryTiedEmbedding.pack_bits(s > 0).cpu(),  # (V, D//8) uint8
                "c": c.cpu(),
                "in_bias": mm.embedding.in_bias.cpu(),
                "out_bias": mm.embedding.out_bias.cpu(),
                "alpha": mm.embedding.log_alpha.exp().cpu(),
                "unigram_p": p.cpu(),
                # artifact 2: the trained transition function (state_dict)
                "transition_fn": mm.transition_fn.state_dict(),
            }
        torch.save(artifact, out)
        sz = os.path.getsize(out) / 1e6
        log(f"done: {2*n_tok/1e6:.2f}M tokens (cluster), {step} steps/rank, "
            f"{time.time()-t0:.0f}s -> {out} ({sz:.1f} MB)")
        log(f"count check: count.sum()={count.sum().item():,}  expected~{world*n_tok:,} (incl. resume prefix)")
