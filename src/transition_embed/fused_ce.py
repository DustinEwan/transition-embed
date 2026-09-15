"""
Fused weighted cross-entropy (Liger-style): never materializes (N, V).

Forward:  single pass over vocab chunks, online logsumexp + target gather.
Backward: recomputes each chunk;  grad = alpha * (p - onehot)  against
          codes / h / out_bias / log_alpha.

Drop-in replacement for chunked_ce (same signature, same w-normalization).
Memory: saved activations drop from ~(N, V) to one chunk (N, chunk).
Cost:   backward does one extra matmul pass (recompute) vs autograd-traced.
"""
import torch


class FusedWeightedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, codes, out_bias, log_alpha, target, w, chunk_size, frozen=False):
        alpha = log_alpha.exp()
        N = h.shape[0]
        V = codes.shape[0]
        dev = h.device
        m = None
        lse_rel = None
        tl = torch.zeros(N, device=dev, dtype=torch.float32)
        for lo in range(0, V, chunk_size):
            hi = min(lo + chunk_size, V)
            lc = alpha * (h @ codes[lo:hi].T) + out_bias[lo:hi]
            cm = lc.max(dim=1).values.float()
            if m is None:
                m = cm
                lse_rel = torch.log(torch.exp(lc - cm[:, None]).sum(1))
            else:
                m_new = torch.maximum(m, cm)
                lse_rel = torch.log(torch.exp(lse_rel + m - m_new)
                                   + torch.exp(lc - m_new[:, None]).sum(1))
                m = m_new
            in_c = (target >= lo) & (target < hi)
            tl += (torch.gather(lc, 1, (target - lo).clamp(0, hi - lo - 1).unsqueeze(1))
                   .squeeze(1).float() * in_c)
        lse = m + lse_rel
        wsum = w.sum().clamp(min=1e-8)
        loss = ((lse - tl) * w).sum() / wsum
        ctx.save_for_backward(h, codes, out_bias, log_alpha, target, w,
                             lse, tl, torch.tensor(float(wsum), device=dev))
        ctx.frozen = frozen
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        h, codes, out_bias, log_alpha, target, w, lse, tl, wsum = ctx.saved_tensors
        alpha = log_alpha.exp()
        ws = w * (grad_output / wsum)
        N, D = h.shape
        V = codes.shape[0]
        cd = codes.to(h.dtype)  # common matmul dtype (h may be bf16 under autocast)
        grad_h = torch.zeros(N, D, device=h.device, dtype=torch.float32)
        frozen = getattr(ctx, "frozen", False)
        grad_codes = None if frozen else torch.zeros(codes.shape, device=h.device, dtype=torch.float32)
        grad_ob = None if frozen else torch.zeros_like(out_bias)
        term1 = torch.zeros((), device=h.device)
        for lo in range(0, V, 16384):  # chunk size not needed in bwd; fixed is fine
            hi = min(lo + 16384, V)
            lc = alpha * (h @ cd[lo:hi].T) + out_bias[lo:hi]
            p = torch.exp(lc.float() - lse[:, None])
            in_c = (target >= lo) & (target < hi)
            idx = (target - lo).clamp(0, hi - lo - 1)
            g = p * ws[:, None]              # loss = lse - tl  =>  grad uses (p - onehot)
            g[in_c, idx[in_c]] -= ws[in_c]  # subtract the onehot, no extra allocation
            gb = g.to(h.dtype)
            grad_h += (gb @ cd[lo:hi]).float() * alpha
            if not frozen:
                grad_codes[lo:hi] = (gb.T @ h).float() * alpha
                grad_ob[lo:hi] = g.sum(0).to(out_bias.dtype)
                dot = (lc.float() - out_bias[lo:hi][None, :]) / alpha
                term1 += (ws[:, None] * p * dot).sum()
        grad_la = None
        if not frozen:
            term2 = (ws * (tl - out_bias[target]) / alpha).sum()
            grad_la = alpha * (term1 - term2)  # d/d log_alpha = alpha * d/d alpha
        return (grad_h.to(h.dtype), grad_codes, grad_ob, grad_la, None, None, None, None)


def fused_weighted_ce(h, codes, out_bias, log_alpha, target, w, chunk_size=16384, frozen=False):
    return FusedWeightedCE.apply(h, codes, out_bias, log_alpha, target, w, chunk_size, frozen)


def _self_check():
    torch.manual_seed(0)
    V, D, N, C = 3000, 64, 128, 512
    h = torch.randn(N, D, requires_grad=True)
    codes = torch.randn(V, D, requires_grad=True)
    ob = torch.randn(V, requires_grad=True)
    la = torch.tensor(-2.0, requires_grad=True)
    target = torch.randint(0, V, (N,))
    w = 0.1 + 9.9 * torch.rand(N)

    # fp32 reference (full materialization)
    alpha = la.exp()
    logits = alpha * (h @ codes.T) + ob
    lse = torch.logsumexp(logits, 1)
    ref = ((lse - logits[torch.arange(N), target]) * w).sum() / w.sum()
    ref.backward()

    def run(chunk):
        h2 = h.detach().clone().requires_grad_(True)
        c2 = codes.detach().clone().requires_grad_(True)
        o2 = ob.detach().clone().requires_grad_(True)
        l2 = la.detach().clone().requires_grad_(True)
        loss = fused_weighted_ce(h2, c2, o2, l2, target, w, chunk)
        loss.backward()
        return loss, h2.grad, c2.grad, o2.grad, l2.grad

    for chunk in (C, 257):  # also checks chunk-size invariance
        loss, gh, gc, go, gl = run(chunk)
        assert abs(ref.item() - loss.item()) < 1e-4, f"loss {ref.item()} vs {loss.item()}"
        assert torch.allclose(h.grad, gh, atol=1e-4, rtol=1e-4), "grad h"
        assert torch.allclose(codes.grad, gc, atol=1e-4, rtol=1e-4), "grad codes"
        assert torch.allclose(ob.grad, go, atol=1e-4, rtol=1e-4), "grad out_bias"
        assert torch.allclose(la.grad, gl, atol=1e-4, rtol=1e-4), "grad log_alpha"
    print(f"fused_ce self-check: OK (loss={ref.item():.6f}, chunks {C}/{257} match fp32 ref)")


if __name__ == "__main__":
    _self_check()
