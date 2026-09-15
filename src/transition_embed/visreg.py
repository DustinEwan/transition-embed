"""
Stratified VISReg: Variance-Invariance-Sketching Regularization (arXiv 2606.02572),
applied independently within frequency strata.

Reference implementation (project page, ~15 lines core), per stratum:

    mu = z.mean(0)
    L_center = mu.pow(2).mean()
    std = (z - mu).std(0, unbiased=False)
    L_scale  = (1 - std).pow(2).mean()          # targets std = 1
    z_norm = (z - mu) / std.detach()           # scale/shape decoupled via stop-grad
    W = randn(D, K); W /= W.norm(2, dim=0)
    L_shape = (sort(z_norm @ W, dim=0).values - Normal(0,1).icdf(u)).pow(2).mean()

Notes:
- normalization is per-dimension standardization (NOT row-wise L2): the
  projection of standardized data onto a random direction is ~ N(0, 1),
  which is what the Gaussian-quantile target assumes
- the SWD term is blind to pure collapse (standardizing a constant column
  amplifies noise back to N(0,1)); the scale term is what catches collapse
- computed on a per-stratum subsample: full-stratum sort is O(K N log N)
  and must redo every step since the codes change
"""
import torch
import torch.nn as nn


def _normal_quantiles(n: int, device, dtype) -> torch.Tensor:
    """Normal(0,1).icdf(arange(1, n+1) / (n+1)) — reference quantile rule."""
    from torch.distributions import Normal
    u = torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)
    return Normal(0.0, 1.0).icdf(u)


class StratifiedVISReg(nn.Module):
    """
    Args:
        code_dim: D
        strata: list of 1-D index tensors partitioning the vocabulary
        n_slices: number of random projection directions K
        subsample: max tokens per stratum for the SWD term
    """

    def __init__(self, code_dim: int, strata: list[torch.Tensor],
                 n_slices: int = 64, subsample: int = 2048):
        super().__init__()
        self.n_slices = n_slices
        self.subsample = subsample
        self.strata = list(strata)  # ragged index tensors; moved via _apply

    def _apply(self, fn):
        out = super()._apply(fn)
        self.strata = [fn(s) for s in self.strata]
        return out

    def forward(self, raw: torch.Tensor):
        """
        Args:
            raw: (V, D) raw embedding values
        Returns:
            (L_center, L_scale, L_shape), averaged over strata
        """
        Lc = raw.new_zeros(())
        Ls = raw.new_zeros(())
        Lh = raw.new_zeros(())
        n_used = 0
        for idx in self.strata:
            X = raw[idx]  # (N_s, D)
            n = X.shape[0]
            if n == 0:
                continue  # empty stratum (e.g. step 0, no data yet)
            n_used += 1

            mu = X.mean(dim=0)
            Lc = Lc + mu.pow(2).mean()

            Xc = X - mu
            std = Xc.std(dim=0, unbiased=False)
            Ls = Ls + (1.0 - std).pow(2).mean()

            if n > self.subsample:
                sel = torch.randperm(n, device=raw.device)[: self.subsample]
                Xs, mus = X[sel], mu
            else:
                Xs, mus = X, mu
            z_norm = (Xs - mus) / std[None, :].detach().clamp(min=1e-8)
            W = torch.randn(raw.shape[1], self.n_slices, device=raw.device, dtype=raw.dtype)
            W = W / W.norm(p=2, dim=0).clamp(min=1e-8)
            p_sorted = torch.sort(z_norm @ W, dim=0).values
            target = _normal_quantiles(p_sorted.shape[0], raw.device, raw.dtype).unsqueeze(1)  # (N,1)
            Lh = Lh + (p_sorted - target).pow(2).mean()

        n_strata = max(n_used, 1)
        return Lc / n_strata, Ls / n_strata, Lh / n_strata


def _self_check():
    torch.manual_seed(0)
    V, D = 4096, 64
    head = torch.arange(0, V // 10)
    tail = torch.arange(V // 10, V)
    reg = StratifiedVISReg(D, [head, tail], n_slices=32, subsample=1024)

    # 1. well-spread data (rows ~ N(0, I)): all terms small
    good = torch.randn(V, D)
    lc, ls, lh = reg(good)
    assert lc.item() < 0.1, f"center should be ~0, got {lc.item()}"
    assert ls.item() < 0.1, f"scale should be ~0 for std=1, got {ls.item()}"
    assert lh.item() < 0.1, f"shape should be ~0 for N(0,I), got {lh.item()}"

    # 2. collapsed data: scale term large (catches collapse),
    #    shape term blind (standardizing a constant column amplifies noise
    #    back to N(0,1)) -- the documented division of labor
    collapsed = torch.ones(V, D) + 0.01 * torch.randn(V, D)
    cc, cs, ch = reg(collapsed)
    assert cs.item() > 0.5, f"scale should be large for collapse, got {cs.item()}"
    assert ch.item() < 0.5, f"shape expected blind to pure collapse, got {ch.item()}"

    # 3. low-rank data (2 of 64 dims active): shape term fires
    aniso = torch.zeros(V, D)
    aniso[:, 0] = torch.randn(V)
    aniso[:, 1] = torch.randn(V)
    ac, as_, ah = reg(aniso)
    assert ah.item() > 0.3, f"shape should catch low-rank, got {ah.item()}"

    # 4. gradients flow to the raw values
    x = torch.randn(V, D, requires_grad=True)
    lc, ls, lh = reg(x)
    (lc + ls + lh).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "no gradient"

    # 5. balanced +/-1 codes (the actual training regime): scale ~ 0
    codes = torch.randint(0, 2, (V, D)) * 2 - 1
    bc, bs, bh = reg(codes.float())
    assert bs.item() < 0.1, f"scale should be ~0 for balanced +/-1, got {bs.item()}"

    print("visreg self-check: OK")


if __name__ == "__main__":
    _self_check()
