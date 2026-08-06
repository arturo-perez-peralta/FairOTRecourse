"""Stable log-domain Sinkhorn used by the enhanced mixed-formulation solver.

Works directly on the cost matrix (never materializes ``exp(-c/eps)``), so it
cannot underflow on large costs or at large lambda.  Potentials ``f = eps * log u``
and ``g = eps * log v`` are kept in cost units, which makes warm-starting across
outer iterations (with a changing ``eps``) and across lambda sweeps trivial.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from fairopt.core.sinkhorn import SinkhornResult


@dataclass
class StableSinkhornResult:
    plan: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    marginal_error: float
    n_iter: int
    converged: bool


def _logsumexp_rows(X: torch.Tensor) -> torch.Tensor:
    """log(sum_j exp(X_ij)) over the last dim, numerically stable."""
    m = X.max(dim=-1).values
    return torch.log(torch.exp(X - m.unsqueeze(-1)).sum(dim=-1)) + m


def stable_sinkhorn(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    max_iter: int = 500,
    tol: float = 1e-9,
    f_init: Optional[torch.Tensor] = None,
    g_init: Optional[torch.Tensor] = None,
) -> StableSinkhornResult:
    """Log-domain Sinkhorn with potentials warm-start.

    ``cost`` is (n, m), ``mu`` (n,), ``nu`` (m,).  All tensors must already be
    on the same device.  Returns a StableSinkhornResult whose ``f/g`` are the
    dual potentials in cost units (``f = eps * log u``).
    """
    n, m = cost.shape
    mu = mu.reshape(-1).to(cost.dtype)
    nu = nu.reshape(-1).to(cost.dtype)

    if f_init is None:
        f = torch.zeros(n, dtype=cost.dtype, device=cost.device)
    else:
        f = f_init.reshape(-1).to(cost.dtype)
    if g_init is None:
        g = torch.zeros(m, dtype=cost.dtype, device=cost.device)
    else:
        g = g_init.reshape(-1).to(cost.dtype)

    log_mu = torch.log(mu.clamp(min=1e-300))
    log_nu = torch.log(nu.clamp(min=1e-300))

    M = cost / epsilon

    converged = False
    last_err = 1.0
    i = 0
    for i in range(max_iter):
        f_prev = f
        g_prev = g

        # f = eps * log(u), g = eps * log(v);  u_i = mu_i / (K v)_i  etc.
        # Gauss-Seidel: g update uses the freshly computed f (matches classic u,v alternation).
        # f_i = eps*( log_mu_i - logsumexp_j( g_j/eps - M_ij ) )
        X = (g / epsilon).unsqueeze(0) - M                      # (n, m)
        f_new = epsilon * (log_mu - _logsumexp_rows(X))

        # g_j = eps*( log_nu_j - logsumexp_i( f_i_new/eps - M_ij ) )
        Y = (f_new / epsilon).unsqueeze(1) - M                  # (n, m)
        g_new = epsilon * (log_nu - _logsumexp_rows(Y.permute(1, 0)))

        # Potential change normalized by ``epsilon``: a shift of ``delta`` in a
        # potential changes the plan by ``exp(delta/eps)``, so ``|delta|/eps`` is
        # the right scale-invariant measure of plan (in)stability.  The old
        # relative-to-|f| version diverged whenever the potentials were near zero
        # (high-eps / near-uniform regime), which is exactly when it should stop.
        err = max(
            (f_new - f_prev).abs().max().item(),
            (g_new - g_prev).abs().max().item(),
        ) / epsilon
        last_err = err
        f = f_new
        g = g_new

        if err < tol:
            converged = True
            break

    u = torch.exp(f / epsilon)
    v = torch.exp(g / epsilon)
    plan = torch.exp((f.unsqueeze(1) + g.unsqueeze(0)) / epsilon - M)

    return StableSinkhornResult(
        plan=plan,
        f=f,
        g=g,
        u=u,
        v=v,
        marginal_error=last_err,
        n_iter=i + 1,
        converged=converged,
    )
