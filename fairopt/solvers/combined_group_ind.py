import math
import torch
from typing import Optional, Dict, List

from fairopt.core.sinkhorn_stable import stable_sinkhorn
from fairopt.utils.simplex import project_simplex
from fairopt.solvers.base import BaseSolver, TransportResult


class CombinedGroupIndividualSolver(BaseSolver):
    """E5 mixed group + individual fairness.

    Alternating scheme (kept intentionally simple / block-coordinate):

    1.  Sinkhorn (log-domain, Gauss-Seidel) per group with the decoupled cost
        ``theta_z * c + lambda_ind * (c - m_z)^2``.
    2.  Damped fixed-point update of the per-group reference costs ``m_z``.
    3.  ``theta`` update: exact simplex-constrained quadratic minimizer when
        ``d_z <= 4`` (bisection on the KKT multiplier, no learning rate), and
        Adam (adaptive) when ``d_z > 4``.

    The entropy ``epsilon`` is annealed from ``eps_start`` down to ``epsilon``
    across outer iterations (and a final settle at ``epsilon``) for speed and
    stability.  Potentials / ``theta`` / ``m_z`` can be warm-started across
    calls (used by the lambda-sweep runner).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        lambda_g: float = 1.0,
        lambda_ind: float = 1.0,
        outer_max_iter: int = 100,
        outer_tol: float = 1e-6,
        theta_lr: float = 0.05,
        min_outer_iter: int = 2,
        m_damping: float = 0.5,
        eps_start: float = 5.0,
        warm_start: bool = False,
        theta_adam_iters: int = 50,
        theta_damping: float = 0.3,
        settle_iters: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__(epsilon, max_iter, tol)
        self.lambda_g = lambda_g
        self.lambda_ind = lambda_ind
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol
        self.theta_lr = theta_lr
        self.min_outer_iter = min_outer_iter
        self.m_damping = m_damping
        self.eps_start = eps_start
        self.warm_start = warm_start
        self.theta_adam_iters = theta_adam_iters
        self.theta_damping = theta_damping
        self.settle_iters = settle_iters
        self.device = device
        self._ws = None

    def reset(self) -> None:
        self._ws = None

    # ------------------------------------------------------------------ #
    # theta updates
    # ------------------------------------------------------------------ #
    def _solve_theta_exact(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        d_z: int,
        lambda_g: float,
    ) -> torch.Tensor:
        """Exact minimizer of ``(d_z/4*lam)||theta-1||^2 - <a,theta>``
        subject to ``sum(theta) = d_z``, ``theta >= 1e-8``.

        This matches the E5/group-fairness convention (``max_{theta} sum theta_z D_z``):
        groups with above-average cost get ``theta > 1`` so their transport is
        squeezed, equalizing per-group average costs as ``lambda_g`` grows.
        Solved by bisection over the single KKT multiplier (the sum constraint).
        ``theta`` is unused (fresh solve) but kept for a uniform signature.
        """
        lam_eff = max(lambda_g, 1e-8)
        coef = 2.0 * lam_eff / d_z
        lb = 1e-8

        def theta_of(nu: float) -> torch.Tensor:
            return torch.clamp(1.0 + coef * (a - nu), min=lb)

        def s(nu: float) -> float:
            return theta_of(nu).sum().item()

        nu_lo, nu_hi = -10.0, 10.0
        while s(nu_lo) < d_z:
            nu_lo -= 10.0
        while s(nu_hi) > d_z:
            nu_hi += 10.0

        for _ in range(60):
            nu_mid = 0.5 * (nu_lo + nu_hi)
            if s(nu_mid) > d_z:
                nu_lo = nu_mid
            else:
                nu_hi = nu_mid
        return theta_of(0.5 * (nu_lo + nu_hi))

    def _solve_theta_adam(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        d_z: int,
        lambda_g: float,
    ) -> torch.Tensor:
        """Adaptive (Adam) version for d_z > 4; matches the E5 theta convention."""
        lam_eff = max(lambda_g, 1e-8)
        theta_t = theta.detach().clone().to(a.device).requires_grad_(True)
        optimizer = torch.optim.Adam([theta_t], lr=max(self.theta_lr, 1e-4))

        for _ in range(self.theta_adam_iters):
            optimizer.zero_grad()
            obj = (d_z / (4 * lam_eff)) * ((theta_t - 1) ** 2).sum() - (theta_t * a).sum()
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                theta_t.clamp_(min=1e-8)

        theta_new = project_simplex(theta_t.detach(), s=d_z)
        return theta_new.clamp(min=1e-8)

    # ------------------------------------------------------------------ #
    # main solve
    # ------------------------------------------------------------------ #
    def _outer_step(
        self,
        theta: torch.Tensor,
        m_z: Dict[str, float],
        f_init: Dict[str, Optional[torch.Tensor]],
        g_init: Dict[str, Optional[torch.Tensor]],
        group_names: List[str],
        group_idx: Dict[str, torch.Tensor],
        sub_mu: Dict[str, torch.Tensor],
        mu1d: torch.Tensor,
        cost: torch.Tensor,
        d_z: int,
        device: str,
        eps: float,
        update_theta,
        prev_theta: torch.Tensor,
        prev_m: Dict[str, float],
    ) -> dict:
        """One block-coordinate iteration at entropy ``eps``: per-group Sinkhorn,
        damped ``m`` fixed-point step, damped ``theta`` step.  Mutates and returns
        the shared state plus per-iteration diagnostics."""
        plans = {}
        total_cost_g = torch.zeros(d_z, device=device)
        total_mass_g = torch.zeros(d_z, device=device)

        for gidx, gname in enumerate(group_names):
            idx0 = group_idx[gname]
            c_g = cost[idx0, :]
            modified_c = theta[gidx] * c_g + self.lambda_ind * (c_g - m_z[gname]) ** 2

            res = stable_sinkhorn(
                modified_c,
                sub_mu[gname],
                mu1d,
                eps,
                max_iter=self.max_iter,
                tol=self.tol,
                f_init=f_init[gname],
                g_init=g_init[gname],
            )
            f_init[gname] = res.f
            g_init[gname] = res.g

            plans[gname] = res.plan
            total_cost_g[gidx] = (res.plan * c_g).sum()
            total_mass_g[gidx] = res.plan.sum()

        avg_costs = total_cost_g / total_mass_g.clamp(min=1e-12)
        mean_cost = avg_costs.mean().item()
        var_cost = avg_costs.var().item() if d_z > 1 else 0.0
        rel_var = var_cost / (mean_cost**2 + 1e-12)

        # ---- m update (damped fixed point: m = alpha*m_new + (1-alpha)*m) ----
        for gidx, gname in enumerate(group_names):
            m_new = (total_cost_g[gidx] / total_mass_g[gidx].clamp(min=1e-12)).item()
            m_z[gname] = self.m_damping * m_new + (1.0 - self.m_damping) * m_z[gname]

        # ---- theta update (damped toward the exact minimizer) ---------- #
        theta_new = update_theta(theta, avg_costs, d_z, self.lambda_g)
        theta = self.theta_damping * theta_new + (1.0 - self.theta_damping) * theta

        theta_delta = (theta - prev_theta).abs().max().item()
        m_delta = max(
            abs(m_z[g] - prev_m[g]) / max(abs(m_z[g]), 1e-12)
            for g in group_names
        )
        prev_theta = theta.detach().clone()
        prev_m = dict(m_z)

        return {
            "theta": theta,
            "m_z": m_z,
            "f_init": f_init,
            "g_init": g_init,
            "plans": plans,
            "avg_costs": avg_costs,
            "mean_cost": mean_cost,
            "var_cost": var_cost,
            "rel_var": rel_var,
            "theta_delta": theta_delta,
            "m_delta": m_delta,
            "prev_theta": prev_theta,
            "prev_m": prev_m,
        }

    def solve(
        self,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        cost_matrix: torch.Tensor,
        groups: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> TransportResult:
        if groups is None:
            raise ValueError("CombinedGroupIndividualSolver requires `groups` dict")

        device = self.device or str(cost_matrix.device)
        cost = cost_matrix.to(device)
        n0, n1 = cost.shape

        group_names = list(groups.keys())
        d_z = len(group_names)
        group_idx = {gname: idx.to(device) for gname, idx in groups.items()}

        mu0d = torch.ones(n0, device=device) / n0
        mu1d = torch.ones(n1, device=device) / n1
        sub_mu = {gname: torch.ones(idx.shape[0], device=device) / idx.shape[0]
                  for gname, idx in group_idx.items()}

        # ---- initialize / warm-start state ----------------------------- #
        if (
            self.warm_start
            and self._ws is not None
            and self._ws["n0"] == n0
            and self._ws["n1"] == n1
            and self._ws["group_names"] == group_names
        ):
            theta = self._ws["theta"].to(device)
            m_z = {g: v for g, v in self._ws["m_z"].items()}
            f_init = {g: self._ws["f"][g].to(device) for g in group_names}
            g_init = {g: self._ws["g"][g].to(device) for g in group_names}
        else:
            theta = torch.ones(d_z, device=device)
            m_z = {
                gname: float(cost[group_idx[gname], :].mean())
                for gname in group_names
            }
            f_init = {g: None for g in group_names}
            g_init = {g: None for g in group_names}

        if d_z <= 4:
            update_theta = self._solve_theta_exact
        else:
            update_theta = self._solve_theta_adam

        best_rel_var = float("inf")
        best_snapshot = None
        prev_theta = theta.detach().clone()
        prev_m = dict(m_z)
        st = None

        for outer in range(self.outer_max_iter):
            eps_k = max(self.epsilon, self.eps_start * (0.3 ** outer))
            at_floor = eps_k <= self.epsilon + 1e-12

            st = self._outer_step(
                theta, m_z, f_init, g_init, group_names, group_idx,
                sub_mu, mu1d, cost, d_z, device, eps_k, update_theta,
                prev_theta, prev_m,
            )
            theta, m_z, f_init, g_init = (
                st["theta"], st["m_z"], st["f_init"], st["g_init"],
            )
            prev_theta, prev_m = st["prev_theta"], st["prev_m"]

            if at_floor and st["rel_var"] < best_rel_var:
                best_rel_var = st["rel_var"]
                best_snapshot = {
                    "plans": st["plans"],
                    "theta": theta.detach().clone(),
                    "m_z": dict(m_z),
                    "avg_costs": st["avg_costs"].detach().clone(),
                    "mean_cost": st["mean_cost"],
                    "var_cost": st["var_cost"],
                }

            if outer < self.min_outer_iter - 1:
                continue
            if best_rel_var < self.outer_tol:
                break
            if at_floor and st["theta_delta"] < 1e-4 and st["m_delta"] < 1e-2:
                break

        # ---- settle at the reported epsilon (tighten m/theta fixed point) ---- #
        settle_count = 0
        for _ in range(self.settle_iters):
            settle_count += 1
            st = self._outer_step(
                theta, m_z, f_init, g_init, group_names, group_idx,
                sub_mu, mu1d, cost, d_z, device, self.epsilon, update_theta,
                prev_theta, prev_m,
            )
            theta, m_z, f_init, g_init = (
                st["theta"], st["m_z"], st["f_init"], st["g_init"],
            )
            prev_theta, prev_m = st["prev_theta"], st["prev_m"]

            if st["rel_var"] < best_rel_var:
                best_rel_var = st["rel_var"]
                best_snapshot = {
                    "plans": st["plans"],
                    "theta": theta.detach().clone(),
                    "m_z": dict(m_z),
                    "avg_costs": st["avg_costs"].detach().clone(),
                    "mean_cost": st["mean_cost"],
                    "var_cost": st["var_cost"],
                }

            if best_rel_var < self.outer_tol:
                break
            if st["theta_delta"] < 1e-5 and st["m_delta"] < 1e-4:
                break

        n_outer = outer + 1 + settle_count
        snap = best_snapshot or {
            "plans": st["plans"],
            "theta": theta.detach().clone(),
            "m_z": dict(m_z),
            "avg_costs": st["avg_costs"],
            "mean_cost": st["mean_cost"],
            "var_cost": st["var_cost"],
        }
        plans = snap["plans"]
        full_plan = torch.zeros(n0, n1, device=device)
        for gname, p in plans.items():
            full_plan[group_idx[gname]] = p

        if self.warm_start:
            self._ws = {
                "n0": n0,
                "n1": n1,
                "group_names": group_names,
                "theta": theta.detach().clone().cpu(),
                "m_z": {g: m_z[g] for g in group_names},
                "f": {g: f_init[g].detach().cpu() for g in group_names},
                "g": {g: g_init[g].detach().cpu() for g in group_names},
            }

        final_avg_costs = snap["avg_costs"]
        final_mean = snap["mean_cost"]
        final_var = snap["var_cost"]

        result = TransportResult(
            plan=full_plan,
            converged=best_rel_var < self.outer_tol,
            n_iter=n_outer,
            metrics={
                "group_var": final_var,
                "mean_cost": final_mean,
                "rel_var": best_rel_var,
                "theta": snap["theta"].detach().tolist(),
                "m_z": {g: snap["m_z"][g] for g in group_names},
                "avg_costs": final_avg_costs.detach().tolist(),
            },
            extra={"plans": plans, "theta": snap["theta"], "m_z": snap["m_z"].copy()},
        )
        return result
