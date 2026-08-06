import torch
from typing import Optional, Dict, List, Tuple

from fairopt.core.sinkhorn import SinkhornSolver
from fairopt.utils.simplex import project_simplex
from fairopt.solvers.base import BaseSolver, TransportResult


class GroupFairnessSolver(BaseSolver):
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        lambda_g: float = 1.0,
        outer_max_iter: int = 100,
        outer_tol: float = 1e-6,
        theta_lr: float = 0.01,
        min_outer_iter: int = 2,
    ):
        super().__init__(epsilon, max_iter, tol)
        self.lambda_g = lambda_g
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol
        self.theta_lr = theta_lr
        self.min_outer_iter = min_outer_iter

    def solve(
        self,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        cost_matrix: torch.Tensor,
        groups: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> TransportResult:
        if groups is None:
            raise ValueError("GroupFairnessSolver requires `groups` dict")

        group_names = list(groups.keys())
        d_z = len(group_names)

        theta = torch.ones(d_z)

        sinkhorn = SinkhornSolver(epsilon=self.epsilon, max_iter=self.max_iter, tol=self.tol)

        best_result = None

        for i in range(self.outer_max_iter):
            plans = {}
            total_cost_g = torch.zeros(d_z)
            total_mass_g = torch.zeros(d_z)

            for gidx, gname in enumerate(group_names):
                idx0 = groups[gname]
                cost_g = cost_matrix[idx0, :]
                n0_g = idx0.shape[0]
                n1 = mu1.shape[0]
                mu_n_g = torch.ones(n0_g) / n0_g
                nu_m = torch.ones(n1) / n1

                theta_g = theta[gidx]
                modified_c = theta_g * cost_g

                result_g = sinkhorn.solve(mu_n_g, nu_m, modified_c)
                plans[gname] = result_g.plan

                total_cost_g[gidx] = (result_g.plan * cost_g).sum()
                total_mass_g[gidx] = plans[gname].sum()

            avg_costs = total_cost_g / total_mass_g.clamp(min=1e-12)
            mean_cost = avg_costs.mean().item()
            var_cost = avg_costs.var().item() if d_z > 1 else 0.0

            lam_eff = max(self.lambda_g, 1e-8)
            theta_grad = d_z / (2 * lam_eff) * (theta - 1) - avg_costs
            theta = theta - self.theta_lr * theta_grad
            theta = project_simplex(theta, s=d_z)
            theta = theta.clamp(min=1e-8)

            rel_var = var_cost / (mean_cost**2 + 1e-12)

            best_result = TransportResult(
                plan=None,
                converged=rel_var < self.outer_tol,
                n_iter=i + 1,
                metrics={
                    "theta": theta.detach().tolist(),
                    "avg_costs": avg_costs.detach().tolist(),
                    "mean_cost": mean_cost,
                    "group_var": var_cost,
                    "rel_var": rel_var,
                },
                extra={"plans": plans, "theta": theta.detach().clone()},
            )

            if i < self.min_outer_iter - 1:
                continue

            if rel_var < self.outer_tol:
                break

        if best_result is not None:
            n0 = mu0.shape[0]
            n1 = mu1.shape[0]
            full_plan = torch.zeros(n0, n1)
            for gname, p in plans.items():
                idx0 = groups[gname]
                full_plan[idx0] = p
            best_result.plan = full_plan

        if best_result is None:
            best_result = TransportResult(converged=False, n_iter=0)

        return best_result
