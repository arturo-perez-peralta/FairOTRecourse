import torch
from typing import Optional, Dict

from fairopt.core.sinkhorn import SinkhornSolver, SinkhornResult
from fairopt.solvers.base import BaseSolver, TransportResult


class IndividualFairnessSolver(BaseSolver):
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        lambda_ind: float = 1.0,
        outer_max_iter: int = 50,
        outer_tol: float = 1e-6,
    ):
        super().__init__(epsilon, max_iter, tol)
        self.lambda_ind = lambda_ind
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol

    def _modified_cost(
        self, cost_matrix: torch.Tensor, m: float
    ) -> torch.Tensor:
        return cost_matrix + self.lambda_ind * (cost_matrix - m) ** 2

    def solve(
        self,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        cost_matrix: torch.Tensor,
        groups: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> TransportResult:
        n0 = mu0.shape[0]
        n1 = mu1.shape[0]
        mu_n = torch.ones(n0) / n0
        nu_m = torch.ones(n1) / n1

        m = (mu_n * cost_matrix.sum(dim=1)).sum().item() / n1

        sinkhorn = SinkhornSolver(epsilon=self.epsilon, max_iter=self.max_iter, tol=self.tol)

        best_result = None
        damping = 0.5

        for i in range(self.outer_max_iter):
            modified_c = self._modified_cost(cost_matrix, m)

            inner_result = sinkhorn.solve(mu_n, nu_m, modified_c)
            plan = inner_result.plan

            m_new = (plan * cost_matrix).sum() / plan.sum()
            m_new = m_new.item()

            m_diff = abs(m_new - m)
            rel_diff = m_diff / max(abs(m), 1e-12)
            m = damping * m_new + (1.0 - damping) * m

            best_result = TransportResult(
                plan=plan,
                f=inner_result.f,
                g=inner_result.g,
                converged=rel_diff < self.outer_tol,
                n_iter=i + 1,
                metrics={"m": m, "m_diff": m_diff, "rel_diff": rel_diff},
            )

            if rel_diff < self.outer_tol:
                break

        if best_result is None:
            best_result = TransportResult(converged=False, n_iter=0)

        return best_result
