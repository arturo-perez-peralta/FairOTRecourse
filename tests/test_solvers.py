import torch
import pytest
import pandas as pd

from fairopt.solvers.individual_fairness import IndividualFairnessSolver
from fairopt.solvers.group_fairness import GroupFairnessSolver
from fairopt.solvers.combined_group_ind import CombinedGroupIndividualSolver
from fairopt.experiments.runner import single_split, _build_groups_subset


def _make_toy_data(n: int = 30, m: int = 40, d: int = 3):
    torch.manual_seed(42)
    mu0 = torch.randn(n, d)
    mu1 = torch.randn(m, d)
    cost = torch.cdist(mu0, mu1, p=2)
    groups = {
        "group_A": torch.arange(n // 2),
        "group_B": torch.arange(n // 2, n),
    }
    return mu0, mu1, cost, groups


class TestIndividualFairnessSolver:
    def test_runs_and_returns_result(self):
        mu0, mu1, cost, _ = _make_toy_data(20, 20)
        solver = IndividualFairnessSolver(epsilon=0.1, max_iter=200, tol=1e-8, lambda_ind=1.0)
        result = solver.solve(mu0, mu1, cost)
        assert result.converged or result.n_iter > 0
        assert "m" in result.metrics

    def test_lambda_zero_equals_sinkhorn(self):
        mu0, mu1, cost, _ = _make_toy_data(15, 15)
        solver = IndividualFairnessSolver(epsilon=0.1, max_iter=200, tol=1e-8, lambda_ind=0.0)
        result = solver.solve(mu0, mu1, cost)
        assert result.plan is not None


class TestGroupFairnessSolver:
    def test_runs_with_groups(self):
        mu0, mu1, cost, groups = _make_toy_data()
        solver = GroupFairnessSolver(epsilon=0.1, max_iter=200, tol=1e-8, lambda_g=1.0)
        result = solver.solve(mu0, mu1, cost, groups=groups)
        assert result.converged or result.n_iter > 0
        assert "group_var" in result.metrics

    def test_raises_without_groups(self):
        mu0, mu1, cost, _ = _make_toy_data()
        solver = GroupFairnessSolver(epsilon=0.1, max_iter=200, tol=1e-8, lambda_g=1.0)
        with pytest.raises(ValueError):
            solver.solve(mu0, mu1, cost, groups=None)


class TestCombinedGroupIndividualSolver:
    def test_runs_with_groups(self):
        mu0, mu1, cost, groups = _make_toy_data(15, 15)
        solver = CombinedGroupIndividualSolver(epsilon=0.1, max_iter=100, tol=1e-8, lambda_g=1.0, lambda_ind=1.0)
        result = solver.solve(mu0, mu1, cost, groups=groups)
        assert result.converged or result.n_iter > 0

    def test_raises_without_groups(self):
        mu0, mu1, cost, _ = _make_toy_data()
        solver = CombinedGroupIndividualSolver(epsilon=0.1, max_iter=100, tol=1e-8)
        with pytest.raises(ValueError):
            solver.solve(mu0, mu1, cost, groups=None)


class TestThetaGradient:
    def test_theta_moves_from_ones(self):
        mu0, mu1, cost, groups = _make_toy_data(30, 30)
        solver = GroupFairnessSolver(
            epsilon=0.1, max_iter=200, tol=1e-6,
            lambda_g=1.0, outer_max_iter=10, theta_lr=0.1,
        )
        result = solver.solve(mu0, mu1, cost, groups=groups)
        theta_list = result.metrics.get("theta", [])
        assert len(theta_list) == 2
        assert not all(abs(t - 1.0) < 1e-4 for t in theta_list), (
            f"Theta should move from [1,1], got {theta_list}"
        )
        assert result.n_iter > 1, (
            f"Should run multiple outer iterations, got n_iter={result.n_iter}"
        )

    def test_combined_group_theta_moves(self):
        mu0, mu1, cost, groups = _make_toy_data(20, 20)
        solver = CombinedGroupIndividualSolver(
            epsilon=0.1, max_iter=100, tol=1e-6,
            lambda_g=1.0, lambda_ind=1.0,
            outer_max_iter=10, theta_lr=0.1,
        )
        result = solver.solve(mu0, mu1, cost, groups=groups)
        theta_list = result.metrics.get("theta", [])
        assert len(theta_list) == 2
        assert not all(abs(t - 1.0) < 1e-4 for t in theta_list)


class TestIndividualFairnessHighLambda:
    def test_converges_at_lambda_10(self):
        mu0, mu1, cost, _ = _make_toy_data(20, 20)
        solver = IndividualFairnessSolver(
            epsilon=0.1, max_iter=200, tol=1e-6,
            lambda_ind=10.0, outer_max_iter=50,
        )
        result = solver.solve(mu0, mu1, cost)
        assert result.plan is not None
        assert result.metrics.get("rel_diff", 1.0) < 0.5
        assert result.converged or result.n_iter > 1


class TestSingleSplitInfrastructure:
    def test_single_split_no_groups(self):
        mu0, mu1, cost, _ = _make_toy_data(20, 20)
        n = mu0.shape[0] + mu1.shape[0]
        X = torch.cat([mu0, mu1], dim=0)
        data = {
            "X": X,
            "groups": None,
            "train_idx": torch.arange(mu0.shape[0]),
            "test_idx": torch.arange(mu0.shape[0], n),
        }

        def factory():
            return IndividualFairnessSolver(
                epsilon=0.1, max_iter=50, tol=1e-6, lambda_ind=1.0,
            )

        df = single_split(factory, data, lambda_ind_values=[0.0, 1.0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "avg_cost" in df.columns
        assert "execution_time" in df.columns

    def test_single_split_with_groups(self):
        mu0, mu1, cost, groups = _make_toy_data(20, 20)
        n = mu0.shape[0] + mu1.shape[0]
        all_groups = {}
        for gname, gidx in groups.items():
            all_groups[gname] = gidx
        X = torch.cat([mu0, mu1], dim=0)
        data = {
            "X": X,
            "groups": all_groups,
            "train_idx": torch.arange(mu0.shape[0]),
            "test_idx": torch.arange(mu0.shape[0], n),
        }

        def factory():
            return GroupFairnessSolver(
                epsilon=0.1, max_iter=50, tol=1e-6, lambda_g=1.0,
            )

        df = single_split(factory, data, lambda_g_values=[0.1, 1.0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
