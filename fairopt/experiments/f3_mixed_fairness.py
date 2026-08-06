#!/usr/bin/env python3
"""F3 — Mixed (group + individual) fairness, enhanced solver.

Uses the rewritten stable solver: decoupled cost ``theta_z*c + lambda_ind*(c-m_z)^2``,
exact bisection theta (d_z <= 4) / Adam theta (d_z > 4), damped m update,
epsilon-annealing, and log-domain Sinkhorn. Same grid as the old E5
(14 lambda_ind x 34 lambda_g) with 10-fold CV, on the 7 core datasets.

One solver instance is reused per fold with warm-start so that adjacent
(lambda_ind, lambda_g) solves reuse Sinkhorn potentials / theta / m_z; the
warm-start is reset at each lambda_ind boundary for stability.

Progressively appends per-dataset results to results/f3_mixed_fairness.csv.
With ``--resume``, datasets already present in the CSV are skipped.
"""
import os
import sys
import time
import warnings
import argparse

os.environ.setdefault("FAIROPT_NUM_THREADS", "16")

warnings.filterwarnings("ignore")

import torch
import pandas as pd
from pathlib import Path

from fairopt.core.cost import pairwise_l2
from fairopt.core.metrics import (
    avg_transport_cost,
    individual_variance,
    per_individual_variance,
    barycentric_projection,
    compute_barycentric_w2,
    per_group_barycentric_w2_sum,
    Timer,
)
from fairopt.experiments.runner import prepare_data, get_split_data, _build_groups_subset
from fairopt.experiments.config import get_config
from fairopt.solvers.combined_group_ind import CombinedGroupIndividualSolver


LAMBDA_IND_VALUES = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
LAMBDA_G_VALUES = [
    0, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,
]
DATASETS = [
    "german", "adult", "compas", "lsac",
    "credit_default", "saheart", "student",
]
MAX_SAMPLES = None

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MAIN_CSV = RESULTS_DIR / "f3_mixed_fairness.csv"


def make_solver(device: str) -> CombinedGroupIndividualSolver:
    return CombinedGroupIndividualSolver(
        epsilon=0.1,
        max_iter=200,
        tol=1e-5,
        lambda_g=1.0,
        lambda_ind=1.0,
        outer_max_iter=100,
        outer_tol=1e-4,
        theta_lr=0.05,
        min_outer_iter=2,
        m_damping=0.5,
        eps_start=0.1,
        warm_start=True,
        theta_adam_iters=50,
        theta_damping=0.4,
        settle_iters=4,
        device=device,
    )


def run_split(
    solver: CombinedGroupIndividualSolver,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    groups_subset,
    lambda_ind: float,
    lambda_g: float,
    cost_matrix: torch.Tensor,
):
    solver.lambda_ind = lambda_ind
    solver.lambda_g = lambda_g

    with Timer() as timer:
        result = solver.solve(mu0, mu1, cost_matrix, groups=groups_subset)

    plan = result.plan
    if plan is None:
        return None

    cost_matrix = cost_matrix.to(plan.device)
    mu1 = mu1.to(plan.device)
    ac = avg_transport_cost(plan, cost_matrix)
    iv = individual_variance(plan, cost_matrix)
    piv = per_individual_variance(plan, cost_matrix)

    projected = barycentric_projection(plan, mu1)
    bw2 = compute_barycentric_w2(projected, mu1)
    if groups_subset is not None and len(groups_subset) > 1:
        bw2_sum = per_group_barycentric_w2_sum(plan, mu1, groups_subset)
    else:
        bw2_sum = bw2

    row = {
        "lambda_ind": lambda_ind,
        "lambda_g": lambda_g,
        "avg_cost": ac,
        "barycentric_w2": bw2,
        "barycentric_w2_sum": bw2_sum,
        "individual_var": iv,
        "per_individual_var": piv,
        "execution_time": timer.elapsed,
        "converged": result.converged,
        "n_iter": result.n_iter,
    }
    if result.metrics:
        for k, v in result.metrics.items():
            if not isinstance(v, (dict, list)):
                row[k] = v
        if "theta" in result.metrics:
            for i, t in enumerate(result.metrics["theta"]):
                row[f"theta_{i}"] = t
    return row


def run_dataset(dataset_name: str, device: str, n_folds: int) -> pd.DataFrame:
    print(f"\n--- F3: {dataset_name} (device={device}) ---", flush=True)
    t0 = time.time()
    cfg = get_config(dataset_name)
    data = prepare_data(
        dataset_name,
        numerical_cols=cfg["numerical_cols"],
        categorical_cols=cfg["categorical_cols"],
        sensitive_cols=cfg["sensitive_single"],
        max_samples=MAX_SAMPLES,
    )
    data["X"] = data["X"].to(device)
    n = data["X"].shape[0]
    groups = data.get("groups")

    n_folds = min(n_folds, n)
    fold_size = n // n_folds
    perm = torch.randperm(n)
    rows = []

    for fold in range(n_folds):
        val_idx = perm[fold * fold_size : (fold + 1) * fold_size]
        train_idx = torch.cat([perm[: fold * fold_size], perm[(fold + 1) * fold_size :]])

        mu0, mu1 = get_split_data(data, train_idx, val_idx)
        groups_subset = _build_groups_subset(groups, train_idx)
        cost_matrix = pairwise_l2(mu0, mu1)

        solver = make_solver(device)
        for lind in LAMBDA_IND_VALUES:
            solver.reset()
            for lgval in LAMBDA_G_VALUES:
                row = run_split(solver, mu0, mu1, groups_subset, lind, lgval, cost_matrix)
                if row is not None:
                    row["fold"] = fold
                    rows.append(row)
            print(f"  fold {fold + 1}/{n_folds} lambda_ind={lind} done ({len(rows)} rows)", flush=True)

        del solver, mu0, mu1, cost_matrix
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results["dataset"] = dataset_name
    elapsed = time.time() - t0
    print(f"  {dataset_name}: {len(results)} rows in {elapsed:.0f}s ({elapsed / 3600:.1f}h)", flush=True)
    return results


def append_csv(results: pd.DataFrame) -> None:
    if MAIN_CSV.exists():
        existing = pd.read_csv(MAIN_CSV)
        combined = pd.concat([existing, results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(MAIN_CSV, index=False)
    print(f"Saved {len(results)} rows to {MAIN_CSV} (total {len(combined)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="F3 mixed group+individual fairness, 10-fold CV")
    ap.add_argument("--resume", action="store_true", help="skip datasets already in the CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU", flush=True)
            device = "cpu"
        else:
            torch.cuda.set_device(0)

    print("=" * 60, flush=True)
    print("F3 enhanced mixed group+individual — 10-fold CV", flush=True)
    print(f"device={device} threads={os.environ.get('FAIROPT_NUM_THREADS')}", flush=True)
    print(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    completed = set()
    if args.resume and MAIN_CSV.exists():
        completed = set(pd.read_csv(MAIN_CSV)["dataset"].unique())
        print(f"Resume: {len(completed)} datasets already done: {sorted(completed)}", flush=True)

    for dataset_name in DATASETS:
        if dataset_name in completed:
            continue
        try:
            results = run_dataset(dataset_name, device, args.folds)
        except Exception as exc:
            print(f"!! {dataset_name} FAILED: {exc!r}", flush=True)
            continue
        append_csv(results)

    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
