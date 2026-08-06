"""F2: Group Fairness — per-group transport-cost equalization.

Regularizes the transport plan per demographic group with the ``theta`` factor
(``theta_z * c``), pushing above-average groups toward the global average cost as
``lambda_g`` grows. Runs a 10-fold CV grid over ``lambda_g`` across 7 datasets and
saves ``results/f2_group_fairness.csv``.
"""
import torch
import pandas as pd

from fairopt.experiments.runner import cross_validate, single_split, save_results, prepare_data
from fairopt.experiments.config import get_config
from fairopt.solvers.group_fairness import GroupFairnessSolver


LAMBDA_G_VALUES = [
    0, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,
]
DATASETS = [
    "german", "adult", "lsac",
    "credit_default", "saheart", "student",
    "communities",
]
MAX_SAMPLES = None


def run(device: str = "cpu", use_cv: bool = False) -> pd.DataFrame:
    all_rows = []

    for dataset_name in DATASETS:
        cfg = get_config(dataset_name)
        data = prepare_data(
            dataset_name,
            numerical_cols=cfg["numerical_cols"],
            categorical_cols=cfg["categorical_cols"],
            sensitive_cols=cfg["sensitive_single"],
            max_samples=MAX_SAMPLES,
        )
        data["X"] = data["X"].to(device)

        def make_solver():
            return GroupFairnessSolver(
                epsilon=0.1, max_iter=200, tol=1e-6,
                lambda_g=1.0, outer_max_iter=100,
                theta_lr=0.1,
            )

        if use_cv:
            results = cross_validate(
                make_solver, data, n_folds=10,
                lambda_g_values=LAMBDA_G_VALUES,
            )
        else:
            results = single_split(
                make_solver, data,
                lambda_g_values=LAMBDA_G_VALUES,
            )
        results["dataset"] = dataset_name
        all_rows.append(results)

    df = pd.concat(all_rows, ignore_index=True)
    save_results(df, "f2_group_fairness")
    return df


if __name__ == "__main__":
    run()
