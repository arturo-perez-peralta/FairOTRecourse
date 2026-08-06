"""F1: Individual Fairness — per-individual transport-cost equalization.

Regularizes the transport plan with ``lambda_ind * (c - m)^2`` so that every
individual pays a similar transport cost. Runs a 10-fold CV grid over
``lambda_ind`` across 7 datasets and saves ``results/f1_individual_fairness.csv``.
"""
import torch
import pandas as pd

from fairopt.experiments.runner import cross_validate, single_split, save_results, prepare_data
from fairopt.experiments.config import get_config
from fairopt.solvers.individual_fairness import IndividualFairnessSolver


LAMBDA_IND_VALUES = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
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
            max_samples=MAX_SAMPLES,
        )
        data["X"] = data["X"].to(device)

        def make_solver():
            return IndividualFairnessSolver(
                epsilon=0.1, max_iter=200, tol=1e-6,
                lambda_ind=1.0, outer_max_iter=50,
            )

        if use_cv:
            results = cross_validate(
                make_solver, data, n_folds=10,
                lambda_ind_values=LAMBDA_IND_VALUES,
            )
        else:
            results = single_split(
                make_solver, data,
                lambda_ind_values=LAMBDA_IND_VALUES,
            )
        results["dataset"] = dataset_name
        all_rows.append(results)

    df = pd.concat(all_rows, ignore_index=True)
    save_results(df, "f1_individual_fairness")
    return df


if __name__ == "__main__":
    run()
