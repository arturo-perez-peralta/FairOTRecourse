import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from fairopt.core.cost import pairwise_l2
from fairopt.core.metrics import (
    avg_transport_cost,
    individual_variance,
    per_individual_variance,
    group_variance,
    barycentric_projection,
    compute_barycentric_w2,
    per_group_barycentric_w2_sum,
    Timer,
)
from fairopt.data.datasets import load_dataset
from fairopt.data.preprocessing import StandardScaler, to_tensor
from fairopt.utils.ilr import ilr_transform


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_DATASET_MAX_SAMPLES = {
    "acsincome": 30000,
    "hmda": 30000,
    "adult": 30000,
}


def prepare_data(
    dataset_name: str,
    numerical_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    sensitive_cols: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    df, target, info = load_dataset(dataset_name)
    n = df.shape[0]

    _ds_max = _DATASET_MAX_SAMPLES.get(dataset_name.lower())
    if _ds_max is not None:
        max_samples = min(_ds_max, n)
    elif max_samples is None:
        max_samples = n  # use all observations

    if n > max_samples:
        rng_ss = np.random.RandomState(42)
        keep = rng_ss.choice(n, max_samples, replace=False)
        df = df.iloc[keep].reset_index(drop=True)
        target = target.iloc[keep].reset_index(drop=True)
        n = max_samples

    groups = None
    if sensitive_cols:
        groups = {}
        available = [c for c in sensitive_cols if c in df.columns]
        if len(available) == 1:
            col = available[0]
            for val in df[col].unique():
                mask = (df[col] == val).values
                groups[f"{col}_{val}"] = torch.from_numpy(mask.nonzero()[0].astype(np.int64))
        else:
            combined = df[available[0]].astype(str)
            for col in available[1:]:
                combined = combined + "_" + df[col].astype(str)
            for val in combined.unique():
                mask = (combined == val).values
                groups[str(val).replace(" ", "_")] = torch.from_numpy(mask.nonzero()[0].astype(np.int64))

    available_num = [c for c in numerical_cols if c in df.columns]
    X_num = df[available_num].values.astype(np.float32)
    scaler = StandardScaler()
    X_num_tensor = scaler.fit_transform(torch.from_numpy(X_num))

    X_cat_tensor = None
    if categorical_cols:
        available_cat = [c for c in categorical_cols if c in df.columns]
        if available_cat:
            X_cat_raw = df[available_cat].values
            X_cat_encoded = np.zeros((X_cat_raw.shape[0], len(available_cat)), dtype=np.int64)
            for i, col in enumerate(available_cat):
                col_data = X_cat_raw[:, i]
                codes, _ = pd.factorize(col_data, use_na_sentinel=False)
                X_cat_encoded[:, i] = codes
            X_cat_tensor = torch.from_numpy(X_cat_encoded)

    n = X_num_tensor.shape[0]
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)
    split = int(n * 0.8)
    train_idx = perm[:split]
    test_idx = perm[split:]

    target_tensor = torch.from_numpy(target.values.astype(np.float32))

    result = {
        "X": X_num_tensor,
        "y": target_tensor,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "groups": groups,
        "info": info,
    }
    if X_cat_tensor is not None:
        result["X_cat"] = X_cat_tensor

    return result


def _dirichlet_preprocess(
    X_num: torch.Tensor,
    X_cat: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> tuple:
    X_num = X_num.cpu()
    X_cat = X_cat.cpu()
    train_idx = train_idx.cpu()
    val_idx = val_idx.cpu()

    X0_num = X_num[train_idx].numpy()
    X1_num = X_num[val_idx].numpy()
    X0_cat = X_cat[train_idx].numpy()
    X1_cat = X_cat[val_idx].numpy()

    ilr0_parts = []
    ilr1_parts = []

    for col_i in range(X0_cat.shape[1]):
        y_train_raw = X0_cat[:, col_i]
        unique = np.unique(y_train_raw)
        if len(unique) <= 1:
            continue

        y_train, _ = pd.factorize(y_train_raw)
        valid = y_train >= 0
        if valid.sum() <= 1:
            continue
        if y_train.min() != 0 or len(np.unique(y_train)) != (y_train.max() + 1):
            y_train_remap, _ = pd.factorize(y_train)
            y_train = y_train_remap

        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss",
        )
        try:
            xgb.fit(X0_num, y_train)
        except ValueError:
            continue
        calibrated = CalibratedClassifierCV(xgb, method="sigmoid", cv=KFold(3))
        try:
            calibrated.fit(X0_num, y_train)
        except ValueError:
            continue

        p0 = calibrated.predict_proba(X0_num)
        p1 = calibrated.predict_proba(X1_num)

        if p0.shape[1] > 1:
            ilr0_parts.append(ilr_transform(torch.from_numpy(p0).float()))
            ilr1_parts.append(ilr_transform(torch.from_numpy(p1).float()))

    mu0 = torch.cat([X_num[train_idx]] + ilr0_parts, dim=-1)
    mu1 = torch.cat([X_num[val_idx]] + ilr1_parts, dim=-1)

    return mu0, mu1


def get_split_data(
    data: Dict,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> tuple:
    X = data["X"]
    X_cat = data.get("X_cat")
    if X_cat is not None:
        return _dirichlet_preprocess(X, X_cat, train_idx, val_idx)
    return X[train_idx], X[val_idx]


def _build_groups_subset(
    groups: Dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    sensitive_cols: Optional[List[str]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    if groups is None:
        return None
    idx_to_pos = {old: new for new, old in enumerate(train_idx.tolist())}
    result = {}
    for gname, gmask in groups.items():
        in_train = torch.isin(gmask, train_idx)
        old_vals = gmask[in_train].tolist()
        if not old_vals:
            continue
        new_vals = [idx_to_pos[v] for v in old_vals]
        result[gname] = torch.tensor(new_vals, dtype=torch.long)
    return result if result else None


def _run_split(
    solver_factory: Callable[[], Any],
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    groups_subset: Optional[Dict[str, torch.Tensor]],
    lambda_ind: float,
    lambda_g: float,
    cost_matrix: Optional[torch.Tensor] = None,
) -> Optional[Dict]:
    if cost_matrix is None:
        cost_matrix = pairwise_l2(mu0, mu1)

    with Timer() as timer:
        solver = solver_factory()
        if hasattr(solver, "lambda_ind"):
            solver.lambda_ind = lambda_ind
        if hasattr(solver, "lambda_g"):
            solver.lambda_g = lambda_g

        result = solver.solve(mu0, mu1, cost_matrix, groups=groups_subset)

    plan = result.plan
    if plan is None:
        return None

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


def cross_validate(
    solver_factory: Callable[[], Any],
    data: Dict,
    n_folds: int = 10,
    lambda_ind_values: Optional[List[float]] = None,
    lambda_g_values: Optional[List[float]] = None,
    sensitive_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    n = data["X"].shape[0]
    groups = data.get("groups")

    fold_size = n // n_folds
    perm = torch.randperm(n)
    rows = []

    lambdas_ind = lambda_ind_values or [0.0]
    lambdas_g = lambda_g_values or [0.0]

    for fold in tqdm(range(n_folds), desc=f"CV {n_folds}-fold"):
        val_idx = perm[fold * fold_size : (fold + 1) * fold_size]
        train_idx = torch.cat([perm[: fold * fold_size], perm[(fold + 1) * fold_size :]])

        mu0, mu1 = get_split_data(data, train_idx, val_idx)
        groups_subset = _build_groups_subset(groups, train_idx, sensitive_cols)

        cost_matrix = pairwise_l2(mu0, mu1)

        for lind in lambdas_ind:
            for lgval in lambdas_g:
                row = _run_split(
                    solver_factory, mu0, mu1, groups_subset,
                    lind, lgval, cost_matrix=cost_matrix,
                )
                if row is not None:
                    row["fold"] = fold
                    rows.append(row)

    df_results = pd.DataFrame(rows)
    fname = f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(RESULTS_DIR / fname, index=False)
    return df_results


def single_split(
    solver_factory: Callable[[], Any],
    data: Dict,
    lambda_ind_values: Optional[List[float]] = None,
    lambda_g_values: Optional[List[float]] = None,
    sensitive_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    train_idx = data["train_idx"]
    val_idx = data["test_idx"]

    mu0, mu1 = get_split_data(data, train_idx, val_idx)
    groups_subset = _build_groups_subset(data.get("groups"), train_idx, sensitive_cols)

    cost_matrix = pairwise_l2(mu0, mu1)

    rows = []
    lambdas_ind = lambda_ind_values or [0.0]
    lambdas_g = lambda_g_values or [0.0]

    for lind in lambdas_ind:
        for lgval in lambdas_g:
            row = _run_split(
                solver_factory, mu0, mu1, groups_subset,
                lind, lgval, cost_matrix=cost_matrix,
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, experiment_name: str) -> None:
    path = RESULTS_DIR / f"{experiment_name}.csv"
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
