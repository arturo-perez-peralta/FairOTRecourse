import os
import sys
import time
import warnings
import argparse
from pathlib import Path

NUM_CORES = "1"
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CORES
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CORES

import torch
torch.set_num_threads(int(NUM_CORES))

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cvxpy as cp
from joblib import Parallel, delayed

from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fairopt.core.cost import pairwise_l2
from fairopt.experiments.runner import prepare_data, get_split_data, _build_groups_subset
from fairopt.experiments.config import get_config
from fairopt.solvers.combined_group_ind import CombinedGroupIndividualSolver
import ot

DATASETS = [
    "saheart", "student", "german", "compas",
    "lsac", "credit_default", "adult",
]

SKIP_BASELINES_DATASETS = []

LAMBDA_IND_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
LAMBDA_G_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

MAX_SAMPLES = None
RESULTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_F3 = RESULTS_DIR / "f3_def.csv"
CSV_BASELINES = RESULTS_DIR / "comparison_baselines_ot.csv"

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_ot_metrics(X_rej, X_acc, plan, groups_rej, model):
    diff = X_rej.unsqueeze(1) - X_acc.unsqueeze(0)
    cost_matrix = (diff ** 2).sum(dim=2)
    
    total_mass = plan.sum().clamp(min=1e-12)
    avg_cost = (plan * cost_matrix).sum().item() / total_mass.item()
    
    ind_var = (plan * ((cost_matrix - avg_cost) ** 2)).sum().item() / total_mass.item()
    
    mass_i = plan.sum(dim=1)
    valid = mass_i > 1e-12
    expected_cost_i = (plan * cost_matrix).sum(dim=1) / mass_i.clamp(min=1e-12)
    per_ind_var = expected_cost_i[valid].var(unbiased=False).item() if valid.sum() > 1 else 0.0
    
    if groups_rej is not None:
        unique_groups = torch.unique(groups_rej)
        if len(unique_groups) > 1:
            group_means = [expected_cost_i[(groups_rej == g) & valid].mean().item() for g in unique_groups if ((groups_rej == g) & valid).sum() > 0]
            group_var = float(np.var(group_means)) if len(group_means) > 1 else float('nan')
        else: group_var = float('nan')
    else: group_var = float('nan')
        
    pi_cond = plan / mass_i.clamp(min=1e-12).unsqueeze(1)
    
    with torch.no_grad():
        logits = model(X_acc)
        y_preds = (logits > 0).float()
        probs = torch.sigmoid(logits)
        
    validity_i = torch.sum(pi_cond * y_preds.unsqueeze(0), dim=1)
    val = validity_i[valid].mean().item() * 100 if valid.sum() > 0 else 0.0
    
    conf_i = torch.sum(pi_cond * probs.unsqueeze(0), dim=1)
    confidence = conf_i[valid].mean().item() * 100 if valid.sum() > 0 else 0.0
    
    actionability = 100.0  
    if valid.sum() > 0:
        proj_X = torch.matmul(pi_cond, X_acc)
        proj_X_np, X_acc_np = proj_X[valid].cpu().detach().numpy(), X_acc.cpu().detach().numpy()
        a, b = np.ones(len(proj_X_np)) / len(proj_X_np), np.ones(len(X_acc_np)) / len(X_acc_np)
        M = ot.dist(proj_X_np, X_acc_np, metric='sqeuclidean')
        wass_dist = ot.emd2(a, b, M)
    else:
        wass_dist = float('nan')
        
    return avg_cost, ind_var, per_ind_var, group_var, val, confidence, wass_dist, actionability

def compute_unified_metrics(X_rej, proj_X, groups_rej, X_acc, model, is_gupta=False, w_g=None, b_g=None):
    cost = torch.sum((X_rej - proj_X)**2, dim=1)
    avg_cost = cost.mean().item()
    
    ind_var = cost.var(unbiased=False).item()
    per_ind_var = ind_var
    
    if groups_rej is not None:
        unique_groups = torch.unique(groups_rej)
        if len(unique_groups) > 1:
            group_means = [cost[groups_rej == g].mean().item() for g in unique_groups if (groups_rej == g).sum() > 0]
            group_var = float(np.var(group_means)) if len(group_means) > 1 else float('nan')
        else: group_var = float('nan')
    else: group_var = float('nan')
        
    if is_gupta:
        w_t, b_t = torch.from_numpy(w_g).float().to(proj_X.device), float(b_g)
        logits = proj_X @ w_t + b_t
    else:
        with torch.no_grad(): logits = model(proj_X)
            
    val = (logits > 0).float().mean().item() * 100
    confidence = torch.sigmoid(logits).mean().item() * 100
    
    actionability = float('nan')
    proj_X_np, X_acc_np = proj_X.cpu().detach().numpy(), X_acc.cpu().detach().numpy()
    a, b = np.ones(len(proj_X_np)) / len(proj_X_np), np.ones(len(X_acc_np)) / len(X_acc_np)
    M = ot.dist(proj_X_np, X_acc_np, metric='sqeuclidean')
    wass_dist = ot.emd2(a, b, M)
    
    return avg_cost, ind_var, per_ind_var, group_var, val, confidence, wass_dist, actionability

def run_gupta_svm(X_train, y_train, X_rej, groups_rej, lambda_gupta=1.0, C_val=1.0):
    X_tr, y_tr, X_rej_np = X_train.cpu().numpy(), np.where(y_train.cpu().numpy() == 0, -1, 1), X_rej.cpu().numpy()
    n, d = X_tr.shape
    w, b, xi, eps = cp.Variable(d), cp.Variable(), cp.Variable(n), cp.Variable()
    
    objective = cp.Minimize(0.5 * cp.sum_squares(w) + C_val * cp.sum(xi) + lambda_gupta * eps)
    constraints = [cp.multiply(y_tr, X_tr @ w + b) >= 1 - xi, xi >= 0, eps >= 0]
    
    if groups_rej is not None:
        g_rej = groups_rej.cpu().numpy()
        unique_groups = np.unique(g_rej)
        if len(unique_groups) > 1:
            group_margins = [cp.sum(-(X_rej_np[g_rej == g] @ w + b)) / (g_rej == g).sum() for g in unique_groups if (g_rej == g).sum() > 0]
            for i in range(len(group_margins)):
                for j in range(i + 1, len(group_margins)):
                    constraints.extend([group_margins[i] - group_margins[j] <= eps, group_margins[j] - group_margins[i] <= eps])
            
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if w.value is None: return X_rej.clone(), np.zeros(d), 0.0
    w_val, b_val = w.value, b.value
    margins = np.maximum(-(X_rej_np @ w_val + b_val), 0)
    proj_np = X_rej_np + (margins[:, np.newaxis] / (np.sum(w_val**2) + 1e-8)) * w_val
    return torch.from_numpy(proj_np).float().to(X_rej.device), w_val, b_val

def run_wachter(X_rej, model):
    X_cf = X_rej.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([X_cf], lr=0.1)
    target = torch.ones(X_cf.shape[0]).to(X_cf.device)
    for _ in range(150):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(X_cf, X_rej) + 10.0 * torch.nn.functional.binary_cross_entropy_with_logits(model(X_cf), target)
        loss.backward()
        optimizer.step()
    return X_cf.detach()

def run_carrizosa_taylor(X_rej: torch.Tensor, X_acc: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if X_acc.shape[0] < 5: return X_rej.clone()
    X_rej_np, device, X_cf = X_rej.cpu().detach().numpy(), X_rej.device, torch.zeros_like(X_rej)
    
    for i in range(X_rej.shape[0]):
        x_0_np = X_rej_np[i].astype(np.float64)
        x_t = torch.tensor(x_0_np, dtype=torch.float32, device=device)
        for step in range(10):
            x_t.requires_grad_(True)
            pred = model(x_t.unsqueeze(0)).squeeze()
            if pred.item() > 0.01 and step > 0: break
            model.zero_grad(); pred.backward()
            g, f_val, x_t_np = x_t.grad.cpu().numpy().flatten().astype(np.float64), float(pred.item()), x_t.cpu().detach().numpy().astype(np.float64)
            
            def obj(x_cand): return 0.5 * np.sum((x_cand - x_0_np) ** 2)
            def constr_valid(x_cand): return g @ x_cand - (0.01 - f_val + g @ x_t_np)

            res = minimize(obj, x0=x_t_np, constraints=[{'type': 'ineq', 'fun': constr_valid}], method='SLSQP')
            if res.success: x_t = torch.tensor(res.x, dtype=torch.float32, device=device)
            else: break
        X_cf[i] = x_t.detach()
    return X_cf

def solve_single_carrizosa(x_0_np, d, W1, b1, W2, b2, W3, b3):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    m = gp.Model("c_exact", env=env)
    m.setParam('TimeLimit', 10)
    x_prime, M = m.addMVar(shape=d, lb=-GRB.INFINITY), 100.0
    
    z1, h1, a1 = m.addMVar(128, lb=-GRB.INFINITY), m.addMVar(128, lb=0.0), m.addMVar(128, vtype=GRB.BINARY)
    m.addConstr(z1 == W1 @ x_prime + b1); m.addConstr(h1 >= z1); m.addConstr(h1 <= M * a1); m.addConstr(h1 <= z1 + M * (1 - a1))
    
    z2, h2, a2 = m.addMVar(64, lb=-GRB.INFINITY), m.addMVar(64, lb=0.0), m.addMVar(64, vtype=GRB.BINARY)
    m.addConstr(z2 == W2 @ h1 + b2); m.addConstr(h2 >= z2); m.addConstr(h2 <= M * a2); m.addConstr(h2 <= z2 + M * (1 - a2))
    
    out = m.addMVar(1, lb=-GRB.INFINITY)
    m.addConstr(out == W3 @ h2 + b3); m.addConstr(out >= 0.01)
    m.setObjective((x_prime - x_0_np) @ (x_prime - x_0_np), GRB.MINIMIZE)
    m.optimize()
    
    return torch.from_numpy(x_prime.X).float() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0 else torch.from_numpy(x_0_np).float()

def run_carrizosa_gurobi(X_rej: torch.Tensor, X_acc: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if X_acc.shape[0] < 5: return X_rej.clone()
    X_rej_np, d = X_rej.cpu().detach().numpy(), X_rej.shape[1]
    
    W1, b1 = model.net[0].weight.data.cpu().numpy(), model.net[0].bias.data.cpu().numpy()
    W2, b2 = model.net[2].weight.data.cpu().numpy(), model.net[2].bias.data.cpu().numpy()
    W3, b3 = model.net[4].weight.data.cpu().numpy(), model.net[4].bias.data.cpu().numpy()
    
    res = Parallel(n_jobs=32)(delayed(solve_single_carrizosa)(X_rej_np[i], d, W1, b1, W2, b2, W3, b3) for i in range(X_rej.shape[0]))
    return torch.stack(res).to(X_rej.device)

def run_facegroup(X_rej, X_acc):
    if X_acc.shape[0] < 5: return X_rej.clone()
    X_rej_np, X_acc_np = X_rej.cpu().numpy(), X_acc.cpu().numpy()
    X_all = np.vstack([X_rej_np, X_acc_np])
    
    try: densities = gaussian_kde(X_all.T)(X_all.T)
    except Exception: densities = np.ones(X_all.shape[0])
    
    distances, indices = NearestNeighbors(n_neighbors=min(10, X_all.shape[0])).fit(X_all).kneighbors(X_all)
    N_all, row, col, data = X_all.shape[0], [], [], []
    for i in range(N_all):
        for j, d in zip(indices[i], distances[i]):
            if i != j:
                row.append(i); col.append(j); data.append(d / (min(densities[i], densities[j]) + 1e-6))
                
    dist_matrix = dijkstra(csgraph=csr_matrix((data, (row, col)), shape=(N_all, N_all)), directed=False, indices=np.arange(len(X_rej_np)))[:, len(X_rej_np):]
    eps, covered, centers, assigned = np.percentile(np.min(dist_matrix, axis=1), 75), set(), [], np.zeros(len(X_rej_np), dtype=int)
    
    for _ in range(min(20, len(X_acc_np))):
        best_cov = []
        for v in range(len(X_acc_np)):
            cov = [u for u in range(len(X_rej_np)) if u not in covered and dist_matrix[u, v] <= eps]
            if len(cov) > len(best_cov): best_v, best_cov = v, cov
        if not best_cov: break
        centers.append(best_v)
        for u in best_cov: covered.add(u); assigned[u] = best_v
            
    for u in range(len(X_rej_np)):
        if u not in covered: assigned[u] = centers[np.argmin(dist_matrix[u, centers])] if len(centers) > 0 else np.argmin(dist_matrix[u])
    return torch.from_numpy(X_acc_np[assigned]).float().to(X_rej.device)

def run_standard_ot(X_rej, X_acc):
    n_src, n_tgt = X_rej.shape[0], X_acc.shape[0]
    if n_src == 0 or n_tgt == 0: return None
    mu, nu = np.ones(n_src)/n_src, np.ones(n_tgt)/n_tgt
    cost = ((X_rej.unsqueeze(1) - X_acc.unsqueeze(0)) ** 2).sum(dim=2).cpu().numpy()
    return torch.from_numpy(ot.emd(mu, nu, cost)).float().to(X_rej.device)

def make_solver(device: str) -> CombinedGroupIndividualSolver:
    return CombinedGroupIndividualSolver(
        epsilon=0.01, max_iter=500, tol=1e-6, outer_max_iter=100, outer_tol=1e-5, theta_lr=0.05,
        min_outer_iter=2, m_damping=0.5, eps_start=0.5, warm_start=True, theta_adam_iters=50, 
        theta_damping=0.4, settle_iters=4, device=device,
    )

def append_to_csv(df_new: pd.DataFrame, path: Path):
    """
    Guarda usando pd.concat para asegurar que columnas dinámicas (ej. thetas de COMPAS) 
    no desestructuren el archivo si este se creó con menos grupos previamente.
    """
    if path.exists():
        df_old = pd.read_csv(path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)

def run_dataset(dataset_name: str, device: str, n_folds: int):
    print(f"\n[{dataset_name}] Procesando Intra-Fold Recourse...", flush=True)
    t0 = time.time()
    cfg = get_config(dataset_name)
    data = prepare_data(
        dataset_name,
        numerical_cols=cfg["numerical_cols"],
        categorical_cols=cfg["categorical_cols"],
        sensitive_cols=cfg["sensitive_single"],
        max_samples=MAX_SAMPLES,
    )
    
    n, groups = data["X"].shape[0], data.get("groups")
    n_folds = min(n_folds, n)
    fold_size = n // n_folds
    
    torch.manual_seed(42)
    np.random.seed(42)
    perm = torch.randperm(n)
    
    f3_rows_all_folds, baselines_rows = [], []

    for fold in range(n_folds):
        print(f"  -> Fold {fold + 1}/{n_folds}", flush=True)
        val_idx = perm[fold * fold_size : (fold + 1) * fold_size]
        train_idx = torch.cat([perm[: fold * fold_size], perm[(fold + 1) * fold_size :]])

        X_train, X_val = get_split_data(data, train_idx, val_idx)
        X_train, X_val, y_train = X_train.to(device), X_val.to(device), data["y"][train_idx].to(device)

        model = MLP(X_train.shape[1]).to(device)
        optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.BCEWithLogitsLoss()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train.float())
            loss.backward(); optimizer.step()
        model.eval()

        with torch.no_grad(): preds_val = (model(X_val) > 0).float()
            
        mask_rej, mask_acc = (preds_val == 0), (preds_val == 1)
        X_rej, X_acc = X_val[mask_rej], X_val[mask_acc]

        if X_rej.shape[0] < 5 or X_acc.shape[0] < 5: 
            print(f"     Skip fold (pocos rechazados/aceptados)")
            continue

        groups_val_raw = _build_groups_subset(groups, val_idx)
        groups_rej_raw, groups_rej_tensor = {}, None
        if groups_val_raw is not None:
            g_val_tensor = torch.zeros(X_val.shape[0], dtype=torch.long, device=device)
            if isinstance(groups_val_raw, dict):
                for g_lbl, (g_name, g_idx) in enumerate(groups_val_raw.items()):
                    g_val_tensor[g_idx] = g_lbl
                groups_rej_tensor = g_val_tensor[mask_rej]
                for g_lbl, (g_name, _) in enumerate(groups_val_raw.items()):
                    groups_rej_raw[g_name] = torch.nonzero(groups_rej_tensor == g_lbl).squeeze(-1)
            else:
                g_val_tensor = groups_val_raw.to(device) if torch.is_tensor(groups_val_raw) else torch.tensor(np.array(groups_val_raw), device=device).squeeze()
                groups_rej_tensor, groups_rej_raw = g_val_tensor[mask_rej], g_val_tensor[mask_rej]
                
        cost_matrix = pairwise_l2(X_rej, X_acc)

        solver = make_solver(device)
        fold_f3_rows = []
        for lind in LAMBDA_IND_VALUES:
            solver.reset() 
            for lgval in LAMBDA_G_VALUES:
                solver.lambda_ind, solver.lambda_g = lind, lgval
                t_s = time.time()
                try:
                    res_ot = solver.solve(X_rej, X_acc, cost_matrix, groups=groups_rej_raw)
                    ac, iv, piv, gv, val, conf, wass, act = compute_ot_metrics(X_rej, X_acc, res_ot.plan, groups_rej_tensor, model)
                    row = {
                        "dataset": dataset_name, "fold": fold, "lambda_ind": lind, "lambda_g": lgval,
                        "avg_cost": ac, "individual_var": iv, "per_individual_var": piv, "group_var": gv, 
                        "validity": val, "confidence": conf, "wasserstein_dist": wass, "actionability": act, 
                        "execution_time": time.time() - t_s
                    }
                    if res_ot.metrics and "theta" in res_ot.metrics:
                        for idx_t, th in enumerate(res_ot.metrics["theta"]): row[f"theta_{idx_t}"] = th
                    fold_f3_rows.append(row)
                except Exception: pass
        f3_rows_all_folds.extend(fold_f3_rows)

        if dataset_name not in SKIP_BASELINES_DATASETS:
            baselines = {}
            try:
                t_s = time.time(); p, w, b = run_gupta_svm(X_train, y_train, X_rej, groups_rej_tensor)
                baselines["Gupta"] = {"proj": p, "is_gupta": True, "w_g": w, "b_g": b, "time": time.time() - t_s}
            except Exception: pass
            
            t_s = time.time(); baselines["Wachter"] = {"proj": run_wachter(X_rej, model), "is_gupta": False, "time": time.time() - t_s}
            t_s = time.time(); baselines["Carrizosa_Taylor"] = {"proj": run_carrizosa_taylor(X_rej, X_acc, model), "is_gupta": False, "time": time.time() - t_s}
            t_s = time.time(); baselines["Carrizosa_Exact"] = {"proj": run_carrizosa_gurobi(X_rej, X_acc, model), "is_gupta": False, "time": time.time() - t_s}
            t_s = time.time(); baselines["GroupFACE"] = {"proj": run_facegroup(X_rej, X_acc), "is_gupta": False, "time": time.time() - t_s}

            for m_name, m_data in baselines.items():
                ac, iv, piv, gv, val, conf, wass, act = compute_unified_metrics(X_rej, m_data["proj"], groups_rej_tensor, X_acc, model, is_gupta=m_data["is_gupta"], w_g=m_data.get("w_g"), b_g=m_data.get("b_g"))
                baselines_rows.append({"dataset": dataset_name, "fold": fold, "method": m_name, "avg_cost": ac, "individual_var": iv, "per_individual_var": piv, "group_var": gv, "validity": val, "confidence": conf, "wasserstein_dist": wass, "actionability": act, "execution_time": m_data["time"]})

        t_s = time.time(); plan_std = run_standard_ot(X_rej, X_acc)
        if plan_std is not None:
            ac, iv, piv, gv, val, conf, wass, act = compute_ot_metrics(X_rej, X_acc, plan_std, groups_rej_tensor, model)
            baselines_rows.append({"dataset": dataset_name, "fold": fold, "method": "Standard OT", "avg_cost": ac, "individual_var": iv, "per_individual_var": piv, "group_var": gv, "validity": val, "confidence": conf, "wasserstein_dist": wass, "actionability": act, "execution_time": time.time() - t_s})

    df_f3_ds = pd.DataFrame(f3_rows_all_folds)
    append_to_csv(df_f3_ds, CSV_F3)

    if not df_f3_ds.empty and dataset_name not in SKIP_BASELINES_DATASETS:
        grouped = df_f3_ds.groupby(['lambda_ind', 'lambda_g'])[['avg_cost', 'per_individual_var', 'group_var']].mean().reset_index()
        grouped['dist_to_ideal'] = np.sqrt(grouped['avg_cost']**2 + grouped['per_individual_var']**2 + grouped['group_var']**2)
        
        ind_df, grp_df = grouped[grouped['lambda_g'] < 1e-8], grouped[grouped['lambda_ind'] < 1e-8]
        best_ind = ind_df.loc[ind_df['dist_to_ideal'].idxmin()] if not ind_df.empty else None
        best_grp = grp_df.loc[grp_df['dist_to_ideal'].idxmin()] if not grp_df.empty else None
        best_mix = grouped.loc[grouped['dist_to_ideal'].idxmin()] if not grouped.empty else None
        
        for fold in range(n_folds):
            if best_ind is not None:
                sub = df_f3_ds[(df_f3_ds['fold'] == fold) & (df_f3_ds['lambda_ind'] == best_ind['lambda_ind']) & (df_f3_ds['lambda_g'] == best_ind['lambda_g'])]
                if not sub.empty:
                    r_ind = sub.iloc[0].copy()
                    r_ind['method'] = 'OT_Ind'
                    baselines_rows.append(r_ind.to_dict())
            if best_grp is not None:
                sub = df_f3_ds[(df_f3_ds['fold'] == fold) & (df_f3_ds['lambda_ind'] == best_grp['lambda_ind']) & (df_f3_ds['lambda_g'] == best_grp['lambda_g'])]
                if not sub.empty:
                    r_grp = sub.iloc[0].copy()
                    r_grp['method'] = 'OT_Group'
                    baselines_rows.append(r_grp.to_dict())
            if best_mix is not None:
                sub = df_f3_ds[(df_f3_ds['fold'] == fold) & (df_f3_ds['lambda_ind'] == best_mix['lambda_ind']) & (df_f3_ds['lambda_g'] == best_mix['lambda_g'])]
                if not sub.empty:
                    r_mix = sub.iloc[0].copy()
                    r_mix['method'] = 'OT_Mixed'
                    baselines_rows.append(r_mix.to_dict())

    if baselines_rows:
        df_base_ds = pd.DataFrame(baselines_rows)
        cols_to_keep = ['dataset', 'fold', 'method', 'avg_cost', 'individual_var', 'per_individual_var', 'group_var', 'validity', 'confidence', 'wasserstein_dist', 'actionability', 'execution_time']
        df_base_ds = df_base_ds[[c for c in cols_to_keep if c in df_base_ds.columns]]
        append_to_csv(df_base_ds, CSV_BASELINES)

    print(f"  {dataset_name} completado en {time.time() - t0:.0f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    device = "cpu" if (args.device.startswith("cuda") and not torch.cuda.is_available()) else args.device
    print("=" * 60, "\nPipeline Unificado: Sensitivity + Baselines (Intra-Fold)\n", "=" * 60, flush=True)

    completed_f3 = set(pd.read_csv(CSV_F3)['dataset'].unique()) if CSV_F3.exists() else set()
    for ds in DATASETS:
        if ds in completed_f3:
            print(f"Saltando {ds} (Ya presente en CSV)")
            continue
        try: run_dataset(ds, device, args.folds)
        except Exception as e: print(f"!! Fallo en {ds}: {e}")

if __name__ == "__main__":
    main()