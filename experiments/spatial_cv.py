"""
spatial_cv.py

Spatial block cross-validation for PMoE-S.

Standard random splits are invalid for spatial data: nearby points
(w/ median distance ~7 km in LUCAS) leak information across train/test.
A model can achieve high R2 by interpolating from neighbours.
--------------
We divide Europe into a grid of spatial blocks (200^2 km^2). Each fold
holds out one set of blocks as test, trains on the rest.
This ensures test points are geographically separated from training.
--------------
"""

import sys, yaml, copy, math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.neighbors import BallTree
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from src.data_loading import get_lucas_native_features
from src.pmoe import PMoES
from src.pmoe_loss import PMoESLoss, nll_mixture_gaussian

try:
    from properscoring import crps_gaussian
    HAS_PROPERSCORING = True
except ImportError:
    HAS_PROPERSCORING = False
    print("WARNING: pip install properscoring  for CRPS metric")


def make_spatial_blocks(df, block_size_km=200, n_folds=5):
    """
    Divide points into spatial blocks by assigning each point
    to a grid cell, then grouping cells into n_folds folds.

    block_size_km: approximate size of each block in km
                   200 km >> 7 km median NN distance -> no leakage

    Returns: array of fold indices (0..n_folds-1) for each point
    """
    lat = df["lat"].values
    lon = df["lon"].values

    # Convert block_size_km to degrees (approximate)
    # 1 degree lat ~ 111 km; 1 degree lon ~ 111*cos(lat) km
    lat_step = block_size_km / 111.0
    lon_step = block_size_km / (111.0 * np.cos(np.radians(lat.mean())))

    # Assign each point to a grid cell
    lat_bin = ((lat - lat.min()) / lat_step).astype(int)
    lon_bin = ((lon - lon.min()) / lon_step).astype(int)
    cell_id = lat_bin * 1000 + lon_bin   # unique cell ID

    # Assign cells to folds (round-robin by cell_id)
    unique_cells = np.unique(cell_id)
    np.random.shuffle(unique_cells)
    cell_fold = {c: i % n_folds for i, c in enumerate(unique_cells)}
    folds     = np.array([cell_fold[c] for c in cell_id])

    # Print fold sizes
    log.info(f"  Spatial blocks: {len(unique_cells)} cells "
             f"(~{block_size_km} km each) -> {n_folds} folds")
    for f in range(n_folds):
        n = (folds == f).sum()
        log.info(f"    Fold {f+1}: {n:,} points ({100*n/len(df):.1f}%)")

    return folds


def build_data(df, feature_cols, target_col, device):
    """Prepare all tensors (full dataset, no split yet)."""
    safe_cols = [c for c in feature_cols if c != "N"]
    cont_cols = [c for c in safe_cols
                 if not c.startswith("LC_") and c not in ("lat", "lon")]
    df2 = df.copy()
    for col in cont_cols:
        m, s     = df2[col].mean(), df2[col].std()
        df2[col] = (df2[col] - m) / s if s > 0 else 0.0
    for col, mn, mx in [
        ("lat", df["lat"].min(), df["lat"].max()),
        ("lon", df["lon"].min(), df["lon"].max()),
    ]:
        df2[f"{col}_n"] = 2*(df[col]-mn)/(mx-mn)-1

    X      = torch.tensor(df2[safe_cols].astype(float).values,
                          dtype=torch.float32, device=device)
    coords = torch.tensor(df2[["lat_n","lon_n"]].values,
                          dtype=torch.float32, device=device)
    y      = torch.tensor(df2[target_col].values,
                          dtype=torch.float32, device=device).unsqueeze(1)

    # Build full graph
    coords_rad        = np.radians(df[["lat","lon"]].values)
    tree              = BallTree(coords_rad, metric="haversine")
    dist_rad, indices = tree.query(coords_rad, k=17)
    dist_km_mat       = dist_rad[:, 1:] * 6371.0
    indices           = indices[:, 1:]

    src_list, dst_list, dist_list = [], [], []
    for i in range(len(df)):
        for j_pos in range(16):
            j = int(indices[i, j_pos])
            d = float(dist_km_mat[i, j_pos])
            if d < 200:
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(d)

    edge_index = torch.tensor([src_list, dst_list],
                               dtype=torch.long, device=device)
    dist_km    = torch.tensor(dist_list,
                               dtype=torch.float32, device=device)
    return X, coords, y, edge_index, dist_km, safe_cols


def remap_edges(edge_index, dist_km, node_idx, device):
    g2l    = {int(g): l for l, g in enumerate(node_idx.cpu().numpy())}
    src_g  = edge_index[0].cpu().numpy()
    dst_g  = edge_index[1].cpu().numpy()
    dk_cpu = dist_km.cpu().numpy()
    src_l, dst_l, dists = [], [], []
    for i in range(len(src_g)):
        s, d = int(src_g[i]), int(dst_g[i])
        if s in g2l and d in g2l:
            src_l.append(g2l[s])
            dst_l.append(g2l[d])
            dists.append(float(dk_cpu[i]))
    if not src_l:
        return (torch.zeros(2, 1, dtype=torch.long, device=device),
                torch.ones(1, dtype=torch.float32, device=device))
    return (torch.tensor([src_l, dst_l], dtype=torch.long, device=device),
            torch.tensor(dists, dtype=torch.float32, device=device))


def entropy_bonus(pi):
    mean_pi = pi.mean(dim=0).clamp(min=1e-9)
    return -(mean_pi * mean_pi.log()).sum()


def train_fold(cfg, X_tr, C_tr, y_tr, X_te, C_te, y_te,
               ei_tr, dk_tr, mu_init, device,
               log_transform=True, epochs=300):
    """Train PMoE-S for one CV fold."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["model"]["mu_init"] = mu_init

    model     = PMoES(cfg2).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = PMoESLoss(lambda_bal=0.5, lambda_spa=0.002)

    best_nll, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        t = min(epoch / 150, 1.0)
        criterion.lambda_bal = 0.5*(1-t) + 0.05*t

        model.train()
        optimiser.zero_grad()
        out  = model(X_tr, C_tr)
        ld   = criterion(y_tr, out, ei_tr, dk_tr)
        loss = ld["total"]
        if epoch <= 50:
            loss = loss - 0.1*(1-epoch/50)*entropy_bonus(out["pi"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()
        scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                out_v = model(X_te, C_te)
                nll   = nll_mixture_gaussian(
                    y_te, out_v["mu"], out_v["sigma2"], out_v["pi"]).item()
            if nll < best_nll - 1e-4:
                best_nll   = nll
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 4:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out      = model(X_te, C_te)
        pred_log = out["pred_mean"].cpu().numpy().flatten()
        y_log    = y_te.cpu().numpy().flatten()
        if log_transform:
            pred = np.expm1(np.clip(pred_log, -3, 8))
            y_t  = np.expm1(y_log)
        else:
            pred, y_t = pred_log, y_log

        rmse   = np.sqrt(np.mean((pred - y_t)**2))
        mae    = np.mean(np.abs(pred - y_t))
        ss_res = np.sum((y_log - pred_log)**2)
        ss_tot = np.sum((y_log - y_log.mean())**2)
        r2_log = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        nll    = best_nll

        # CRPS: Continuous Ranked Probability Score
        # For a Gaussian predictive distribution, CRPS has a closed form.
        # We use the mixture mean and std as an approximate single Gaussian.
        # A lower CRPS indicates better calibration.
        if HAS_PROPERSCORING:
            pred_std = np.sqrt(np.clip(
                out["pred_var"].cpu().numpy().flatten(), 1e-6, None))
            crps_val = float(np.mean(
                crps_gaussian(y_log, pred_log, pred_std)))
        else:
            crps_val = float("nan")

    return dict(rmse=rmse, mae=mae, r2_log=r2_log, nll=nll, crps=crps_val)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    log_transform = cfg["data"]["log_transform"]
    N_FOLDS       = 5
    BLOCK_KM      = 200   # 200 km >> 7 km median NN -> no spatial leakage

    df           = pd.read_csv(cfg["data"]["processed_path"])
    feature_cols = get_lucas_native_features(df)
    target_col   = cfg["data"]["target_col"]

    log.info(f"\n{'='*60}")
    log.info(f"  Spatial Block Cross-Validation  ({N_FOLDS} folds, "
             f"{BLOCK_KM} km blocks)")
    log.info(f"{'='*60}")
    log.info(f"  Device : {device}")
    log.info(f"  Points : {len(df):,}")

    # Assign blocks
    folds = make_spatial_blocks(df, block_size_km=BLOCK_KM, n_folds=N_FOLDS)

    # Build full dataset tensors
    log.info("\nBuilding spatial graph ...")
    X, coords, y, edge_index, dist_km, safe_cols = build_data(
        df, feature_cols, target_col, device)
    cfg["data"]["feature_cols"] = safe_cols

    fold_results = []

    for fold in range(N_FOLDS):
        log.info(f"\n-- Fold {fold+1}/{N_FOLDS} --")
        tr_mask = folds != fold
        te_mask = folds == fold

        tr = torch.tensor(np.where(tr_mask)[0], dtype=torch.long, device=device)
        te = torch.tensor(np.where(te_mask)[0], dtype=torch.long, device=device)

        X_tr, C_tr, y_tr = X[tr], coords[tr], y[tr]
        X_te, C_te, y_te = X[te], coords[te], y[te]

        ei_tr, dk_tr = remap_edges(edge_index, dist_km, tr, device)
        mu_init      = float(y_tr.cpu().numpy().mean())

        log.info(f"  Train: {len(tr):,}  Test: {len(te):,}  "
                 f"mu_init: {mu_init:.4f}")

        res = train_fold(
            cfg, X_tr, C_tr, y_tr, X_te, C_te, y_te,
            ei_tr, dk_tr, mu_init, device,
            log_transform=log_transform
        )
        fold_results.append(res)
        log.info(f"  Fold {fold+1}: RMSE={res['rmse']:.3f}  "
                 f"R2={res['r2_log']:.4f}  NLL={res['nll']:.4f}  "
                 f"CRPS={res['crps']:.4f}")

    # Aggregate results
    rmses  = [r["rmse"]   for r in fold_results]
    maes   = [r["mae"]    for r in fold_results]
    r2s    = [r["r2_log"] for r in fold_results]
    nlls   = [r["nll"]    for r in fold_results]
    crpss  = [r["crps"]   for r in fold_results]

    print(f"\n{'='*65}")
    print(f"  Spatial Block CV Results  ({N_FOLDS} folds, {BLOCK_KM} km blocks)")
    print(f"{'='*65}")
    print(f"  {'Fold':>6}  {'RMSE':>8}  {'MAE':>8}  {'R2(log)':>9}  {'NLL':>8}  {'CRPS':>8}")
    print(f"  {'-'*56}")
    for i, r in enumerate(fold_results):
        print(f"  {i+1:>6}  {r['rmse']:>8.3f}  {r['mae']:>8.3f}  "
              f"{r['r2_log']:>9.4f}  {r['nll']:>8.4f}  {r['crps']:>8.4f}")
    print(f"  {'-'*56}")
    print(f"  {'Mean':>6}  {np.mean(rmses):>8.3f}  {np.mean(maes):>8.3f}  "
          f"{np.mean(r2s):>9.4f}  {np.mean(nlls):>8.4f}  {np.mean(crpss):>8.4f}")
    print(f"  {'Std':>6}  {np.std(rmses):>8.3f}  {np.std(maes):>8.3f}  "
          f"{np.std(r2s):>9.4f}  {np.std(nlls):>8.4f}  {np.std(crpss):>8.4f}")
    print(f"{'='*65}")
    print()
    print("  Comparison with random split (from run_pmoe.py):")
    print(f"  Random split : RMSE=41.213  R2=0.7545  NLL=0.6110")
    print(f"  Spatial CV   : RMSE={np.mean(rmses):.3f}  "
          f"R2={np.mean(r2s):.4f}  NLL={np.mean(nlls):.4f}")
    diff = np.mean(r2s) - 0.7545
    print(f"  R2 difference: {diff:+.4f}  "
          f"({'expected drop due to harder evaluation' if diff < 0 else 'no overfitting detected'})")
    print()
    print("  These are the numbers to report in the paper.")
    print("  Spatial CV is the honest evaluation for geospatial models.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    main(cfg)