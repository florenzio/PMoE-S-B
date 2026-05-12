"""
spatial_cv_bayesian.py
----------------------
Spatial block CV for PMoE-S-B (Bayesian experts).
Same protocol as spatial_cv.py in deterministic
    200 km blocks, 5 folds

"""

import sys, yaml, copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.neighbors import BallTree

# setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import get_lucas_native_features
from src.pmoe_bayesian import PMoESBayesian
from src.pmoe_bayesian_loss import PMoESBayesianLoss, nll_mixture_gaussian

try:
    from properscoring import crps_gaussian
    HAS_PS = True
except ImportError:
    HAS_PS = False

def make_spatial_blocks(df, block_size_km=200, n_folds=5, seed=42):
    # geracao de folds espaciais baseados em blocos km
    lat, lon = df["lat"].values, df["lon"].values
    lat_step = block_size_km / 111.0
    lon_step = block_size_km / (111.0 * np.cos(np.radians(lat.mean())))
    lat_bin = ((lat - lat.min()) / lat_step).astype(int)
    lon_bin = ((lon - lon.min()) / lon_step).astype(int)
    cell_id = lat_bin * 1000 + lon_bin
    unique = np.unique(cell_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    c2f = {c: i % n_folds for i, c in enumerate(unique)}
    return np.array([c2f[c] for c in cell_id])

def build_data(df, feature_cols, target_col, device):
    # preprocessamento e construcao do grafo de vizinhanca
    safe_cols = [c for c in feature_cols if c != "N"]
    df2 = df.copy()
    
    # normalizacao das variaveis continuas
    cont_cols = [c for c in safe_cols if not c.startswith("LC_") and c not in ("lat","lon")]
    for col in cont_cols:
        m, s = df2[col].mean(), df2[col].std()
        df2[col] = (df2[col] - m) / s if s > 0 else 0.0
    
    # normalizacao de coords para o gating
    for col, mn, mx in [("lat", df.lat.min(), df.lat.max()), ("lon", df.lon.min(), df.lon.max())]:
        df2[f"{col}_n"] = 2*(df[col]-mn)/(mx-mn)-1

    X = torch.tensor(df2[safe_cols].astype(float).values, dtype=torch.float32, device=device)
    coords = torch.tensor(df2[["lat_n","lon_n"]].values, dtype=torch.float32, device=device)
    y = torch.tensor(df2[target_col].values, dtype=torch.float32, device=device).unsqueeze(1)

    # haversine para vizinhos ate 200km
    coords_rad = np.radians(df[["lat","lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, indices = tree.query(coords_rad, k=17)
    dist_km_mat, indices = dist_rad[:, 1:] * 6371.0, indices[:, 1:]

    src, dst, dists = [], [], []
    for i in range(len(df)):
        for j_pos in range(16):
            j, d = int(indices[i, j_pos]), float(dist_km_mat[i, j_pos])
            if d < 200:
                src.append(i); dst.append(j); dists.append(d)

    return X, coords, y, torch.tensor([src, dst], dtype=torch.long, device=device), \
           torch.tensor(dists, dtype=torch.float32, device=device), safe_cols

def remap_edges(edge_index, dist_km, node_idx, device):
    # remapeamento de indices globais para locais (por fold)
    g2l = {int(g): l for l, g in enumerate(node_idx.cpu().numpy())}
    src_g, dst_g, dk_cpu = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy(), dist_km.cpu().numpy()
    src_l, dst_l, d_list = [], [], []
    for i in range(len(src_g)):
        s, d = int(src_g[i]), int(dst_g[i])
        if s in g2l and d in g2l:
            src_l.append(g2l[s]); dst_l.append(g2l[d]); d_list.append(float(dk_cpu[i]))
    if not src_l:
        return torch.zeros(2,1,dtype=torch.long,device=device), torch.ones(1,device=device)
    return torch.tensor([src_l, dst_l], dtype=torch.long, device=device), \
           torch.tensor(d_list, dtype=torch.float32, device=device)

def train_fold(cfg, X_tr, C_tr, y_tr, X_te, C_te, y_te, ei_tr, dk_tr, mu_init, device, log_transform=True):
    # setup do modelo bayesiano e loss
    cfg2 = copy.deepcopy(cfg)
    cfg2["model"]["mu_init"] = mu_init
    model = PMoESBayesian(cfg2).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    criterion = PMoESBayesianLoss(lambda_bal=0.5, lambda_spa=0.002, beta=0.0, n_data=len(X_tr))

    best_nll, best_state, pat = float("inf"), None, 0

    for ep in range(1, 301):
        criterion.beta = min(ep / 100, 1.0)
        criterion.lambda_bal = 0.5*(1-min(ep/150,1.0)) + 0.05*min(ep/150,1.0)
        
        model.train(); opt.zero_grad()
        out = model(X_tr, C_tr, n_samples=1)
        loss = criterion(y_tr, out, ei_tr, dk_tr)["total"]
        if ep <= 50: # encorajamento inicial de exploracao de experts
            mean_pi = out["pi"].mean(dim=0).clamp(min=1e-9)
            loss -= 0.1*(1-ep/50) * (-(mean_pi * mean_pi.log()).sum())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sch.step()

        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                ov = model(X_te, C_te, n_samples=10)
                nll = nll_mixture_gaussian(y_te, ov["mu"], ov["sigma2"], ov["pi"]).item()
            if nll < best_nll - 1e-4:
                best_nll, pat = nll, 0
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 4: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(X_te, C_te, n_samples=30)
        p_log, y_log = out["pred_mean"].cpu().numpy().flatten(), y_te.cpu().numpy().flatten()
        pred = np.expm1(np.clip(p_log, -3, 8)) if log_transform else p_log
        y_real = np.expm1(y_log) if log_transform else y_log

        res = {
            "rmse": np.sqrt(np.mean((pred - y_real)**2)),
            "mae": np.mean(np.abs(pred - y_real)),
            "r2_log": 1 - np.sum((y_log - p_log)**2)/np.sum((y_log - y_log.mean())**2),
            "nll": best_nll,
            "mean_ale": float(np.sqrt(out["var_ale"].mean().cpu())),
            "mean_epi": float(np.sqrt(out["var_epi"].mean().cpu()))
        }
        if HAS_PS:
            std = np.sqrt(np.clip(out["pred_var"].cpu().numpy().flatten(), 1e-6, None))
            res["crps"] = float(np.mean(crps_gaussian(y_log, p_log, std)))
        return res

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(cfg["data"]["processed_path"])
    f_cols = get_lucas_native_features(df)
    X, C, y, EI, DK, safe = build_data(df, f_cols, cfg["data"]["target_col"], device)
    cfg["data"]["feature_cols"] = safe

    folds = make_spatial_blocks(df, 200, 5, cfg["seed"])
    fold_results = []

    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        te_idx = np.where(folds == f)[0]
        tr_t, te_t = torch.tensor(tr_idx, device=device), torch.tensor(te_idx, device=device)
        
        ei_tr, dk_tr = remap_edges(EI, DK, tr_t, device)
        mu_init = float(y[tr_t].mean().cpu())

        res = train_fold(cfg, X[tr_t], C[tr_t], y[tr_t], X[te_t], C[te_t], y[te_t], 
                         ei_tr, dk_tr, mu_init, device, cfg["data"]["log_transform"])
        fold_results.append(res)

    return fold_results

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml")