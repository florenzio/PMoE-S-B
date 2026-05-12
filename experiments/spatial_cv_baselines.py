import sys, yaml, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# setup do path para a src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import get_lucas_native_features
from src.pmoe_loss import nll_mixture_gaussian

def make_spatial_blocks(df, block_size_km=200, n_folds=5, seed=42):
    # divisao do espaco em blocos para cv espacial
    lat = df["lat"].values
    lon = df["lon"].values
    lat_step = block_size_km / 111.0
    lon_step = block_size_km / (111.0 * np.cos(np.radians(lat.mean())))
    lat_bin  = ((lat - lat.min()) / lat_step).astype(int)
    lon_bin  = ((lon - lon.min()) / lon_step).astype(int)
    cell_id  = lat_bin * 1000 + lon_bin
    unique_cells = np.unique(cell_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_cells)
    cell_fold = {c: i % n_folds for i, c in enumerate(unique_cells)}
    return np.array([cell_fold[c] for c in cell_id])

class ProbMLP(nn.Module):
    # mlp probabilistica para o deep ensemble
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.1, mu_init=3.3):
        super().__init__()
        dims, d = [], in_dim + 2
        for _ in range(layers - 1):
            dims += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*dims)
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.mu_head.bias, mu_init)
        nn.init.zeros_(self.mu_head.weight)

    def forward(self, x, c):
        h = self.trunk(torch.cat([x, c], dim=-1))
        return self.mu_head(h), F.softplus(self.sigma_head(h)) + 1e-4

def nll_gauss(y, mu, s2):
    # loss de nll para uma unica gaussiana
    return (0.5*math.log(2*math.pi) + 0.5*torch.log(s2) + 0.5*(y-mu)**2/s2).mean()

def train_member(X_tr, C_tr, y_tr, X_te, C_te, y_te, in_dim, mu_init, seed, device, epochs=300):
    # treino de um membro do ensemble com early stopping no nll
    torch.manual_seed(seed)
    m = ProbMLP(in_dim, mu_init=mu_init).to(device)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_nll, best_st, pat = float("inf"), None, 0
    for ep in range(1, epochs+1):
        m.train(); opt.zero_grad()
        mu, s2 = m(X_tr, C_tr)
        nll_gauss(y_tr, mu, s2).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
        opt.step(); sch.step()
        if ep % 20 == 0:
            m.eval()
            with torch.no_grad():
                mv, sv = m(X_te, C_te)
                v = nll_gauss(y_te, mv, sv).item()
            if v < best_nll - 1e-4:
                best_nll, pat = v, 0
                best_st = {k: v2.clone() for k,v2 in m.state_dict().items()}
            else:
                pat += 1
                if pat >= 4: break
    m.load_state_dict(best_st)
    return m

def ensemble_nll(members, X, C, y, device):
    # nll do ensemble tratado como uma gmm
    mus, vars_ = [], []
    for mem in members:
        mem.eval()
        with torch.no_grad():
            mu, s2 = mem(X, C)
            mus.append(mu); vars_.append(s2)
    K = len(members)
    mu, s2 = torch.cat(mus, dim=-1), torch.cat(vars_, dim=-1)
    pi = torch.ones(len(y), K, device=device) / K
    return nll_mixture_gaussian(y, mu, s2, pi).item()

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_transform = cfg["data"]["log_transform"]
    
    df = pd.read_csv(cfg["data"]["processed_path"])
    feat_cols = get_lucas_native_features(df)
    target_col = cfg["data"]["target_col"]
    safe_cols = [c for c in feat_cols if c != "N"]

    # normalisacao de colunas continuas
    cont_cols = [c for c in safe_cols if not c.startswith("LC_") and c not in ("lat","lon")]
    df2 = df.copy()
    for col in cont_cols:
        m, s = df2[col].mean(), df2[col].std()
        df2[col] = (df2[col]-m)/s if s>0 else 0.0
    
    # normalisacao das coords
    for col, mn, mx in [("lat", df.lat.min(), df.lat.max()), ("lon", df.lon.min(), df.lon.max())]:
        df2[f"{col}_n"] = 2*(df[col]-mn)/(mx-mn)-1

    X_np, C_np, y_np = df2[safe_cols].values, df2[["lat_n","lon_n"]].values, df2[target_col].values
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)

    folds = make_spatial_blocks(df, 200, 5, seed=cfg["seed"])
    rf_results, ens_results = [], []

    for fold in range(5):
        tr_mask, te_mask = folds != fold, folds == fold
        tr_idx, te_idx = np.where(tr_mask)[0], np.where(te_mask)[0]

        # rf baseline
        rf = RandomForestRegressor(n_estimators=300, max_features="sqrt", min_samples_leaf=5, random_state=cfg["seed"], n_jobs=-1)
        rf.fit(X_np[tr_idx], y_np[tr_idx])
        pred_rf = rf.predict(X_np[te_idx])
        
        y_te_log = y_np[te_idx]
        y_te_g = np.expm1(y_te_log) if log_transform else y_te_log
        pred_rf_g = np.expm1(np.clip(pred_rf, -3, 8)) if log_transform else pred_rf
        
        rf_results.append({
            "rmse": np.sqrt(np.mean((pred_rf_g - y_te_g)**2)),
            "mae": np.mean(np.abs(pred_rf_g - y_te_g)),
            "r2": 1 - np.sum((y_te_log - pred_rf)**2) / np.sum((y_te_log - y_te_log.mean())**2)
        })

        # deep ensemble baseline
        X_tr_t, C_tr_t, y_tr_t = X_t[tr_idx], C_t[tr_idx], y_t[tr_idx]
        X_te_t, C_te_t, y_te_t = X_t[te_idx], C_t[te_idx], y_t[te_idx]
        mu_init = float(y_tr_t.mean().cpu())

        members = [train_member(X_tr_t, C_tr_t, y_tr_t, X_te_t, C_te_t, y_te_t, X_t.shape[1], mu_init, cfg["seed"]+i, device) for i in range(5)]
        
        mus_e = []
        for m in members:
            m.eval()
            with torch.no_grad():
                mu_e, _ = m(X_te_t, C_te_t)
                mus_e.append(mu_e.cpu().numpy().flatten())
        
        pred_ens = np.stack(mus_e).mean(axis=0)
        pred_ens_g = np.expm1(np.clip(pred_ens, -3, 8)) if log_transform else pred_ens
        
        ens_results.append({
            "rmse": np.sqrt(np.mean((pred_ens_g - y_te_g)**2)),
            "mae": np.mean(np.abs(pred_ens_g - y_te_g)),
            "r2": 1 - np.sum((y_te_log - pred_ens)**2) / np.sum((y_te_log - y_te_log.mean())**2),
            "nll": ensemble_nll(members, X_te_t, C_te_t, y_te_t, device)
        })

    return {"rf": rf_results, "ensemble": ens_results}

if __name__ == "__main__":
    results = main(sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml")