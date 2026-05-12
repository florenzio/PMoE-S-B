import sys, yaml, math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from src.data_loading import get_lucas_native_features
from src.pmoe import PMoES
from src.pmoe_loss import PMoESLoss, nll_mixture_gaussian


def build_data(df, feature_cols, target_col, device):
    safe_cols = [c for c in feature_cols if c != "N"]
    lc_cols   = [c for c in safe_cols if c.startswith("LC_")]
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
        df2[f"{col}_n"] = 2 * (df[col] - mn) / (mx - mn) - 1

    X      = torch.tensor(df2[safe_cols].astype(float).values,
                          dtype=torch.float32, device=device)
    coords = torch.tensor(df2[["lat_n", "lon_n"]].values,
                          dtype=torch.float32, device=device)
    y      = torch.tensor(df2[target_col].values,
                          dtype=torch.float32, device=device).unsqueeze(1)

    log.info("Building spatial kNN graph (k=16) ...")
    coords_rad        = np.radians(df[["lat", "lon"]].values)
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
    log.info(f"  {len(src_list):,} edges  ({len(src_list)/len(df):.1f} per node)")
    return X, coords, y, edge_index, dist_km, safe_cols


def remap_edges(edge_index, dist_km, node_idx, device):
    g2l    = {int(g): l for l, g in enumerate(node_idx.cpu().numpy())}
    src_g  = edge_index[0].cpu().numpy()
    dst_g  = edge_index[1].cpu().numpy()
    dk_cpu = dist_km.cpu().numpy()

    src_local, dst_local, dists = [], [], []
    for i in range(len(src_g)):
        s, d = int(src_g[i]), int(dst_g[i])
        if s in g2l and d in g2l:
            src_local.append(g2l[s])
            dst_local.append(g2l[d])
            dists.append(float(dk_cpu[i]))

    if len(src_local) == 0:
        ei = torch.zeros(2, 1, dtype=torch.long, device=device)
        dk = torch.ones(1, dtype=torch.float32, device=device)
        return ei, dk

    ei = torch.tensor([src_local, dst_local], dtype=torch.long, device=device)
    dk = torch.tensor(dists, dtype=torch.float32, device=device)
    return ei, dk


def entropy_bonus(pi: torch.Tensor) -> torch.Tensor:
    """
    Encourage uniform routing by maximising entropy of mean routing.
    H = -sum_k p_k * log(p_k) where p_k = mean_i pi_k(i)
    Maximising H prevents early expert collapse.
    """
    mean_pi = pi.mean(dim=0).clamp(min=1e-9)
    return -(mean_pi * mean_pi.log()).sum()


def val_metrics(model, X_te, C_te, y_te, log_transform):
    model.eval()
    with torch.no_grad():
        out      = model(X_te, C_te)
        nll      = nll_mixture_gaussian(
                       y_te, out["mu"], out["sigma2"], out["pi"]).item()
        pred_log = out["pred_mean"].cpu().numpy().flatten()
        y_log    = y_te.cpu().numpy().flatten()
        if log_transform:
            pred = np.expm1(np.clip(pred_log, -3, 8))
            y_t  = np.expm1(y_log)
        else:
            pred, y_t = pred_log, y_log

        rmse   = np.sqrt(np.mean((pred - y_t) ** 2))
        ss_res = np.sum((y_log - pred_log) ** 2)
        ss_tot = np.sum((y_log - y_log.mean()) ** 2)
        r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        pi_mean = out["pi"].mean(0)
        active  = (pi_mean > 0.01).sum().item()

    return nll, rmse, r2_log, active, out


def train(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    df            = pd.read_csv(cfg["data"]["processed_path"])
    feature_cols  = get_lucas_native_features(df)
    target_col    = cfg["data"]["target_col"]
    log_transform = cfg["data"]["log_transform"]

    log.info(f"\n{'='*58}")
    log.info(f"  PMoE-S Training  (with curriculum load balancing)")
    log.info(f"{'='*58}")
    log.info(f"  Device  : {device}")
    log.info(f"  Points  : {len(df):,}")
    log.info(f"  Experts : {cfg['model']['num_experts']}")

    X, coords, y, edge_index, dist_km, safe_cols = build_data(
        df, feature_cols, target_col, device)
    cfg["data"]["feature_cols"] = safe_cols

    np.random.seed(cfg["seed"])
    idx   = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    tr    = torch.tensor(idx[:split], dtype=torch.long, device=device)
    te    = torch.tensor(idx[split:], dtype=torch.long, device=device)

    X_tr, C_tr, y_tr = X[tr], coords[tr], y[tr]
    X_te, C_te, y_te = X[te], coords[te], y[te]

    log.info("Remapping edges to local indices ...")
    ei_tr, dk_tr = remap_edges(edge_index, dist_km, tr, device)
    log.info(f"  Training edges: {ei_tr.shape[1]:,}")

    # Compute mu_init from training data
    mu_init = float(y_tr.cpu().numpy().mean())
    log.info(f"  mu_init : {mu_init:.4f}")
    cfg["model"]["mu_init"] = mu_init

    model     = PMoES(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Params  : {n_params:,}\n")

    optimiser = optim.AdamW(model.parameters(),
                            lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=cfg["training"]["epochs"])

    lambda_spa      = cfg["training"]["lambda_spa"]
    lambda_bal_end  = cfg["training"]["lambda_bal"]   # final value (0.05)
    lambda_bal_start = 0.5                            # start high to prevent collapse
    curriculum_epochs = 150                           # ramp down over first 150 epochs

    criterion = PMoESLoss(lambda_bal=lambda_bal_start,
                          lambda_spa=lambda_spa)

    best_nll, patience_cnt = float("inf"), 0
    epochs   = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]

    log.info(f"  lambda_bal: {lambda_bal_start} -> {lambda_bal_end} "
             f"over {curriculum_epochs} epochs (curriculum)")
    log.info(f"  lambda_spa: {lambda_spa}")
    log.info(f"\n{'Epoch':>6}  {'Train NLL':>10}  {'Val NLL':>10}  "
             f"{'R2(log)':>8}  {'RMSE':>8}  {'Active':>8}")
    log.info("-" * 60)

    for epoch in range(1, epochs + 1):

        # Curriculum: decay lambda_bal from start to end
        t         = min(epoch / curriculum_epochs, 1.0)
        lam_bal   = lambda_bal_start * (1 - t) + lambda_bal_end * t
        criterion.lambda_bal = lam_bal

        model.train()
        optimiser.zero_grad()
        out       = model(X_tr, C_tr)
        loss_dict = criterion(y_tr, out, ei_tr, dk_tr)

        # Entropy bonus during warmup (first 50 epochs)
        # Forces diversity early before NLL takes over
        if epoch <= 50:
            ent_weight = 0.1 * (1 - epoch / 50)
            ent_loss   = -entropy_bonus(out["pi"]) * ent_weight
            total_loss = loss_dict["total"] + ent_loss
        else:
            total_loss = loss_dict["total"]

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            nll, rmse, r2_log, active, _ = val_metrics(
                model, X_te, C_te, y_te, log_transform)

            log.info(f"{epoch:>6}  {loss_dict['nll']:>10.4f}  {nll:>10.4f}  "
                     f"{r2_log:>8.4f}  {rmse:>8.2f}  "
                     f"{active:>6}/{cfg['model']['num_experts']}")

            if nll < best_nll - 1e-4:
                best_nll     = nll
                patience_cnt = 0
                Path("data").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "data/best_pmoe.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= patience // 5:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

    # Final evaluation
    log.info("\n-- Loading best model --")
    model.load_state_dict(torch.load("data/best_pmoe.pt",
                                     map_location=device,
                                     weights_only=False))
    _, out_final = val_metrics.__wrapped__ if hasattr(val_metrics, '__wrapped__') \
        else (None, None), None

    model.eval()
    with torch.no_grad():
        out   = model(X_te, C_te)
        pred_log = out["pred_mean"].cpu().numpy().flatten()
        y_log    = y_te.cpu().numpy().flatten()
        if log_transform:
            pred = np.expm1(np.clip(pred_log, -3, 8))
            y_t  = np.expm1(y_log)
        else:
            pred, y_t = pred_log, y_log

        rmse  = np.sqrt(np.mean((pred - y_t) ** 2))
        mae   = np.mean(np.abs(pred - y_t))
        r2    = 1 - np.sum((y_t - pred)**2) / np.sum((y_t - y_t.mean())**2)
        ss    = np.sum((y_log - pred_log)**2)
        ss_t  = np.sum((y_log - y_log.mean())**2)
        r2_log = 1 - ss / ss_t

        var_ale = out["var_ale"].cpu().numpy().flatten()
        var_epi = out["var_epi"].cpu().numpy().flatten()

        pi_mean = out["pi"].mean(0).cpu().numpy()
        active  = (pi_mean > 0.01).sum()


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    train(cfg)