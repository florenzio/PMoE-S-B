# spatial cv ablation p/ versão bayesiana
# NOTE: feito depois da versao deterministica, para correr o ablation com cv espacial

import sys, yaml, copy
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
from src.pmoe_bayesian import PMoESBayesian
from src.pmoe import PMoES
from src.pmoe_bayesian_loss import PMoESBayesianLoss, nll_mixture_gaussian
from src.pmoe_loss import PMoESLoss

# crps é opcional, n vale a pena crashar por causa disto
try:
    from properscoring import crps_gaussian
    HAS_PS = True
except ImportError:
    HAS_PS = False


def make_spatial_blocks(df, block_size_km=200, n_folds=5, seed=42):

    # aprox simples
    lat_step = block_size_km / 111.0
    lon_step = block_size_km / (111.0 * np.cos(np.radians(df["lat"].mean())))

    # bins espaciais
    lat_bin  = ((df["lat"].values - df["lat"].min()) / lat_step).astype(int)
    lon_bin  = ((df["lon"].values - df["lon"].min()) / lon_step).astype(int)

    # id meio cursed mas funciona 
    cell_id  = lat_bin * 1000 + lon_bin
    unique   = np.unique(cell_id)
    rng      = np.random.default_rng(seed)
    rng.shuffle(unique)

    # cell -> fold
    c2f      = {c: i % n_folds for i, c in enumerate(unique)}
    return np.array([c2f[c] for c in cell_id])


def build_data(df, feature_cols, target_col, device):

    safe_cols = [c for c in feature_cols if c != "N"]
    # lc_* sao categóricas one-hot
    cont_cols = [c for c in safe_cols
                 if not c.startswith("LC_") and c not in ("lat","lon")]
    df2 = df.copy()
    # std normal
    for col in cont_cols:
        m, s     = df2[col].mean(), df2[col].std()
        df2[col] = (df2[col] - m) / s if s > 0 else 0.0
    # coords entre [-1,1]
    for col, mn, mx in [("lat", df["lat"].min(), df["lat"].max()),("lon", df["lon"].min(), df["lon"].max()),]:
        df2[f"{col}_n"] = 2*(df[col]-mn)/(mx-mn)-1

    X = torch.tensor(df2[safe_cols].astype(float).values, dtype=torch.float32, device=device)

    coords = torch.tensor(df2[["lat_n","lon_n"]].values, dtype=torch.float32, device=device)

    y = torch.tensor(df2[target_col].values, dtype=torch.float32, device=device).unsqueeze(1)

    # grafo espacial p loss smoothness
    coords_rad        = np.radians(df[["lat","lon"]].values)
    tree              = BallTree(coords_rad, metric="haversine")
    # 16 vizinhos + self
    dist_rad, indices = tree.query(coords_rad, k=17)
    dist_km_mat       = dist_rad[:, 1:] * 6371.0
    indices           = indices[:, 1:]
    src_l, dst_l, dist_l = [], [], []

    for i in range(len(df)):
    # vectorizar depois...
        for j in range(16):
            jj = int(indices[i, j])
            d  = float(dist_km_mat[i, j])
            # cutoff espacial
            if d < 200:
                src_l.append(i)
                dst_l.append(jj)
                dist_l.append(d)

    edge_index = torch.tensor(
        [src_l, dst_l],
        dtype=torch.long,
        device=device)

    dist_km = torch.tensor(
        dist_l,
        dtype=torch.float32,
        device=device)
    return X, coords, y, edge_index, dist_km, safe_cols


def remap_edges(edge_index, dist_km, node_idx, device):

    # global -> local idx
    g2l = {int(g): l for l, g in enumerate(node_idx.cpu().numpy())}
    src_g  = edge_index[0].cpu().numpy()
    dst_g  = edge_index[1].cpu().numpy()
    dk_cpu = dist_km.cpu().numpy()
    src_l, dst_l, dists = [], [], []

    for i in range(len(src_g)):
        s, d = int(src_g[i]), int(dst_g[i])

        # manter só edges do subset atual
        if s in g2l and d in g2l:
            src_l.append(g2l[s])
            dst_l.append(g2l[d])
            dists.append(float(dk_cpu[i]))

    # edge case raro
    if not src_l:
        return (
            torch.zeros(2,1,dtype=torch.long,device=device),
            torch.ones(1,dtype=torch.float32,device=device))

    return (
        torch.tensor([src_l, dst_l], dtype=torch.long, device=device),
        torch.tensor(dists, dtype=torch.float32, device=device))


def entropy_bonus(pi):

    # evita collapse cedo demais
    m = pi.mean(dim=0).clamp(min=1e-9)
    return -(m * m.log()).sum()


def train_fold_bayesian(
    cfg,
    X_tr,
    C_tr,
    y_tr,
    X_te,
    C_te,
    y_te,
    ei_tr,
    dk_tr,
    mu_init,
    device,
    lambda_bal_s,
    lambda_bal_e,
    lambda_spa,
    log_transform=True,
    epochs=300
):

    cfg2 = copy.deepcopy(cfg)
    # init melhor q random puro
    cfg2["model"]["mu_init"]   = mu_init
    cfg2["model"]["prior_var"] = 1.0

    model = PMoESBayesian(cfg2).to(device)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=epochs
    )

    N_TRAIN   = len(X_tr)
    KL_WARMUP = 100

    criterion = PMoESBayesianLoss(
        lambda_bal=lambda_bal_s,
        lambda_spa=lambda_spa,
        beta=0.0,
        n_data=N_TRAIN
    )

    best_nll      = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, epochs+1):
        # warmup kl senao explode logo no inicio
        criterion.beta = min(epoch / KL_WARMUP, 1.0)

        # decay gradual do balancing
        criterion.lambda_bal = (
            lambda_bal_s*(1-min(epoch/150,1))
            + lambda_bal_e*min(epoch/150,1)
        )

        model.train()
        optimiser.zero_grad()
        out  = model(X_tr, C_tr, n_samples=1)
        ld   = criterion(y_tr, out, ei_tr, dk_tr)
        loss = ld["total"]

        # pequeno empurrao p experts nao colapsarem
        if epoch <= 50:
            loss = loss - 0.1*(1-epoch/50)*entropy_bonus(out["pi"])

        loss.backward()

        # sem isto dava nan as vezes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimiser.step()
        scheduler.step()

        # val a cada 20 epochs pq isto e mto lento 
        if epoch % 20 == 0:

            model.eval()

            with torch.no_grad():

                out_v = model(X_te, C_te, n_samples=10)

                nll = nll_mixture_gaussian(
                    y_te,
                    out_v["mu"],
                    out_v["sigma2"],
                    out_v["pi"]
                ).item()

            # tiny margin p evitar ruido
            if nll < best_nll - 1e-4:

                best_nll = nll

                best_state = {
                    k: v.clone()
                    for k,v in model.state_dict().items()
                }

                patience_cnt = 0

            else:

                patience_cnt += 1

                # early stop meio agressivo mas ok
                if patience_cnt >= 4:
                    break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        out = model(X_te, C_te, n_samples=30)

        pred_log = out["pred_mean"].cpu().numpy().flatten()
        y_log    = y_te.cpu().numpy().flatten()

        if log_transform:

            # clamp pq exp() pode fazer porcaria
            pred = np.expm1(np.clip(pred_log, -3, 8))
            y_t  = np.expm1(y_log)

        else:
            pred, y_t = pred_log, y_log

        rmse = np.sqrt(np.mean((pred - y_t)**2))

        # r2 no espaço log
        r2_log = (
            1
            - np.sum((y_log-pred_log)**2)
            / np.sum((y_log-y_log.mean())**2)
        )

        nll = best_nll

        # experts minimamente usados
        active = (out["pi"].mean(0) > 0.01).sum().item()

        if HAS_PS:

            std = np.sqrt(
                np.clip(
                    out["pred_var"].cpu().numpy().flatten(),
                    1e-6,
                    None
                )
            )

            crps_v = float(
                np.mean(crps_gaussian(y_log, pred_log, std))
            )

        else:
            crps_v = float("nan")

    return dict(
        rmse=rmse,
        r2_log=r2_log,
        nll=nll,
        crps=crps_v,
        active=active
    )


def run_variant(
    name,
    cfg,
    X,
    coords,
    y,
    edge_index,
    dist_km,
    folds,
    device,
    log_transform,
    lbs,
    lbe,
    lsp,
    n_folds=5,
    epochs=300
):

    log.info(f"\n{'─'*58}")
    log.info(f"  {name}")
    log.info(f"  λ_bal: {lbs}→{lbe}  λ_spa: {lsp}")
    log.info(f"{'─'*58}")

    results = []

    for fold in range(n_folds):

        # split espacial
        tr = torch.tensor(
            np.where(folds != fold)[0],
            dtype=torch.long,
            device=device
        )

        te = torch.tensor(
            np.where(folds == fold)[0],
            dtype=torch.long,
            device=device
        )

        X_tr, C_tr, y_tr = X[tr], coords[tr], y[tr]
        X_te, C_te, y_te = X[te], coords[te], y[te]

        ei_tr, dk_tr = remap_edges(
            edge_index,
            dist_km,
            tr,
            device
        )

        mu_init = float(y_tr.cpu().numpy().mean())

        res = train_fold_bayesian(
            cfg,
            X_tr,
            C_tr,
            y_tr,
            X_te,
            C_te,
            y_te,
            ei_tr,
            dk_tr,
            mu_init,
            device,
            lbs,
            lbe,
            lsp,
            log_transform,
            epochs
        )

        results.append(res)

        log.info(
            f"  Fold {fold+1}: "
            f"RMSE={res['rmse']:.2f}  "
            f"R2={res['r2_log']:.4f}  "
            f"NLL={res['nll']:.4f}  "
            f"CRPS={res['crps']:.4f}  "
            f"Active={res['active']}/8"
        )

    return results


def summarise(results):

    # media/std rapido
    return {
        k: (
            np.mean([r[k] for r in results]),
            np.std([r[k]  for r in results])
        )
        for k in ["rmse","r2_log","nll","crps","active"]
    }


def main(cfg_path):

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    log_transform = cfg["data"]["log_transform"]

    N_FOLDS  = 5
    BLOCK_KM = 200

    df = pd.read_csv(cfg["data"]["processed_path"])

    feature_cols = get_lucas_native_features(df)

    target_col = cfg["data"]["target_col"]

    log.info(f"\n{'='*58}")
    log.info(f"  PMoE-S-B Ablation w/ Spatial Block CV")
    log.info(f"  ({N_FOLDS} folds, {BLOCK_KM} km blocks)")
    log.info(f"{'='*58}")

    X, coords, y, edge_index, dist_km, safe_cols = build_data(
        df,
        feature_cols,
        target_col,
        device
    )

    cfg["data"]["feature_cols"] = safe_cols

    folds = make_spatial_blocks(
        df,
        BLOCK_KM,
        N_FOLDS,
        cfg["seed"]
    )

    # tirar losses uma a uma
    variants = [
        ("PMoE-S-B (full)",   0.5, 0.05, 0.002),
        ("no-L_spa",          0.5, 0.05, 0.000),
        ("no-L_bal",          0.0, 0.00, 0.002),
    ]

    all_results = {}

    for name, lbs, lbe, lsp in variants:

        results = run_variant(
            name,
            cfg,
            X,
            coords,
            y,
            edge_index,
            dist_km,
            folds,
            device,
            log_transform,
            lbs,
            lbe,
            lsp,
            N_FOLDS
        )

        all_results[name] = summarise(results)


if __name__ == "__main__":

    # default caso corra direto
    cfg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "experiments/config.yaml"
    )

    main(cfg)