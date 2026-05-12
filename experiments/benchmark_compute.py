import sys, yaml, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import get_lucas_native_features
from src.pmoe_bayesian import PMoESBayesian
from src.pmoe import PMoES

# N values to test for scaling benchmark
N_SIZES = [5_000, 10_000, 18_711, 50_000, 100_000]
N_WARMUP  = 3
N_REPEATS = 10


def build_data(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df     = pd.read_csv(cfg["data"]["processed_path"])
    fcols  = get_lucas_native_features(df)
    safe   = [c for c in fcols if c != "N"]
    cfg["data"]["feature_cols"] = safe

    df2 = df.copy()
    for col in [c for c in safe
                if not c.startswith("LC_") and c not in ("lat","lon")]:
        m, s = df2[col].mean(), df2[col].std()
        df2[col] = (df2[col]-m)/s if s > 0 else 0.0
    for col, mn, mx in [("lat", df.lat.min(), df.lat.max()),
                         ("lon", df.lon.min(), df.lon.max())]:
        df2[f"{col}_n"] = 2*(df[col]-mn)/(mx-mn)-1

    X = torch.tensor(df2[safe].astype(float).values,
                     dtype=torch.float32, device=device)
    C = torch.tensor(df2[["lat_n","lon_n"]].values,
                     dtype=torch.float32, device=device)
    y = torch.tensor(df2[cfg["data"]["target_col"]].values,
                     dtype=torch.float32, device=device).unsqueeze(1)
    return X, C, y, cfg, device


def measure_inference(model, X, C, n_samples=None, n_repeats=N_REPEATS):
    """Returns mean inference time in milliseconds."""
    device = X.device
    is_bayesian = hasattr(model, 'experts') and n_samples is not None

    def fwd():
        if is_bayesian:
            return model(X, C, n_samples=n_samples)
        else:
            return model(X, C)

    for _ in range(N_WARMUP):
        with torch.no_grad():
            fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def measure_train_epoch(model, X, C, y, optimizer,
                        n_repeats=5, n_samples=None):
    """Returns mean time per training epoch in seconds."""
    device   = X.device
    is_bayes = n_samples is not None
    times    = []
    for _ in range(n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        if is_bayes:
            out = model(X, C, n_samples=1)
        else:
            out = model(X, C)
        loss = ((out["pred_mean"] - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return float("nan")


def main(cfg_path):
    X, C, y, cfg, device = build_data(cfg_path)
    N = len(X)

    results = {}

    # PMoE-S-B
    cfg["model"]["mu_init"]   = float(y.mean())
    cfg["model"]["prior_var"] = 1.0
    torch.cuda.reset_peak_memory_stats(device)
    model    = PMoESBayesian(cfg).to(device)
    n_params = count_params(model)
    opt      = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    t_train, _ = measure_train_epoch(
        model, X, C, y, opt, n_samples=1)

    model.eval()
    t_inf_1,  _ = measure_inference(model, X, C, n_samples=1)
    t_inf_50, _ = measure_inference(model, X, C, n_samples=50)
    mem = peak_memory_mb(device)

    results["PMoE-S-B"] = dict(
        params=n_params,
        t_train_s=t_train,
        t_inf_ms_1=t_inf_1,
        t_inf_ms_50=t_inf_50,
        mem_mb=mem,
        complexity="$\\mathcal{O}(N)$",
    )

    # PMoE-S (deterministic)
    torch.cuda.reset_peak_memory_stats(device)
    model_det    = PMoES(cfg).to(device)
    n_params_det = count_params(model_det)
    opt_det      = torch.optim.AdamW(model_det.parameters(), lr=1e-3)

    model_det.train()
    t_train_det, _ = measure_train_epoch(
        model_det, X, C, y, opt_det, n_samples=None)  # no n_samples

    model_det.eval()
    t_inf_det, _ = measure_inference(model_det, X, C, n_samples=None)
    mem_det = peak_memory_mb(device)

    results["PMoE-S"] = dict(
        params=n_params_det,
        t_train_s=t_train_det,
        t_inf_ms_1=t_inf_det,
        t_inf_ms_50=t_inf_det,
        mem_mb=mem_det,
        complexity="$\\mathcal{O}(N)$",
    )

    # Sparse GP (SVGP)
    try:
        import gpytorch
        from torch.utils.data import TensorDataset, DataLoader
        # M=500 inducing points, RBF kernel
        M = 500
        inducing = X[:M].clone()

        class SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, ind):
                var = gpytorch.variational.CholeskyVariationalDistribution(ind.size(0))
                strat = gpytorch.variational.VariationalStrategy(
                    self, ind, var, learn_inducing_locations=True)
                super().__init__(strat)
                self.mean = gpytorch.means.ConstantMean()
                self.cov  = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel())
            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean(x), self.cov(x))

        torch.cuda.reset_peak_memory_stats(device)
        gp    = SVGPModel(inducing).to(device)
        lik   = gpytorch.likelihoods.GaussianLikelihood().to(device)
        opt_g = torch.optim.Adam(
            list(gp.parameters()) + list(lik.parameters()), lr=1e-2)
        mll   = gpytorch.mlls.VariationalELBO(lik, gp, num_data=N)

        gp.train(); lik.train()
        t0 = time.perf_counter()
        for _ in range(5):
            opt_g.zero_grad()
            out_ = gp(X)
            loss_ = -mll(out_, y.squeeze())
            loss_.backward()
            opt_g.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gp_train = (time.perf_counter()-t0)/5

        gp.eval(); lik.eval()
        t0 = time.perf_counter()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = lik(gp(X))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gp_inf = (time.perf_counter()-t0)*1000
        mem_gp = peak_memory_mb(device)
        n_gp   = count_params(gp) + count_params(lik)

        results["Sparse GP (SVGP)"] = dict(
            params=n_gp,
            t_train_s=t_gp_train,
            t_inf_ms_1=t_gp_inf,
            t_inf_ms_50=t_gp_inf,
            mem_mb=mem_gp,
            complexity="$\\mathcal{O}(NM^2)$",
        )
    except Exception as e:
        print(f"  Sparse GP failed: {e}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    main(cfg)