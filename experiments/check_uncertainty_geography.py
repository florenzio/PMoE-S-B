import sys, yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import get_lucas_native_features
from src.pmoe_bayesian import PMoESBayesian


def load_outputs(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df           = pd.read_csv(cfg["data"]["processed_path"])
    feature_cols = get_lucas_native_features(df)
    safe_cols    = [c for c in feature_cols if c != "N"]
    cfg["data"]["feature_cols"] = safe_cols

    df2       = df.copy()
    cont_cols = [c for c in safe_cols
                 if not c.startswith("LC_") and c not in ("lat","lon")]
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

    cfg["model"]["mu_init"]   = float(df[cfg["data"]["target_col"]].mean())
    cfg["model"]["prior_var"] = 1.0

    model = PMoESBayesian(cfg).to(device)
    model.load_state_dict(torch.load("data/best_pmoe_bayesian.pt",
                          map_location=device, weights_only=False))
    model.eval()
    with torch.no_grad():
        out = model(X, coords, n_samples=50)

    lats    = df["lat"].values
    lons    = df["lon"].values
    var_ale = out["var_ale"].cpu().numpy().flatten()
    var_epi = out["var_epi"].cpu().numpy().flatten()
    std_ale = np.sqrt(np.clip(var_ale, 0, None))
    std_epi = np.sqrt(np.clip(var_epi, 0, None))

    return lats, lons, std_ale, std_epi


def region_stats(lats, lons, values, name):
    regions = {
        "Scandinavia/Boreal (lat>58)":
            (lats > 58),
        "Central-Eastern Europe (45<lat<56, 14<lon<30)":
            (lats > 45) & (lats < 56) & (lons > 14) & (lons < 30),
        "Western Balkans (41<lat<46, 14<lon<23)":
            (lats > 41) & (lats < 46) & (lons > 14) & (lons < 23),
        "Mediterranean (lat<42, lon>-5)":
            (lats < 42) & (lons > -5),
        "Atlantic/Western Europe (lat>47, lon<5)":
            (lats > 47) & (lons < 5),
        "Baltic (54<lat<60, 18<lon<30)":
            (lats > 54) & (lats < 60) & (lons > 18) & (lons < 30),
    }

    results = []
    for region_name, mask in regions.items():
        vals = values[mask]
        if len(vals) < 10:
            continue
        results.append((vals.mean(), region_name, len(vals),
                        np.percentile(vals, 75),
                        np.percentile(vals, 90),
                        vals.max()))
    # Rank regions by mean
    results.sort(reverse=True)
    for i, (mean, rname, n, p75, p90, mx) in enumerate(results):
        print(f"    {i+1}. {rname}: {mean:.4f}")
    return results


def main(cfg_path):
    lats, lons, std_ale, std_epi = load_outputs(cfg_path)
    ale_rank = region_stats(lats, lons, std_ale, "Aleatoric uncertainty (std)")
    epi_rank = region_stats(lats, lons, std_epi, "Epistemic uncertainty (std)")

    # tests 
    scand_mask = lats > 58
    ce_mask    = (lats > 45) & (lats < 56) & (lons > 14) & (lons < 30)
    balkans    = (lats > 41) & (lats < 46) & (lons > 14) & (lons < 23)

    # Find top 5% epistemic7aleatoric points and their location
    p95 = np.percentile(std_epi, 95)
    top_mask = std_epi > p95
    print(f"    Lat range: {lats[top_mask].min():.1f} to {lats[top_mask].max():.1f}")
    print(f"    Lon range: {lons[top_mask].min():.1f} to {lons[top_mask].max():.1f}")
    print(f"    Mean lat:  {lats[top_mask].mean():.1f}")
    print(f"    Mean lon:  {lons[top_mask].mean():.1f}")

    p95_ale = np.percentile(std_ale, 95)
    top_ale = std_ale > p95_ale
    print(f"\n  Top 5% aleatoric points (n={top_ale.sum()}):")
    print(f"    Lat range: {lats[top_ale].min():.1f} to {lats[top_ale].max():.1f}")
    print(f"    Lon range: {lons[top_ale].min():.1f} to {lons[top_ale].max():.1f}")
    print(f"    Mean lat:  {lats[top_ale].mean():.1f}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    main(cfg)