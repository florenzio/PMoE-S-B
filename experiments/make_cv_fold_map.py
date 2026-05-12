import sys, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pathlib import Path

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.gridliner as cgridliner
    HAS_CARTOPY = True
except ImportError:
    print("ERROR: pip install cartopy"); sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# Colourblind-friendly palette (5 folds)
FOLD_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
FOLD_LABELS  = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]

DATA_CRS  = ccrs.PlateCarree()
PROJ      = ccrs.PlateCarree()
EXTENT    = [-25, 45, 26, 72]
LAND_COL  = "#f0f0f0"
OCEAN_COL = "#d0e8f5"


def make_spatial_blocks(df, block_size_km=200, n_folds=5, seed=42):
    """Assign each point to a spatial block fold."""
    lat_step = block_size_km / 111.0
    lon_step = block_size_km / (
        111.0 * np.cos(np.radians(df["lat"].mean())))
    lat_bin  = ((df["lat"].values - df["lat"].min()) / lat_step).astype(int)
    lon_bin  = ((df["lon"].values - df["lon"].min()) / lon_step).astype(int)
    cell_id  = lat_bin * 1000 + lon_bin
    unique   = np.unique(cell_id)
    rng      = np.random.default_rng(seed)
    rng.shuffle(unique)
    c2f = {c: i % n_folds for i, c in enumerate(unique)}
    return np.array([c2f[c] for c in cell_id])


def add_scale_bar_zebra(ax, length_km=1000, location=(0.04, 0.04),
                        n_segments=4, bar_height=0.012, fontsize=6.5):
    km_per_deg = 111.32 * np.cos(np.radians(50.0))
    bar_deg    = length_km / km_per_deg
    seg_deg    = bar_deg / n_segments
    xl, xr = ax.get_xlim(); yb, yt = ax.get_ylim()
    xr_ = xr - xl; yr_ = yt - yb
    x0 = xl + location[0] * xr_
    y0 = yb + location[1] * yr_
    bh = bar_height * yr_
    for i in range(n_segments):
        ax.add_patch(plt.Rectangle(
            (x0 + i*seg_deg, y0), seg_deg, bh,
            facecolor="black" if i%2==0 else "white",
            edgecolor="black", linewidth=0.7,
            transform=DATA_CRS, zorder=11, clip_on=False))
    half = length_km // 2
    for tx, lb in [(x0, "0"),
                   (x0 + bar_deg/2, f"{half}"),
                   (x0 + bar_deg,   f"{length_km} km")]:
        ax.text(tx, y0 + bh + 0.4, lb, ha="center", va="bottom",
                fontsize=fontsize, color="black",
                fontfamily="sans-serif",
                transform=DATA_CRS, zorder=12)


def add_north_arrow(ax, location=(0.06, 0.13), size=0.07, fontsize=8):
    from matplotlib.patches import Polygon as MplPolygon
    trans = ax.transAxes
    x, y  = location
    h, w  = size, size*0.32
    ws, hs = size*0.15, size*0.25
    for verts, fc in [
        ([(x,y+h),(x-w,y),(x,y+h*0.18),(x+w,y)],   "black"),
        ([(x,y-h),(x-w,y),(x,y-h*0.18),(x+w,y)],   "white"),
        ([(x+hs,y),(x,y+ws),(x+hs*0.25,y),(x,y-ws)],"#888"),
        ([(x-hs,y),(x,y+ws),(x-hs*0.25,y),(x,y-ws)],"#888"),
    ]:
        ax.add_patch(MplPolygon(verts, closed=True,
                                facecolor=fc, edgecolor="black",
                                linewidth=0.5, transform=trans,
                                zorder=20, clip_on=False))
    ax.add_patch(plt.Circle((x, y), size*0.07,
                             facecolor="white", edgecolor="black",
                             linewidth=0.5, transform=trans,
                             zorder=21, clip_on=False))
    ax.text(x, y+h+0.03, "N", ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            fontfamily="sans-serif", transform=trans,
            zorder=22, clip_on=False)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    df    = pd.read_csv(cfg["data"]["processed_path"])
    folds = make_spatial_blocks(df, block_size_km=200, n_folds=5,
                                seed=cfg.get("seed", 42))

    print(f"  N = {len(df)}")
    for fold in range(5):
        n = (folds == fold).sum()
        print(f"  Fold {fold+1}: {n} points ({100*n/len(df):.1f}%)")

    # Figure 
    fig, ax = plt.subplots(
        figsize=(9, 6.5),
        subplot_kw={"projection": PROJ})

    ax.set_extent(EXTENT, crs=DATA_CRS)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical","ocean","50m",
        facecolor=OCEAN_COL, edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical","land","50m",
        facecolor=LAND_COL, edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural","admin_0_countries","50m",
        facecolor="none", edgecolor="#555555",
        linewidth=0.4, zorder=2))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical","coastline","50m",
        facecolor="none", edgecolor="#333333",
        linewidth=0.5, zorder=2))

    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(0.8)

    # Graticule
    lons = np.arange(-20, 46, 20)
    lats = np.arange(30, 73, 10)
    gl = ax.gridlines(crs=DATA_CRS, draw_labels=True,
                      xlocs=lons, ylocs=lats,
                      linewidth=0.3, color="#aaaaaa",
                      alpha=0.8, linestyle="--", zorder=1)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "black",
                       "fontfamily": "sans-serif"}
    gl.ylabel_style = {"size": 8, "color": "black",
                       "fontfamily": "sans-serif"}
    gl.xformatter   = cgridliner.LONGITUDE_FORMATTER
    gl.yformatter   = cgridliner.LATITUDE_FORMATTER

    # Draw 200 km block grid lines
    lat_step = 200 / 111.0
    lon_step = 200 / (111.0 * np.cos(np.radians(df["lat"].mean())))
    for lat_line in np.arange(26, 72, lat_step):
        ax.plot([-25, 45], [lat_line, lat_line],
                color="#cccccc", linewidth=0.2, linestyle="-",
                transform=DATA_CRS, zorder=1, alpha=0.5)
    for lon_line in np.arange(-25, 45, lon_step):
        ax.plot([lon_line, lon_line], [26, 72],
                color="#cccccc", linewidth=0.2, linestyle="-",
                transform=DATA_CRS, zorder=1, alpha=0.5)

    # Scatter points coloured by fold
    # Subsample for clarity if N is large (show a pp 30pcnt of points)
    rng  = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.30
    for fold in range(5):
        sel = (folds == fold) & mask
        ax.scatter(df["lon"].values[sel],
                   df["lat"].values[sel],
                   c=FOLD_COLOURS[fold],
                   s=6, alpha=0.85, linewidths=0,
                   transform=DATA_CRS, zorder=4,
                   rasterized=True)

    # Legend
    handles = [mpatches.Patch(facecolor=FOLD_COLOURS[i],
                               edgecolor="#333333", linewidth=0.5,
                               label=FOLD_LABELS[i])
               for i in range(5)]
    ax.legend(handles=handles, title="CV Fold",
              title_fontsize=8, fontsize=7.5,
              loc="upper left",
              framealpha=0.92, edgecolor="#aaaaaa",
              borderpad=0.6, handlelength=1.2)

    # Annotation: block size
    ax.text(0.99, 0.01,
            "200 km spatial blocks\n(28$\\times$ median NN distance)",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7,
            color="#333333", fontfamily="sans-serif",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      alpha=0.85, pad=3, linewidth=0.5))

    add_scale_bar_zebra(ax, length_km=1000,
                        location=(0.03, 0.04), fontsize=7)
    add_north_arrow(ax, location=(0.07, 0.13),
                    size=0.06, fontsize=8)

    out_dir = Path("data/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = str(out_dir / "figA1_cv_folds.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight",
                facecolor="white", format="pdf")
    plt.savefig(out_pdf.replace(".pdf",".png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved: {out_pdf}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    main(cfg)