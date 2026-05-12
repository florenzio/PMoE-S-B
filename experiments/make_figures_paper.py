import sys, yaml, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy import stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.gridliner as cgridliner
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loading import get_lucas_native_features
from src.pmoe_bayesian import PMoESBayesian

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.linewidth":    0.6,
})

EXPERT_COLOURS = [
    "#E69F00","#56B4E9","#009E73","#F0E442",
    "#0072B2","#D55E00","#CC79A7","#999999",
]

# Map constants 
PROJ      = ccrs.PlateCarree()   # WGS84 / EPSG:4326
# Square-ish extent: lon range = 70°, lat range = 46° → pad lat to 70°
# centred at ~49°N  → [14, 49] with ±35 = [14-35, 14+35] for lat
EXTENT_SQ = [-25, 45, 20, 76]   # 70° × 56°  (approx square in PlateCarree)
EXTENT    = EXTENT_SQ
DATA_CRS  = ccrs.PlateCarree()
LAND_COL  = "#f0f0f0"
OCEAN_COL = "#d0e8f5"


# Map helpers 

def add_map_features(ax, lon_step=20, lat_step=20, label_size=5.5,
                     graticule_labels=True):
    """TGRS-standard: PlateCarree, blue ocean, grey land, black frame."""
    ax.set_extent(EXTENT, crs=DATA_CRS)

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "ocean", "50m",
        facecolor=OCEAN_COL, edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "50m",
        facecolor=LAND_COL, edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_countries", "50m",
        facecolor="none", edgecolor="#555555",
        linewidth=0.35, zorder=2))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        facecolor="none", edgecolor="#333333",
        linewidth=0.5, zorder=2))

    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(0.7)

    lons = np.arange(-20, 46, lon_step)
    lats = np.arange(20, 77, lat_step)
    gl = ax.gridlines(
        crs=DATA_CRS, draw_labels=graticule_labels,
        xlocs=lons, ylocs=lats,
        linewidth=0.3, color="#aaaaaa", alpha=0.8,
        linestyle="--", zorder=1)
    if graticule_labels:
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": label_size, "color": "black",
                           "fontfamily": "sans-serif"}
        gl.ylabel_style = {"size": label_size, "color": "black",
                           "fontfamily": "sans-serif"}
        gl.xformatter   = cgridliner.LONGITUDE_FORMATTER
        gl.yformatter   = cgridliner.LATITUDE_FORMATTER


def add_scale_bar_zebra(ax, length_km=1000, location=(0.04, 0.04),
                        n_segments=4, bar_height=0.012, fontsize=6.0):
    """Alternating black/white scale bar, SI units, no thousands separator."""
    km_per_deg = 111.32 * np.cos(np.radians(50.0))
    bar_deg    = length_km / km_per_deg
    seg_deg    = bar_deg / n_segments

    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    xr_  = xr - xl;  yr_ = yt - yb

    x0 = xl + location[0] * xr_
    y0 = yb + location[1] * yr_
    bh = bar_height * yr_

    for i in range(n_segments):
        ax.add_patch(plt.Rectangle(
            (x0 + i*seg_deg, y0), seg_deg, bh,
            facecolor="black" if i%2==0 else "white",
            edgecolor="black", linewidth=0.6,
            transform=DATA_CRS, zorder=11, clip_on=False))

    half = length_km // 2
    for tx, lb in [(x0, "0"),
                   (x0 + bar_deg/2, f"{half}"),
                   (x0 + bar_deg,   f"{length_km} km")]:
        ax.text(tx, y0 + bh + 0.4, lb,
                ha="center", va="bottom", fontsize=fontsize,
                color="black", fontfamily="sans-serif",
                transform=DATA_CRS, zorder=12)


def add_north_arrow(ax, location=(0.06, 0.15), size=0.075, fontsize=7.5):
    """Four-pointed cartographic north arrow (black N / white S / grey E,W)."""
    trans = ax.transAxes
    x, y  = location
    h, w  = size, size*0.32
    ws, hs = size*0.15, size*0.25

    for verts, fc in [
        ([(x, y+h), (x-w, y), (x, y+h*0.18), (x+w, y)], "black"),   # N
        ([(x, y-h), (x-w, y), (x, y-h*0.18), (x+w, y)], "white"),   # S
        ([(x+hs, y), (x, y+ws), (x+hs*0.25, y), (x, y-ws)], "#888"),# E
        ([(x-hs, y), (x, y+ws), (x-hs*0.25, y), (x, y-ws)], "#888"),# W
    ]:
        ec = "black"
        ax.add_patch(MplPolygon(verts, closed=True,
                                facecolor=fc, edgecolor=ec, linewidth=0.5,
                                transform=trans, zorder=20, clip_on=False))

    ax.add_patch(plt.Circle((x, y), size*0.07,
                             facecolor="white", edgecolor="black",
                             linewidth=0.5, transform=trans,
                             zorder=21, clip_on=False))
    ax.text(x, y+h+0.035, "N", ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            fontfamily="sans-serif", transform=trans,
            zorder=22, clip_on=False)


def _add_cbar(fig, sc, ax, label, fontsize_label=6, fontsize_tick=5,
              ticks=None, cbar_width="6%", cbar_pad=1.03):
    """
    Vertical colorbar anchored to the exact drawn bbox of ax.
    cbar_pad : x-offset of colorbar left edge in axes fraction (>1 = right of ax)
    """
    cax = inset_axes(ax,
                     width=cbar_width, height="100%",
                     loc="lower left",
                     bbox_to_anchor=(cbar_pad, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    kw = dict(cax=cax, orientation="vertical", extend="neither")
    if ticks is not None:
        kw["ticks"] = ticks
    cbar = fig.colorbar(sc, **kw)
    cbar.set_label(label, fontsize=fontsize_label,
                   fontfamily="sans-serif", labelpad=3)
    cbar.ax.tick_params(labelsize=fontsize_tick, width=0.3,
                        length=2, direction="in")
    cbar.outline.set_linewidth(0.4)
    return cbar


# Data loader 
def load_model_outputs(cfg_path):
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
                     dtype=torch.float32, device=device)

    cfg["model"]["mu_init"]   = float(y.mean())
    cfg["model"]["prior_var"] = 1.0
    model = PMoESBayesian(cfg).to(device)
    model.load_state_dict(torch.load("data/best_pmoe_bayesian.pt",
                          map_location=device, weights_only=False))
    model.eval()
    with torch.no_grad():
        out = model(X, C, n_samples=50)

    return (df.lat.values, df.lon.values,
            out["pi"].cpu().numpy(),
            out["pred_mean"].cpu().numpy().flatten(),
            out["var_ale"].cpu().numpy().flatten(),
            out["var_epi"].cpu().numpy().flatten(),
            y.cpu().numpy(),
            cfg["model"]["num_experts"])


# Fig 2: Uncertainty maps

def fig2_uncertainty(lats, lons, var_ale, var_epi, out_path):
    if not HAS_CARTOPY:
        print("  Skipping Fig 2: cartopy not available"); return

    std_ale = np.sqrt(np.clip(var_ale, 0, None))
    std_epi = np.sqrt(np.clip(var_epi, 0, None))

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5),
        subplot_kw={"projection": PROJ},
        gridspec_kw={
            "wspace":  0.28,   # more horizontal space between panels
            "left":    0.04,
            "right":   0.88,   # leave room for right colorbar
        },
    )

    panels = [
        (std_ale, "YlOrRd",
         r"$\hat{\sigma}_{\mathrm{ale}}$ (std, log-SOC)",
         "Aleatoric uncertainty (intrinsic soil variability)"),
        (std_epi, "Blues",
         r"$\hat{\sigma}_{\mathrm{epi}}$ (std, log-SOC)",
         "Epistemic uncertainty (data scarcity)"),
    ]

    for col, (vals, cmap, cbar_lbl, title) in enumerate(panels):
        ax = axes[col]
        add_map_features(ax, lon_step=20, lat_step=20, label_size=6)
        vmax = np.percentile(vals, 98)
        sc   = ax.scatter(lons, lats, c=vals,
                          cmap=cmap, vmin=0, vmax=vmax,
                          s=2.5, alpha=0.88, linewidths=0,
                          transform=DATA_CRS, zorder=3, rasterized=True)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        _add_cbar(fig, sc, ax, cbar_lbl,
                  fontsize_label=7, fontsize_tick=6,
                  cbar_width="5%", cbar_pad=1.025)
        add_scale_bar_zebra(ax, length_km=1000,
                            location=(0.03, 0.04), fontsize=6.0)
        add_north_arrow(ax, location=(0.07, 0.14),
                        size=0.06, fontsize=7.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", format="pdf")
    plt.savefig(out_path.replace(".pdf",".png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# Fig 4: Architecture 

def fig4_architecture(out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, color, label, sublabel=None, fs=7.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x,y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="#333333",
            linewidth=0.8, zorder=3))
        ax.text(x+w/2, y+h/2+(0.08 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x+w/2, y+h/2-0.18, sublabel,
                    ha="center", va="center",
                    fontsize=6, color="#444444", zorder=4)

    def arrow(x1,y1,x2,y2,lbl=None):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->",
                                   color="#333333", lw=0.8))
        if lbl:
            ax.text((x1+x2)/2,(y1+y2)/2+0.12, lbl,
                    ha="center", va="bottom",
                    fontsize=5.5, color="#555555")

    box(0.1,1.5,1.3,1.0,"#EEF4FF","Input",
        "$\\mathbf{x}_i\\!\\in\\!\\mathbb{R}^{84}$\n"
        "$\\mathbf{s}_i\\!\\in\\!\\mathbb{R}^2$")
    box(1.8,2.1,1.4,0.7,"#FFF3CD","Gating MLP","3-layer, 128 units")
    arrow(1.4,2.0,1.8,2.45,"$[\\mathbf{x}_i;\\bar{\\mathbf{s}}_i]$")
    box(3.6,2.1,1.2,0.7,"#FFE0B2","Sparsemax",
        "$\\boldsymbol{\\pi}_i\\in\\Delta^K$")
    arrow(3.2,2.45,3.6,2.45,"$\\mathbf{g}_i$")
    ax.text(4.2,1.7,"$\\mathcal{L}_{\\mathrm{spa}}$",
            ha="center",va="center",fontsize=7,color="#D55E00",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2",facecolor="#FFF0E8",
                      edgecolor="#D55E00",linewidth=0.6))
    for i,ey in enumerate([3.1,2.55,2.0,1.45,0.9]):
        box(5.2,ey,1.3,0.4,EXPERT_COLOURS[i%8]+"44",
            f"Expert {i+1}" if i<4 else "...",
            "BayesLinear ×3", fs=6.5)
        arrow(4.8,2.45,5.2,ey+0.2,
              "$\\pi_{ik}$" if i==0 else None)
    ax.text(5.85,0.55,"$K=8$ experts",
            ha="center",fontsize=6,color="#555555")
    box(7.0,1.75,1.4,1.4,"#E8F5E9","MoG Output",
        "$p(y|\\mathbf{x},\\mathbf{s})$")
    for ey in [3.1,2.55,2.0,1.45,0.9]:
        arrow(6.5,ey+0.2,7.0,2.45,
              "$(\\mu_k,\\sigma_k^2)$" if ey==3.1 else None)
    box(8.8,2.3,1.1,0.55,"#FCE4EC","Aleatoric",
        "$\\hat\\sigma^2_{\\mathrm{ale}}$")
    box(8.8,1.6,1.1,0.55,"#E3F2FD","Epistemic",
        "$\\hat\\sigma^2_{\\mathrm{epi}}$")
    arrow(8.4,2.45,8.8,2.57); arrow(8.4,2.2,8.8,1.87)
    ax.text(5.0,3.65,
            "PMoE-S-B: Probabilistic MoE with Spatial Regularisation (Bayesian)",
            ha="center",fontsize=9,fontweight="bold")

    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", format="pdf")
    plt.savefig(out_path.replace(".pdf",".png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# Fig 5: Routing weights (square mini-maps) 

def fig5_routing_weights(lats, lons, pi, K, out_path):
    if not HAS_CARTOPY:
        print("  Skipping Fig 5: cartopy not available"); return

    active = sorted(
        [(k, pi[:,k].sum()) for k in range(K)
         if (pi[:,k]>0.01).sum()>100],
        key=lambda x: -x[1])

    fig = plt.figure(figsize=(15, 6.4))

    gs_top = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.03, right=0.97,
        top=0.97, bottom=0.52,
        wspace=0.18)             
    gs_bot = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.16, right=0.84,
        top=0.48, bottom=0.03,
        wspace=0.18)

    axes_top = [fig.add_subplot(gs_top[0,c], projection=PROJ)
                for c in range(4)]
    axes_bot = [fig.add_subplot(gs_bot[0,c], projection=PROJ)
                for c in range(3)]
    all_axes = axes_top + axes_bot

    for idx, (k, _) in enumerate(active[:7]):
        ax = all_axes[idx]
        add_map_features(ax, lon_step=20, lat_step=20,
                         label_size=5.0, graticule_labels=True)

        weights = pi[:, k]
        n_dom   = int((pi.argmax(1)==k).sum())

        sc = ax.scatter(lons, lats, c=weights,
                        cmap="YlOrRd", vmin=0, vmax=1,
                        s=2.5, alpha=0.85, linewidths=0,
                        transform=DATA_CRS, zorder=3, rasterized=True)

        ax.set_title(
            f"Expert {k+1}  ($n_{{\\mathrm{{dom}}}}={n_dom}$)",
            fontsize=8, pad=3, fontweight="bold",
            fontfamily="sans-serif")

        cbar = _add_cbar(
            fig, sc, ax,
            label=r"$\pi_{:k}$ [0, 1]",
            fontsize_label=6, fontsize_tick=5,
            ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.set_yticklabels(
            ["0.00","0.25","0.50","0.75","1.00"], fontsize=5)

        add_scale_bar_zebra(ax, length_km=1000,
                            location=(0.03, 0.04),
                            n_segments=4, bar_height=0.016,
                            fontsize=5.0)
        add_north_arrow(ax, location=(0.08, 0.17),
                        size=0.07, fontsize=6.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white", format="pdf")
    plt.savefig(out_path.replace(".pdf",".png"),
                dpi=300, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# Fig 6: Calibration 

def fig6_calibration(y_true, pred_mean, var_ale, var_epi, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    ax = axes[0]
    pred_std = np.sqrt(np.clip(var_ale+var_epi, 1e-6, None))
    pit      = stats.norm.cdf(y_true, loc=pred_mean, scale=pred_std)
    ax.hist(pit, bins=20, density=True,
            color="#56B4E9", edgecolor="white",
            linewidth=0.4, alpha=0.85, zorder=3)
    ax.axhline(1.0, color="#D55E00", linewidth=1.0,
               linestyle="--", label="Ideal uniform", zorder=4)
    ax.set_xlabel("Probability integral transform (PIT)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("Calibration: PIT histogram\n(PMoE-S-B, full dataset)",
                 fontsize=8, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(0,1); ax.set_ylim(0,2.2)
    ax.spines[["top","right"]].set_visible(False)
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    ax.text(0.97,0.97,
            f"KS stat = {ks_stat:.3f}\n$p$ = {ks_p:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#aaaaaa", linewidth=0.5))

    ax2    = axes[1]
    models = ["PMoE-S-B\n(Bayesian)","PMoE-S\n(det.)",
              "Sparse GP","DKL"]
    cmean  = [0.252, 0.248, 0.253, 0.2527]
    cstd   = [0.018, 0.009, 0.0150, 0.0149]
    cols   = ["#56B4E9","#009E73","#E69F00","#CC79A7"]
    bars   = ax2.bar(models, cmean, yerr=cstd, color=cols,
                     edgecolor="#333333", linewidth=0.5, width=0.55,
                     capsize=4, error_kw=dict(lw=0.8), zorder=3)
    ax2.set_ylabel("CRPS (log-SOC scale, $\\downarrow$)", fontsize=8)
    ax2.set_title("CRPS comparison\n(spatial block CV, 5 folds)",
                  fontsize=8, fontweight="bold")
    ax2.set_ylim(0.22, 0.275)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.yaxis.grid(True, linewidth=0.3, color="#dddddd", zorder=0)
    for bar, m, s in zip(bars, cmean, cstd):
        ax2.text(bar.get_x()+bar.get_width()/2, m+s+0.001,
                 f"{m:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold")

    plt.tight_layout(w_pad=2.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", format="pdf")
    plt.savefig(out_path.replace(".pdf",".png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


def main(cfg_path):
    (lats, lons, pi, pred_mean,
     var_ale, var_epi, y_true, K) = load_model_outputs(cfg_path)

    out_dir = Path("data/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig2_uncertainty(lats, lons, var_ale, var_epi,
                     str(out_dir/"fig2_uncertainty.pdf"))

    fig4_architecture(str(out_dir/"fig4_architecture.pdf"))

    if HAS_CARTOPY:
        fig5_routing_weights(lats, lons, pi, K,
                             str(out_dir/"fig5_routing_weights.pdf"))

    fig6_calibration(y_true, pred_mean, var_ale, var_epi,
                     str(out_dir/"fig6_calibration.pdf"))



if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    main(cfg)