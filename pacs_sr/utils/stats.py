#!/usr/bin/env python3
# sr_stats_plots.py
"""
Create scientific statistics panels for the SR study.

Inputs
------
--stats_npz     Path to sr_stats_summary.npz (v3)
--metrics_npz   Path to metrics.npz (optional; only used for n counts)
--out_dir       Output directory for figures (created if missing)

Outputs
-------
<out_dir>/sr_panel_<pulse>_A.pdf/png   Forest + adjusted means (per pulse)
<out_dir>/sr_panel_<pulse>_B.pdf/png   Wilcoxon heatmap + ICC bars (per pulse)

Notes
-----
- Categorical color mapping uses Paul Tol's Bright palette; BSPLINE is gray.
- Diverging colormap for signed -log10(p) uses a Tol-like blue–gray–red ramp.
- All numbers shown are derived from sr_stats_summary.npz produced by v3.
"""

from __future__ import annotations
import argparse, json, logging, pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib import lines as mlines
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
# ------------------------- logging and style ------------------------------

LOG = logging.getLogger("SR:plots")

def configure_matplotlib() -> None:
    """
    Configure matplotlib and scienceplots with LaTeX and requested typography.
    Falls back gracefully if LaTeX or scienceplots are not available.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])  # base science style
    except Exception as e:
        logging.warning("scienceplots not available: %s. Continuing with default style.", e)

    # Requested typography
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })
    # LaTeX text rendering
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s. Falling back to non-LaTeX text.", e)
        plt.rcParams['text.usetex'] = False

def _forest_items_order(contr_df: pd.DataFrame) -> Tuple[List[Tuple[int, str]], List[str]]:
    """
    Return ordered list of (resolution, model) rows and matching ytick labels
    as used by the forest plot. Excludes the baseline model.
    """
    if contr_df.empty:
        return [], []
    res_order = sorted({int(x) for x in contr_df["resolution"].astype(int).unique()})
    baseline = contr_df["baseline"].dropna().unique()
    baseline = baseline[0] if len(baseline) else "BSPLINE"
    models = [m for m in contr_df["model"].unique() if m != baseline]

    items, ylabels = [], []
    for r in res_order:
        for m in models:
            sub = contr_df[(contr_df["resolution"].astype(int) == r) & (contr_df["model"] == m)]
            if sub.empty:
                continue
            items.append((r, m))
            ylabels.append(f"{m}  ({r} mm)")
    # forest draws top-to-bottom; reverse here to match its y-order
    items = items[::-1]
    ylabels = ylabels[::-1]
    return items, ylabels

# ------------------------- data I/O helpers -------------------------------

@dataclass(frozen=True)
class Inputs:
    stats_npz: pathlib.Path
    metrics_npz: pathlib.Path | None
    out_dir: pathlib.Path

def load_stats(stats_npz: pathlib.Path) -> Dict[str, object]:
    """
    Load sr_stats_summary.npz and normalize to DataFrames and dicts.
    """
    d = np.load(stats_npz, allow_pickle=True)
    meta = d["meta"].item() if isinstance(d["meta"].item(), dict) else json.loads(d["meta"].item())
    # lmm_emm: dict[pulse]['emm_table'] recarray
    lmm_emm = d["lmm_emm"].item()
    emm_df: Dict[str, pd.DataFrame] = {}
    for p, sub in lmm_emm.items():
        rec = sub["emm_table"]
        df = pd.DataFrame(rec) if getattr(rec, "dtype", None) is not None else pd.DataFrame()
        emm_df[p] = df

    # contrasts: dict[pulse]['contrasts'] list[dict]
    lmm_contr = d["lmm_contrasts"].item()
    contr_df: Dict[str, pd.DataFrame] = {}
    for p, sub in lmm_contr.items():
        lst = sub["contrasts"]
        contr_df[p] = pd.DataFrame(lst) if lst else pd.DataFrame()

    # NP tests
    fried = d["friedman"].item()
    wilc = d["wilcoxon"].item()
    fried_df: Dict[str, pd.DataFrame] = {}
    wilc_df: Dict[str, pd.DataFrame] = {}
    for p in contr_df.keys():
        fried_df[p] = pd.DataFrame(fried[p]["by_res_roi"]) if fried[p]["by_res_roi"] else pd.DataFrame()
        wilc_df[p] = pd.DataFrame(wilc[p]["pairs"]) if wilc[p]["pairs"] else pd.DataFrame()

    # ICC
    icc = d["icc"].item()
    icc_df: Dict[str, pd.DataFrame] = {}
    for p, lst in icc.items():
        icc_df[p] = pd.DataFrame(lst) if lst else pd.DataFrame()

    return {
        "meta": meta,
        "emm": emm_df,
        "contr": contr_df,
        "fried": fried_df,
        "wilc": wilc_df,
        "icc": icc_df
    }

def load_metrics_counts(metrics_npz: pathlib.Path | None) -> pd.DataFrame:
    """
    Optional: load metrics.npz to extract per (pulse,resolution,model,roi) counts.
    Returns empty df if not provided. Used only for annotations.
    """
    if metrics_npz is None:
        return pd.DataFrame()
    d = np.load(metrics_npz, allow_pickle=True)
    if "metrics" not in d.files:
        return pd.DataFrame()
    arr = d["metrics"]  # (P,3,3,M,4,4,2)
    patient_ids = d["patient_ids"]
    pulses = list(d["pulses"]); resolutions = list(d["resolutions_mm"]); models = list(d["models"])
    metric_names = list(d["metric_names"]); roi_labels = list(d["roi_labels"]); stat_names = list(d["stat_names"])
    mean_idx = stat_names.index("mean")
    rows = []
    P, n_p, n_r, n_m, n_me, n_ro, _ = arr.shape
    for pi in range(P):
        for ip in range(n_p):
            for ir in range(n_r):
                for im in range(n_m):
                    for iro in range(n_ro):
                        # count presence for primary metric only to avoid double-counting
                        val = arr[pi, ip, ir, im, 0, iro, mean_idx]  # metric idx 0 is fine for counts
                        if np.isfinite(val):
                            rows.append({
                                "patient": patient_ids[pi],
                                "pulse": pulses[ip],
                                "resolution": resolutions[ir],
                                "model": models[im],
                                "roi": roi_labels[iro]
                            })
    df = pd.DataFrame(rows)
    cnt = df.groupby(["pulse","resolution","model","roi"], as_index=False).size().rename(columns={"size":"n"})
    return cnt

# ------------------------- color palettes (Paul Tol) ----------------------

def tol_bright() -> Dict[str, str]:
    """
    Paul Tol Bright palette. Return as dict name->hex for convenience.
    """
    return {
        "blue":   "#4477AA",
        "cyan":   "#66CCEE",
        "green":  "#228833",
        "yellow": "#CCBB44",
        "red":    "#EE6677",
        "purple": "#AA3377",
        "grey":   "#BBBBBB",
        "black":  "#000000",
    }

def model_colors(models: List[str]) -> Dict[str, str]:
    """
    Map your four models to distinct Tol colors, with BSPLINE in gray.
    """
    base = tol_bright()
    order = ["BSPLINE","ECLARE","SMORE","UNIRES"]
    palette = {
        "LINEAR_SUPER_LEARNER": base["purple"],
        "BSPLINE": base["grey"],
        "ECLARE":  base["blue"],
        "SMORE":   base["red"],
        "UNIRES":  base["green"],
    }
    # fill any unexpected names deterministically
    extras = [c for k,c in base.items() if k not in ["grey"]]
    for m, col in zip([m for m in models if m not in palette], extras):
        palette[m] = col
    return palette

def tol_diverging(n_bins: int = 11) -> LinearSegmentedColormap:
    """
    Paul Tol–like diverging colormap (blue→gray→red) with n_bins discrete colors.
    Uses linear interpolation between control points.
    """
    ctrl = ["#3B7C9C", "#87B5D6", "#E5E5E5", "#E7A5A5", "#C85A5A"]
    return LinearSegmentedColormap.from_list("tol_div", ctrl, N=n_bins)

# ------------------------- plotting primitives ---------------------------

def forest_contrasts(ax: plt.Axes, df: pd.DataFrame, palette: Dict[str,str], title: str,
                     items_in: List[Tuple[int,str]] | None = None) -> List[Tuple[int,str]]:
    if df.empty:
        ax.text(0.5, 0.5, "No contrasts", ha="center", va="center"); ax.set_axis_off(); return []
    if items_in is None:
        items, ylabels = _forest_items_order(df)
    else:
        items = items_in
        # rebuild matching labels
        ylabels = [f"{m}  ({r} mm)" for r, m in items][::-1][::-1]  # stable
    if not items:
        ax.text(0.5, 0.5, "No contrasts", ha="center", va="center"); ax.set_axis_off(); return []

    # Build a dict for quick access
    key = {(int(row["resolution"]), row["model"]): row for _, row in df.iterrows()}

    y = np.arange(len(items))[::-1]
    for i, (r, m) in enumerate(items):
        row = key.get((int(r), m))
        if row is None: 
            continue
        g = row.get("hedges_g", np.nan)
        lo, hi = row.get("g_ci", (np.nan, np.nan))
        ax.plot([lo, hi], [y[i], y[i]], color=palette.get(m, "#555555"), lw=1.4)
        ax.plot([g], [y[i]], marker="s", ms=4, color=palette.get(m, "#555555"))

    ax.axvline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Hedges $g$ vs BSPLINE (primary metric)")
    ax.set_title(title, loc="left")
    return items


def adjusted_means_panel(ax: plt.Axes, df: pd.DataFrame, palette: Dict[str,str], title: str) -> Tuple[List, List]:
    if df.empty:
        ax.text(0.5, 0.5, "No EMMs", ha="center", va="center"); ax.set_axis_off(); return [], []
    res_order = sorted({int(x) for x in df["resolution"].astype(int).unique()})
    models = list(df["model"].unique())
    x = np.arange(len(res_order))
    width = 0.8 / max(len(models), 1)
    handles, labels = [], []
    for j, m in enumerate(models):
        sub = df[df["model"] == m].copy(); sub.index = sub["resolution"].astype(int)
        means = [sub.loc[r, "mean"] if r in sub.index else np.nan for r in res_order]
        se = [sub.loc[r, "se"] if r in sub.index else np.nan for r in res_order]
        offs = x - 0.4 + j*width + width/2
        h = ax.errorbar(offs, means, yerr=[1.96*np.array(se), 1.96*np.array(se)],
                        fmt="o-", lw=1.2, ms=3.5, color=palette.get(m, "#555555"),
                        label=m, capsize=2)
        handles.append(h); labels.append(m)
    ax.set_xticks(x); ax.set_xticklabels([f"{r} mm" for r in res_order])
    ax.set_xlabel("Through-plane spacing"); ax.set_ylabel(r"Adjusted mean (±95\% CI)")
    ax.set_title(title, loc="left")
    return handles, labels

def wilcoxon_heatmap(ax: plt.Axes, df: pd.DataFrame, models: List[str], title: str,
                     transpose_to_items: List[Tuple[int,str]] | None = None) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No Wilcoxon results", ha="center", va="center"); ax.set_axis_off(); return

    rois = ["all","core","edema","surround"]
    baseline = df["baseline"].dropna().unique()
    baseline = baseline[0] if len(baseline) else "BSPLINE"

    if transpose_to_items is None:
        # original layout: rows=ROI, cols=resolutionxmodel (non-baseline)
        res_order = sorted(df["resolution"].astype(int).unique())
        cols = [(r, m) for r in res_order for m in models if m != baseline]
        mat = np.full((len(rois), len(cols)), np.nan, float)
        for i, roi in enumerate(rois):
            for j, (r, m) in enumerate(cols):
                sub = df[(df["roi"] == roi) & (df["resolution"].astype(int) == r) & (df["model"] == m)]
                if sub.empty: continue
                p = float(sub.get("p_holm", sub.get("p_raw", np.nan)))
                p = max(p, 1e-300) if np.isfinite(p) else 1.0
                sign = np.sign(float(sub["dz"].values[0])) if "dz" in sub else 1.0
                mat[i, j] = sign * min(10.0, -np.log10(p))
        # draw
        n_bins = 11; cmap = tol_diverging(n_bins=n_bins)
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 5.0
        bounds = np.linspace(-vmax, vmax, n_bins + 1); norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(np.arange(len(rois))); ax.set_yticklabels(rois)
        ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels([f"{m}\n{r}mm" for (r, m) in cols])
        ax.set_title(title, loc="left")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"sign($\Delta$) $\times$ $-\log_{10} p_{\mathrm{Holm}}$")
        return

    # transposed layout: rows = items (match forest), cols = ROI
    items = transpose_to_items
    mat = np.full((len(items), len(rois)), np.nan, float)
    for i, (r, m) in enumerate(items):
        for j, roi in enumerate(rois):
            sub = df[(df["roi"] == roi) & (df["resolution"].astype(int) == int(r)) & (df["model"] == m)]
            if sub.empty: continue
            p = float(sub.get("p_holm", sub.get("p_raw", np.nan)))
            p = max(p, 1e-300) if np.isfinite(p) else 1.0
            sign = np.sign(float(sub["dz"].values[0])) if "dz" in sub else 1.0
            mat[i, j] = sign * min(10.0, -np.log10(p))
    n_bins = 11; cmap = tol_diverging(n_bins=n_bins)
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 5.0
    bounds = np.linspace(-vmax, vmax, n_bins + 1); norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    # y ticks aligned to forest order
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels([f"{m}  ({r} mm)" for r, m in items])
    ax.set_xticks(np.arange(len(rois))); ax.set_xticklabels(rois)
    ax.set_title(title, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"sign($\Delta$) $\times$ $-\log_{10} p_{\mathrm{Holm}}$")

def icc_bars(ax: plt.Axes, df: pd.DataFrame, palette: Dict[str,str], title: str, roi: str = "all") -> None:
    """Median ICC(2,1) per model × resolution as grouped bars. x-axis = spacing."""
    if df.empty:
        ax.text(0.5, 0.5, "No ICC results", ha="center", va="center"); ax.set_axis_off(); return
    df = df[df["roi"] == roi]
    if df.empty:
        ax.text(0.5, 0.5, f"No ICC for ROI={roi}", ha="center", va="center"); ax.set_axis_off(); return
    agg = (df.groupby(["model","resolution"], as_index=False)["ICC2_1"]
             .median().rename(columns={"ICC2_1":"median"}))
    res_order = sorted(agg["resolution"].astype(int).unique())
    models = list(agg["model"].unique())
    x = np.arange(len(res_order))
    width = 0.8 / max(len(models), 1)
    for j, m in enumerate(models):
        sub = agg[agg["model"] == m].copy(); sub.index = sub["resolution"].astype(int)
        vals = [sub.loc[r, "median"] if r in sub.index else np.nan for r in res_order]
        offs = x - 0.4 + j*width + width/2
        ax.bar(offs, vals, width=width, color=palette.get(m, "#555555"), edgecolor="none", label=m, alpha=0.95)
    ax.axhline(0.90, color="#666666", lw=0.8, ls="--")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"{r} mm" for r in res_order])
    ax.set_xlabel("Through-plane spacing"); ax.set_ylabel("ICC(2,1) median")
    ax.set_title(title + f" (ROI={roi})", loc="left")

# ------------------------- figure orchestration ---------------------------

def model_legend_handles(models: List[str], palette: Dict[str,str]) -> Tuple[List[plt.Line2D], List[str]]:
    """Create unified legend proxies for model colors."""
    handles = [mlines.Line2D([], [], color=palette.get(m, "#555555"), marker="s", linestyle="-") for m in models]
    return handles, models

def make_panels_for_pulse(pulse: str,
                          stats: Dict[str, object],
                          out_dir: pathlib.Path) -> None:
    configure_matplotlib()
    meta = stats["meta"]; palette = model_colors(meta["models"])
    emm   = stats["emm"].get(pulse, pd.DataFrame())
    contr = stats["contr"].get(pulse, pd.DataFrame())
    wilc  = stats["wilc"].get(pulse, pd.DataFrame())
    icc   = stats["icc"].get(pulse, pd.DataFrame())

    # Always make a unified legend for models
    leg_handles, leg_labels = model_legend_handles(meta["models"], palette)

    if icc is not None and not icc.empty:
        # 1 × 4 panel: [Forest][Wilcoxon^T][Adjusted means][ICC bars]
        fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)

        # Col 0: Forest
        items = forest_contrasts(axes[0], contr, palette, title=f"{pulse.upper()}: Effect vs BSPLINE")

        # Col 1: Wilcoxon transposed, share y with forest, hide its y ticklabels
        if items:
            axes[1].get_shared_y_axes().joined(axes[0], axes[1])
        wilcoxon_heatmap(axes[1], wilc, meta["models"], title=f"{pulse.upper()}: Wilcoxon vs BSPLINE",
                         transpose_to_items=items if items else None)
        axes[1].tick_params(axis='y', which='both', labelleft=False)

        # Col 2: Adjusted means, x = spacing
        adjusted_means_panel(axes[2], emm, palette, title=f"{pulse.upper()}: Adjusted means")

        # Col 3: ICC bars, x = spacing
        icc_bars(axes[3], icc, palette, title=f"{pulse.upper()}: Radiomic stability", roi="all")

        # Unified legend outside bottom
        if leg_handles and leg_labels:
            fig.legend(leg_handles, leg_labels, loc="lower center",
                       ncol=len(leg_labels), frameon=False, bbox_to_anchor=(0.5, -0.12))
            fig.subplots_adjust(bottom=0.18)

        for ext in ("pdf","png"):
            out = out_dir / f"sr_panel_{pulse}_1x4.{ext}"
            fig.savefig(out); LOG.info("Saved %s", out)
        plt.close(fig)
        return

    # Fallback: ICC missing → 1 × 3 [Adjusted][Forest][Wilcoxon^T]
    LOG.warning("ICC empty for pulse=%s. Building 1×3 panel.", pulse)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2),
                             gridspec_kw={"width_ratios":[1.2, 1.2, 1.4]},
                             constrained_layout=True)
    adjusted_means_panel(axes[0], emm, palette, title=f"{pulse.upper()}: Adjusted means")
    items = forest_contrasts(axes[1], contr, palette, title=f"{pulse.upper()}: Effect vs BSPLINE")
    if items:
        axes[2].get_shared_y_axes().joined(axes[1], axes[2])
    wilcoxon_heatmap(axes[2], wilc, meta["models"], title=f"{pulse.upper()}: Wilcoxon vs BSPLINE",
                     transpose_to_items=items if items else None)
    axes[2].tick_params(axis='y', which='both', labelleft=False)
    if leg_handles and leg_labels:
        fig.legend(leg_handles, leg_labels, loc="lower center",
                   ncol=len(leg_labels), frameon=False, bbox_to_anchor=(0.5, -10.5))
        fig.subplots_adjust(bottom=0.20)
    for ext in ("pdf","png"):
        out = out_dir / f"sr_panel_{pulse}_COMBINED.{ext}"
        fig.savefig(out); LOG.info("Saved %s", out)
    plt.close(fig)

# ------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_npz", type=pathlib.Path, required=True)
    ap.add_argument("--metrics_npz", type=pathlib.Path, required=False, default=None)
    ap.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s | %(message)s")
    LOG.info("Starting SR statistics plotting …")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.stats_npz)
    LOG.info("Loaded stats: pulses=%s | models=%s | resolutions=%s",
             stats["meta"]["pulses"], stats["meta"]["models"], stats["meta"]["resolutions_mm"])

    # optional counts if you want later annotations
    cnt = load_metrics_counts(args.metrics_npz)
    if not cnt.empty:
        LOG.info("Loaded counts from metrics.npz (rows=%d).", len(cnt))

    for pulse in stats["meta"]["pulses"]:
        make_panels_for_pulse(pulse, stats, args.out_dir)

    LOG.info("All figures saved to %s", args.out_dir)

if __name__ == "__main__":
    main()
