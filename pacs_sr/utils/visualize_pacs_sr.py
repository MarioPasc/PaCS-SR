
#!/usr/bin/env python3
"""
PaCS-SR visualizations.

Generates figures from {experiment}/{spacing}/model_data/fold_*/{pulse}/
- metrics.json           -> per-fold aggregate metrics
- *_weights_test.npz     -> weight maps + metadata (per patient)
- *_weight_analysis_*.npz-> entropy + dominant model maps (if saved)
- output_volumes/*.nii.gz-> blended predictions (optional slice views)

Usage:
  python -m pacs_sr.visualize.visualize_pacs_sr --config /path/to/config.yaml

Optional filters:
  --spacing 3mm --pulse t1c --fold 1 --limit-patients 10

Outputs go to: {visualizations.out_root}/{experiment}/{spacing}/{pulse}/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

# Hard rules from tool: do not set styles or colors.

# ------------------------- helpers -------------------------

def mid_slices(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc = vol.shape[0] // 2
    yc = vol.shape[1] // 2
    xc = vol.shape[2] // 2
    return vol[zc, :, :], vol[:, yc, :], vol[:, :, xc]

def load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def find_files(base: Path, pattern: str) -> List[Path]:
    return sorted(base.rglob(pattern))

def safe_load_npz(p: Path) -> Optional[dict]:
    try:
        data = np.load(p, allow_pickle=True)
        return {k: data[k] for k in data.files}
    except Exception:
        return None

def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps: 
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

# ------------------------- core -------------------------

def collect_fold_metrics(space_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted((space_dir / "model_data").glob("fold_*")):
        fold_str = fold_dir.name.split("_")[-1]
        for pulse_dir in sorted(fold_dir.iterdir()):
            metrics_p = pulse_dir / "metrics.json"
            if metrics_p.exists():
                m = load_json(metrics_p)
                if "test" in m and m["test"]:
                    row = {"fold": int(fold_str), "pulse": pulse_dir.name}
                    row.update({f"test_{k}": float(v) for k, v in m["test"].items()})
                    # include optional train
                    if "train" in m and m["train"]:
                        row.update({f"train_{k}": float(v) for k, v in m["train"].items()})
                    rows.append(row)
    return pd.DataFrame(rows)

def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = [c for c in df.columns if c.startswith("test_")]
    grp = df.groupby("pulse")[metrics].agg(["mean","std"]).reset_index()
    # flatten columns
    grp.columns = ["pulse"] + [f"{m}_{s}" for m in metrics for s in ("mean","std")]
    return grp

def collect_aggregated(full_cfg, results_root: Path) -> pd.DataFrame:
    """
    Build a single table with aggregated test metrics per spacing and pulse.
    Columns:
      spacing, pulse, metric, mean, std, n_folds
    """
    exp = full_cfg.pacs_sr.experiment_name
    rows = []
    for sp in full_cfg.pacs_sr.spacings:
        space_dir = results_root / sp
        df = collect_fold_metrics(space_dir)
        if df.empty:
            continue
        # summarize per pulse
        metrics = [c for c in df.columns if c.startswith("test_")]
        g = df.groupby("pulse")
        for pu, dpu in g:
            for m in metrics:
                mu = float(dpu[m].mean())
                sd = float(dpu[m].std(ddof=0))
                rows.append({
                    "spacing": sp, "pulse": pu,
                    "metric": m.replace("test_", ""),
                    "mean": mu, "std": sd, "n_folds": int(dpu.shape[0])
                })
    return pd.DataFrame(rows)

def plot_aggregated_heatmaps(df: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    """
    For each metric, create a spacing × pulse heatmap of the mean value.
    """
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in sorted(df["metric"].unique()):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        piv = sub.pivot(index="spacing", columns="pulse", values="mean")
        fig = plt.figure()
        plt.imshow(piv.values, aspect="auto", origin="upper")
        plt.xticks(range(piv.shape[1]), piv.columns, rotation=0)
        plt.yticks(range(piv.shape[0]), piv.index)
        plt.colorbar(label=f"{metric.upper()} (mean)")
        plt.title(f"{title_prefix} | {metric.upper()}")
        fig.tight_layout()
        fig.savefig(out_dir / f"agg_heatmap_{metric}.png", dpi=200)
        plt.close(fig)

def plot_aggregated_bars(df: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    """
    For each spacing, draw grouped bars per sequence with error bars using mean±std.
    """
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in sorted(df["spacing"].unique()):
        sub_sp = df[df["spacing"] == sp]
        for metric in sorted(sub_sp["metric"].unique()):
            sub = sub_sp[sub_sp["metric"] == metric]
            if sub.empty:
                continue
            pulses = list(sub["pulse"].values)
            y = sub["mean"].values
            yerr = sub["std"].values
            x = range(len(pulses))
            fig = plt.figure()
            plt.bar(x, y, yerr=yerr)
            plt.xticks(list(x), pulses, rotation=0)
            plt.ylabel(metric.upper())
            plt.title(f"{title_prefix} | {sp}")
            fig.tight_layout()
            fig.savefig(out_dir / f"agg_{sp}_{metric}.png", dpi=200)
            plt.close(fig)

def plot_metric_bars(df: pd.DataFrame, out_dir: Path, title: str) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [c for c in df.columns if c.startswith("test_") and c.endswith("_mean")]
    x = np.arange(len(df["pulse"].values))
    for metric in metrics:
        fig = plt.figure()
        y = df[metric].values
        yerr = df[metric.replace("_mean","_std")].values if metric.replace("_mean","_std") in df.columns else None
        plt.bar(x, y, yerr=yerr)
        plt.xticks(x, df["pulse"].values, rotation=0)
        plt.ylabel(metric.replace("test_","").replace("_mean","").upper())
        plt.title(title)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=200)
        plt.close(fig)

def load_weight_maps_or_analysis(pulse_dir: Path) -> List[dict]:
    """
    Prefer *_weight_analysis_test.npz, else *_weights_test.npz.
    Return list of dicts with keys: weight_maps, entropy_map, dominant_map, model_names, patient_id
    """
    out = []
    # analysis files
    analysis = find_files(pulse_dir, "*_weight_analysis_test.npz")
    if analysis:
        for p in analysis:
            d = safe_load_npz(p)
            if d is None: 
                continue
            out.append({
                "weight_maps": d.get("weight_maps"),
                "entropy_map": d.get("entropy_map"),
                "dominant_map": d.get("dominant_map"),
                "model_names": list(d.get("model_names")) if "model_names" in d else None,
                "patient_id": str(d.get("patient_id")) if "patient_id" in d else p.stem
            })
        return out
    # fallback to raw weight maps
    raw = find_files(pulse_dir, "*_weights_test.npz")
    for p in raw:
        d = safe_load_npz(p)
        if d is None: 
            continue
        wm = d.get("weight_maps")
        model_names = list(d.get("model_names")) if "model_names" in d else None
        pid = str(d.get("patient_id")) if "patient_id" in d else p.stem
        # synthesize entropy and dominant maps
        s = wm.sum(axis=-1, keepdims=True)
        wn = wm / (s + 1e-8)
        entropy = -np.sum(wn * np.log(wn + 1e-10), axis=-1)
        dom = np.argmax(wn, axis=-1).astype(np.int32)
        out.append({
            "weight_maps": wm, "entropy_map": entropy, "dominant_map": dom,
            "model_names": model_names, "patient_id": pid
        })
    return out

def plot_weight_stats(weight_items: List[dict], out_dir: Path, model_names: List[str], limit: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) global histograms per model across all patients (subsample to limit memory)
    if not weight_items:
        return
    # stack limited patients
    items = weight_items[:max(1, min(limit, len(weight_items)))]
    W = [it["weight_maps"] for it in items if it["weight_maps"] is not None]
    if W:
        W = np.concatenate(W, axis=0) if W[0].ndim==4 else W[0]  # defensive
        # reshape to (Nvox, M)
        M = W.shape[-1]
        flat = W.reshape(-1, M)
        # hist per model
        for i, name in enumerate(model_names or [f"model_{i}" for i in range(M)]):
            fig = plt.figure()
            plt.hist(flat[:, i], bins=50)
            plt.xlabel(f"Weight for {name}")
            plt.ylabel("Voxel count")
            plt.title("Weight distribution")
            fig.tight_layout()
            fig.savefig(out_dir / f"weights_hist_{i:02d}_{name}.png", dpi=200)
            plt.close(fig)

    # 2) entropy histogram across all voxels
    ent_all = []
    for it in items:
        if it["entropy_map"] is not None:
            ent_all.append(it["entropy_map"].reshape(-1))
    if ent_all:
        ent_all = np.concatenate(ent_all, axis=0)
        fig = plt.figure()
        plt.hist(ent_all, bins=50)
        plt.xlabel("Entropy")
        plt.ylabel("Voxel count")
        plt.title("Weight entropy distribution")
        fig.tight_layout()
        fig.savefig(out_dir / f"entropy_hist.png", dpi=200)
        plt.close(fig)

def plot_patient_maps(item: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = item["patient_id"]
    wm = item["weight_maps"]
    ent = item["entropy_map"]
    dom = item["dominant_map"]
    model_names = item["model_names"] or [f"model_{i}" for i in range(wm.shape[-1])]

    # middle slices
    z_ent, y_ent, x_ent = mid_slices(ent)
    z_dom, y_dom, x_dom = mid_slices(dom.astype(np.float32))

    # entropy
    for plane, arr in zip(("axial","coronal","sagittal"), (z_ent, y_ent, x_ent)):
        fig = plt.figure()
        plt.imshow(normalize01(arr), origin="lower")
        plt.axis("off")
        plt.title(f"{pid} | entropy | {plane}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{pid}_entropy_{plane}.png", dpi=200)
        plt.close(fig)

    # dominant model
    for plane, arr in zip(("axial","coronal","sagittal"), (z_dom, y_dom, x_dom)):
        fig = plt.figure()
        plt.imshow(arr, origin="lower")
        plt.axis("off")
        plt.title(f"{pid} | dominant-model | {plane}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{pid}_dominant_{plane}.png", dpi=200)
        plt.close(fig)

    # per-model weight mid-slice panels
    z_w, y_w, x_w = mid_slices(wm[..., 0]*0)  # shape refs
    mids = { "axial": wm[wm.shape[0]//2, :, :, :],
             "coronal": wm[:, wm.shape[1]//2, :, :],
             "sagittal": wm[:, :, wm.shape[2]//2, :] }
    for plane, W in mids.items():
        for i, name in enumerate(model_names):
            fig = plt.figure()
            plt.imshow(normalize01(W[..., i]), origin="lower")
            plt.axis("off")
            plt.title(f"{pid} | {name} | {plane}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{pid}_weights_{plane}_{i:02d}_{name}.png", dpi=200)
            plt.close(fig)

def try_plot_blend_views(blend_path: Path, out_dir: Path, tag: str) -> None:
    if not blend_path.exists():
        return
    img = nib.load(str(blend_path))
    vol = img.get_fdata().astype(np.float32)
    for plane, arr in zip(("axial","coronal","sagittal"), mid_slices(vol)):
        fig = plt.figure()
        plt.imshow(normalize01(arr), origin="lower")
        plt.axis("off")
        plt.title(f"{tag} | {plane}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_blend_{plane}.png", dpi=200)
        plt.close(fig)

# ------------------------- main -------------------------

def run(cfg_path: Path, spacing: Optional[str], pulse: Optional[str], fold: Optional[int], limit_patients: Optional[int] = None):
    from pacs_sr.config.config import load_full_config
    full = load_full_config(cfg_path)
    exp = full.pacs_sr.experiment_name
    spacings = [spacing] if spacing else list(full.pacs_sr.spacings)
    pulses = [pulse] if pulse else list(full.pacs_sr.pulses)

    # Visualizations section is optional
    viz = getattr(full, "visualizations", None)
    if viz is None:
        raise SystemExit("Missing 'visualizations' section in YAML")
    results_root = Path(viz.results_root)
    out_root = Path(viz.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Experiment: {exp}")
    print(f"[DEBUG] Results root: {results_root}")
    print(f"[DEBUG] Output root: {out_root}")
    print(f"[DEBUG] Spacings to process: {spacings}")
    print(f"[DEBUG] Pulses to process: {pulses}")

    # ---------- aggregated overview across all spacings × sequences ----------
    agg_all = collect_aggregated(full, results_root)
    print(f"[DEBUG] Aggregated dataframe shape: {agg_all.shape}")
    print(f"[DEBUG] Aggregated dataframe empty: {agg_all.empty}")
    if not agg_all.empty:
        print(f"[DEBUG] Aggregated metrics:\n{agg_all.head()}")
        overview_dir = out_root / exp / "aggregated"
        overview_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving aggregated outputs to: {overview_dir}")
        agg_all.to_csv(overview_dir / f"{exp}_aggregated_metrics.csv", index=False)
        plot_aggregated_heatmaps(agg_all, overview_dir, title_prefix=f"{exp} aggregated")
        plot_aggregated_bars(agg_all, overview_dir, title_prefix=f"{exp} aggregated")

    for sp in spacings:
        space_dir = results_root / sp
        print(f"\n[DEBUG] Processing spacing: {sp}")
        print(f"[DEBUG] Space directory: {space_dir}")
        print(f"[DEBUG] Space directory exists: {space_dir.exists()}")
        # ---------- metrics per fold ----------
        df = collect_fold_metrics(space_dir)
        if not df.empty:
            df.to_csv(out_root / f"{exp}_{sp}_fold_metrics.csv", index=False)
            agg = summarize_metrics(df)
            agg.to_csv(out_root / f"{exp}_{sp}_summary.csv", index=False)
            plot_metric_bars(agg, out_root / exp / sp / "metrics", title=f"{exp} | {sp}")

        # ---------- per-pulse weight maps ----------
        folds = [fold] if fold else sorted({int(p.name.split('_')[-1]) for p in (space_dir / "model_data").glob("fold_*")})
        for pu in pulses:
            for f in folds:
                pulse_dir = space_dir / "model_data" / f"fold_{f}" / pu
                if not pulse_dir.exists():
                    continue
                items = load_weight_maps_or_analysis(pulse_dir)
                if not items:
                    continue
                model_names = items[0]["model_names"] or list(full.pacs_sr.models)
                # global stats across patients
                plot_weight_stats(items, out_root / exp / sp / pu / f"fold_{f}" / "weights_global", model_names=model_names)
                # a few patients panels
                lim = min(limit_patients or 4, len(items))
                for it in items[:lim]:
                    plot_patient_maps(it, out_root / exp / sp / pu / f"fold_{f}" / "patients")

                # try to render a blend for the first few patients
                blends_dir = space_dir / "output_volumes"
                # guess filenames: "{patient_id}-{pulse}.nii.gz"
                for it in items[:lim]:
                    blend_path = blends_dir / f"{it['patient_id']}-{pu}.nii.gz"
                    tag = f"{it['patient_id']}-{pu}"
                    try_plot_blend_views(blend_path, out_root / exp / sp / pu / f"fold_{f}" / "patients", tag)

    print(f"Done. Figures at: {out_root}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--spacing", type=str, default=None)
    ap.add_argument("--pulse", type=str, default=None)
    ap.add_argument("--fold", type=int, default=None, help="1-indexed")
    ap.add_argument("--limit-patients", type=int, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.spacing, args.pulse, args.fold, args.limit_patients)
