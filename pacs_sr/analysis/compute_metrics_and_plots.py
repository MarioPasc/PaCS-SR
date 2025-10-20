#!/usr/bin/env python3
# compute_metrics_and_plots.py
# Author: Mario's analysis pipeline helper
# Python >=3.10

from __future__ import annotations
import argparse
import concurrent.futures as fx
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import logging
from typing import Iterable, Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ------------------------------ Configuration --------------------------------

@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration.

    Attributes:
        models_root: Root directory that contains per-model subfolders.
        gt_root: Root directory of ground-truth volumes (per-case subfolder).
        outdir: Output directory for CSVs and plots.
        models: List of model names to evaluate (must match subfolders).
        spacings: List like ["3mm","5mm","7mm"].
        sequences: List like ["t1c","t1n","t2w","t2f"].
        workers: Parallel workers for metric computation.
        normalize: One of {"none","minmax","robust"}; applied per-case pair.
        ssim_win: Odd integer SSIM window size; clipped to volume dims if needed.
    """
    models_root: Path
    gt_root: Path
    outdir: Path
    models: Tuple[str, ...]
    spacings: Tuple[str, ...]
    sequences: Tuple[str, ...]
    workers: int
    normalize: str
    ssim_win: int = 7

class DataMismatchError(RuntimeError):
    """Raised when predicted and ground-truth volumes are incompatible."""


# ------------------------------ Utilities ------------------------------------

SEQ_RE = re.compile(r"^(?P<case>[^/]+)-(?:t1c|t1n|t2w|t2f)\.nii\.gz$")
FILE_RE = re.compile(r"^(?P<case>.+)-(?P<seq>t1c|t1n|t2w|t2f)\.nii\.gz$")

def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "metrics.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
    )
    logging.info("Logging initialized.")


def list_prediction_files(models_root: Path, model: str, spacing: str) -> List[Path]:
    base = models_root / model / spacing / "output_volumes"
    if not base.exists():
        logging.warning("Missing directory: %s", base)
        return []
    return sorted(base.glob("*.nii.gz"))


def parse_case_and_seq(pred_path: Path) -> Tuple[str, str]:
    m = FILE_RE.match(pred_path.name)
    if not m:
        raise ValueError(f"Unexpected filename pattern: {pred_path}")
    return m.group("case"), m.group("seq")


def gt_path_for_case(gt_root: Path, case_id: str, seq: str) -> Path:
    return gt_root / case_id / f"{case_id}-{seq}.nii.gz"


def robust_rescale(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        # Degenerate; fallback to minmax
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def minmax_rescale(x: np.ndarray) -> np.ndarray:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32, copy=False)


def maybe_normalize(a: np.ndarray, b: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)
    if mode == "minmax":
        return minmax_rescale(a), minmax_rescale(b)
    if mode == "robust":
        return robust_rescale(a), robust_rescale(b)
    raise ValueError(f"Unknown normalize mode: {mode}")


def safe_load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))  # type: ignore
    arr = np.asanyarray(img.get_fdata(dtype=np.float32))  # type: ignore
    return arr  # shape: (X, Y, Z) or similar


def compute_psnr_ssim(
    pred_vol: np.ndarray, gt_vol: np.ndarray, ssim_win: int
) -> Tuple[float, float]:
    if pred_vol.shape != gt_vol.shape:
        raise DataMismatchError(f"Shape mismatch: pred {pred_vol.shape} vs gt {gt_vol.shape}")

    # PSNR over full 3D volume; assumes intensities in [0,1] if normalized
    data_range = float(np.max(gt_vol) - np.min(gt_vol)) or 1.0
    psnr = float(peak_signal_noise_ratio(gt_vol, pred_vol, data_range=data_range))

    # SSIM: try true N-D SSIM; on failure, average slice-wise
    win = min(ssim_win, *(d if d % 2 == 1 else d - 1 for d in gt_vol.shape))  # keep odd
    win = max(win, 3)

    try:
        ssim_val = float(
            structural_similarity(  # type: ignore
                gt_vol, pred_vol,
                data_range=data_range,
                win_size=win,
                gaussian_weights=True,
                sigma=1.5,
                channel_axis=None,
                use_sample_covariance=False,
                K1=0.01, K2=0.03,
            )
        )
    except Exception:
        # Fallback: mean SSIM across axial slices
        ssim_vals = []
        for z in range(gt_vol.shape[-1]):
            ssim_vals.append(
                structural_similarity(
                    gt_vol[..., z], pred_vol[..., z],
                    data_range=float(np.max(gt_vol[..., z]) - np.min(gt_vol[..., z])) or 1.0,
                    win_size=min(win, gt_vol.shape[0] - (1 - gt_vol.shape[0] % 2),
                                 gt_vol.shape[1] - (1 - gt_vol.shape[1] % 2)),
                    gaussian_weights=True, sigma=1.5,
                    channel_axis=None,
                    use_sample_covariance=False,
                    K1=0.01, K2=0.03,
                )
            )
        ssim_val = float(np.mean(ssim_vals))

    return psnr, ssim_val


def task_compute(
    cfg: Config, model: str, spacing: str, pred_path: Path
) -> Optional[Dict[str, object]]:
    try:
        case_id, seq = parse_case_and_seq(pred_path)
        if seq not in cfg.sequences:
            return None

        gt_path = gt_path_for_case(cfg.gt_root, case_id, seq)
        if not gt_path.exists():
            logging.warning("Missing GT: %s", gt_path)
            return None

        pred = safe_load_nii(pred_path)
        gt = safe_load_nii(gt_path)
        pred, gt = maybe_normalize(pred, gt, cfg.normalize)
        psnr, ssim = compute_psnr_ssim(pred, gt, cfg.ssim_win)

        return {
            "model": model,
            "spacing": spacing,
            "sequence": seq,
            "case_id": case_id,
            "psnr": psnr,
            "ssim": ssim,
            "pred_path": str(pred_path),
            "gt_path": str(gt_path),
        }
    except DataMismatchError as e:
        logging.error("Shape mismatch for %s: %s", pred_path, e)
        return None
    except Exception as e:
        logging.exception("Failed on %s: %s", pred_path, e)
        return None


# ------------------------------ Orchestration --------------------------------

def collect_tasks(cfg: Config) -> List[Tuple[str, str, Path]]:
    tasks: List[Tuple[str, str, Path]] = []
    for model in cfg.models:
        for spacing in cfg.spacings:
            preds = list_prediction_files(cfg.models_root, model, spacing)
            if not preds:
                logging.warning("No predictions for %s/%s", model, spacing)
            for p in preds:
                # Only accept expected sequences
                m = FILE_RE.match(p.name)
                if not m:
                    continue
                seq = m.group("seq")
                if seq in cfg.sequences:
                    tasks.append((model, spacing, p))
    logging.info("Discovered %d prediction files across %d models and %d spacings.",
                 len(tasks), len(cfg.models), len(cfg.spacings))
    return tasks


def compute_all(cfg: Config) -> pd.DataFrame:
    tasks = collect_tasks(cfg)
    records: List[Dict[str, object]] = []

    with fx.ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [ex.submit(task_compute, cfg, model, spacing, p) for model, spacing, p in tasks]
        for i, fut in enumerate(futs, 1):
            res = fut.result()
            if res is not None:
                records.append(res)
            if i % 50 == 0:
                logging.info("Processed %d/%d files.", i, len(futs))

    df = pd.DataFrame.from_records(records)
    if df.empty:
        logging.error("No metrics computed. Check paths and patterns.")
        return df

    # Sort categorical orders
    df["model"] = pd.Categorical(df["model"], categories=list(cfg.models), ordered=True)
    df["spacing"] = pd.Categorical(df["spacing"], categories=list(cfg.spacings), ordered=True)
    df["sequence"] = pd.Categorical(df["sequence"], categories=list(cfg.sequences), ordered=True)

    # Persist raw metrics
    out_csv = cfg.outdir / "metrics_by_case.csv"
    df.to_csv(out_csv, index=False)
    logging.info("Wrote per-case metrics: %s", out_csv)

    # Aggregate
    agg = (
        df.groupby(["model", "spacing", "sequence"])
          .agg(psnr_mean=("psnr", "mean"),
               psnr_std=("psnr", "std"),
               ssim_mean=("ssim", "mean"),
               ssim_std=("ssim", "std"),
               n=("psnr", "count"))
          .reset_index()
          .sort_values(["sequence", "spacing", "model"])
    )
    agg_csv = cfg.outdir / "metrics_summary.csv"
    agg.to_csv(agg_csv, index=False)
    logging.info("Wrote summary metrics: %s", agg_csv)

    return df


# ------------------------------ Plotting -------------------------------------

def plot_line_by_spacing(df: pd.DataFrame, seq: str, metric: str, outpath: Path) -> None:
    sub = df[df["sequence"] == seq]
    if sub.empty:
        logging.warning("No data to plot for sequence=%s, metric=%s", seq, metric)
        return
    grp = sub.groupby(["model", "spacing"])[metric].agg(["mean", "std"]).reset_index()

    spacings = list(sub["spacing"].cat.categories)
    models = list(sub["model"].cat.categories)

    plt.figure(figsize=(7, 4.2))
    for m in models:
        g = grp[grp["model"] == m]
        means = [g[g["spacing"] == s]["mean"].values[0] if (g["spacing"] == s).any() else np.nan
                 for s in spacings]
        stds = [g[g["spacing"] == s]["std"].values[0] if (g["spacing"] == s).any() else np.nan
                for s in spacings]
        x = np.arange(len(spacings))
        plt.errorbar(x, means, yerr=stds, marker="o", label=m)

    plt.xticks(np.arange(len(spacings)), spacings)
    plt.xlabel("Spacing")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs spacing  |  sequence: {seq}")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    logging.info("Saved %s", outpath)


def plot_box_by_model(df: pd.DataFrame, spacing: str, metric: str, outpath: Path) -> None:
    sub = df[df["spacing"] == spacing]
    if sub.empty:
        logging.warning("No data to plot for spacing=%s, metric=%s", spacing, metric)
        return

    # One panel per sequence
    sequences = list(sub["sequence"].cat.categories)
    ncols = len(sequences)
    plt.figure(figsize=(4.0 * ncols, 4.2))

    for i, seq in enumerate(sequences, start=1):
        ax = plt.subplot(1, ncols, i)
        ss = sub[sub["sequence"] == seq]
        data = [ss[ss["model"] == m][metric].values for m in ss["model"].cat.categories]
        bp = ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(ss["model"].cat.categories) + 1))
        ax.set_xticklabels(list(ss["model"].cat.categories))
        ax.set_title(f"{seq}")
        ax.set_ylabel(metric.upper() if i == 1 else "")
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{metric.upper()} distribution by model  |  spacing: {spacing}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", outpath)


def make_all_plots(df: pd.DataFrame, outdir: Path) -> None:
    # Lines: per sequence, metric vs spacing with model means and stds
    for seq in df["sequence"].cat.categories:
        for metric in ["psnr", "ssim"]:
            out = outdir / f"{metric}_by_spacing_{seq}.png"
            plot_line_by_spacing(df, seq, metric, out)

    # Boxes: per spacing, boxplot across models, faceted by sequence
    for spacing in df["spacing"].cat.categories:
        for metric in ["psnr", "ssim"]:
            out = outdir / f"{metric}_by_model_{spacing}.png"
            plot_box_by_model(df, spacing, metric, out)


# ------------------------------ CLI ------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Compute PSNR and SSIM for MRI SR models, aggregate, and plot."
    )
    p.add_argument("--models-root", type=Path, required=True,
                   help="Root with model subfolders (e.g., BSPLINE, ECLARE, SMORE, PACS_SR).")
    p.add_argument("--gt-root", type=Path, required=True,
                   help="Root of high-resolution GT, one subfolder per case.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory for CSVs and plots.")
    p.add_argument("--models", nargs="+", default=["BSPLINE", "ECLARE", "SMORE", "PACS_SR"])
    p.add_argument("--spacings", nargs="+", default=["3mm", "5mm", "7mm"])
    p.add_argument("--sequences", nargs="+", default=["t1c", "t1n", "t2w", "t2f"])
    p.add_argument("--workers", type=int, default=max(1, (os_cpu := ( __import__('os').cpu_count() or 1) - 1)))
    p.add_argument("--normalize", choices=["none", "minmax", "robust"], default="robust",
                   help="Intensity normalization applied per-volume before metrics.")
    p.add_argument("--ssim-win", type=int, default=7, help="Odd window size for SSIM (>=3).")

    args = p.parse_args()

    cfg = Config(
        models_root=args.models_root,
        gt_root=args.gt_root,
        outdir=args.outdir,
        models=tuple(args.models),
        spacings=tuple(args.spacings),
        sequences=tuple(args.sequences),
        workers=max(1, args.workers),
        normalize=args.normalize,
        ssim_win=max(3, args.ssim_win if args.ssim_win % 2 == 1 else args.ssim_win - 1),
    )
    return cfg


def main() -> int:
    cfg = parse_args()
    setup_logging(cfg.outdir)
    logging.info("Config: %s", dataclasses.asdict(cfg))

    df = compute_all(cfg)
    if df.empty:
        logging.error("No results. Exiting with error.")
        return 2

    make_all_plots(df, cfg.outdir)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
