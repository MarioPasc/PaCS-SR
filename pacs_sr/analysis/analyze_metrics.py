"""
End-to-end metrics computation and statistical analysis for volumetric predictions.

Loads GT and prediction volumes, computes chosen metrics per case/sequence/method,
and performs paired statistical tests with multiple-comparison control and effect sizes.
Results are written as tidy CSV/JSON tables.

Assumptions:
- Filenames follow a template: {case}_{seq}{ext} under gt_dir and each pred_dir.
- Supported formats: .nii.gz, .nii, .npy, .npz. Masks share the same basename as GT.
"""

from __future__ import annotations
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# skimage is standard for PSNR/SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# Optional: nibabel only if using NIfTI
try:
    import nibabel as nib
except Exception:
    nib = None

# Optional: statsmodels for multiple comparisons
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

# Local config parser
from pacs_sr.config.config import AnalysisConfig, parse_analysis_config


# -----------------------------
# I/O utilities
# -----------------------------

def _load_array(path: Path, dtype: str) -> np.ndarray:
    """
    Load an array from NIfTI, NPY, or NPZ. Returns array cast to desired dtype.
    """
    ext = path.suffix if path.suffix != ".gz" else "".join(path.suffixes[-2:])
    if ext in (".nii", ".nii.gz"):
        if nib is None:
            raise RuntimeError("nibabel is required for NIfTI files.")
        arr = np.asanyarray(nib.load(str(path)).get_fdata())  # type: ignore
    elif ext == ".npy":
        arr = np.load(str(path), mmap_mode=None)
    elif ext == ".npz":
        data = np.load(str(path))
        # Heuristic: use array under key 'arr' or the first array
        arr = data["arr"] if "arr" in data.files else data[data.files[0]]
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return arr.astype(dtype, copy=False)


def _maybe_crop(arr: np.ndarray, border: int) -> np.ndarray:
    """
    Crop a constant border in all dimensions if border > 0.
    """
    if border <= 0:
        return arr
    sl = tuple(slice(border, -border if border > 0 else None) for _ in range(arr.ndim))
    return arr[sl]


def _apply_mask(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Flatten voxels within mask if provided; otherwise flatten all voxels.
    """
    if mask is not None:
        return arr[mask > 0].ravel()
    return arr.ravel()


def _normalize(arr: np.ndarray, kind: str, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    """
    Apply normalization by config. Clipping happens before normalization if configured.
    """
    if clip is not None:
        lo, hi = clip
        arr = np.clip(arr, lo, hi, out=arr)
    if kind == "none":
        return arr
    if kind == "zscore":
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        if sd == 0.0:
            return arr * 0.0
        return (arr - mu) / sd
    if kind == "minmax":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi == lo:
            return arr * 0.0
        return (arr - lo) / (hi - lo)
    raise ValueError(f"Unknown normalization kind: {kind}")


# -----------------------------
# Metrics
# -----------------------------

def _data_range(y_true: np.ndarray, y_pred: np.ndarray, cfg_val: str | float) -> float:
    """
    Resolve data range for metrics. If 'auto', uses max of ranges from inputs.
    """
    if cfg_val == "auto":
        ymin = min(float(np.min(y_true)), float(np.min(y_pred)))
        ymax = max(float(np.max(y_true)), float(np.max(y_pred)))
        return max(1e-8, ymax - ymin)
    return float(cfg_val)

def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    diff = y_true - y_pred
    return float(np.mean(diff * diff))

def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(metric_mse(y_true, y_pred)))

def metric_ncc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized cross-correlation."""
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.linalg.norm(yt) * np.linalg.norm(yp)
    if denom == 0.0:
        return 0.0
    return float(np.dot(yt, yp) / denom)

def metric_psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float) -> float:
    """Peak Signal-to-Noise Ratio using skimage."""
    return float(psnr_sk(y_true, y_pred, data_range=data_range))

def metric_ssim(y_true: np.ndarray, y_pred: np.ndarray, data_range: float, win_size: int,
                gaussian_weights: bool, sigma: float, use_sample_covariance: bool) -> float:
    """Structural Similarity Index."""
    # skimage expects images. We provide 3D volumes; it handles N-D.
    return float(
        ssim_sk(  # type: ignore
            y_true, y_pred,
            data_range=data_range,
            win_size=win_size,
            gaussian_weights=gaussian_weights,
            sigma=sigma,
            use_sample_covariance=use_sample_covariance,
            channel_axis=None
        )
    )

# -----------------------------
# Statistical utilities
# -----------------------------

def cohen_dz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's dz for paired samples: mean(diff) / std(diff).
    """
    diff = x - y
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: dominance measure in [-1,1].
    """
    # Efficient O(n log n) via ranking
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    i = j = more = less = 0
    nx, ny = len(x_sorted), len(y_sorted)
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            more += nx - i
            j += 1
        elif x_sorted[i] < y_sorted[j]:
            less += ny - j
            i += 1
        else:
            # Ties: advance both
            i += 1
            j += 1
    denom = nx * ny if nx and ny else 1
    return float((more - less) / denom)

def _fdr_adjust(pvals: np.ndarray, method: str) -> np.ndarray:
    """
    Multiple-comparison correction. Uses statsmodels if available; falls back to BH.
    """
    if method in ("none", None):
        return pvals
    if method == "bonferroni":
        return np.minimum(1.0, pvals * len(pvals))
    if method == "fdr_bh":
        if multipletests is not None:
            _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
            return p_adj
        # BH fallback
        order = np.argsort(pvals)
        ranked = pvals[order]
        n = len(pvals)
        adj = np.empty_like(ranked)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            adj[i] = min(prev, ranked[i] * n / (i + 1))
            prev = adj[i]
        out = np.empty_like(adj)
        out[order] = adj
        return out
    raise ValueError(f"Unknown multiple comparison method: {method}")

# -----------------------------
# Worker
# -----------------------------

def _compute_case_metrics(
    case: str,
    seq: str,
    method: str,
    paths: Dict[str, Path],
    crop_border: int,
    norm_kind: str,
    norm_clip: Optional[Tuple[float, float]],
    dtype: str,
    metrics_cfg: dict,
    mask_path: Optional[Path],
    chunk_voxels: int
) -> Dict[str, object]:
    """
    Compute metrics for one (case, seq, method).
    """
    gt = _load_array(paths["gt"], dtype)
    pr = _load_array(paths["pred"], dtype)
    msk = _load_array(mask_path, dtype="float32") if mask_path else None

    # Crop then normalize
    gt = _maybe_crop(gt, crop_border)
    pr = _maybe_crop(pr, crop_border)
    if msk is not None:
        msk = _maybe_crop(msk, crop_border)

    gt = _normalize(gt, norm_kind, norm_clip)
    pr = _normalize(pr, norm_kind, norm_clip)

    # Chunk to bound memory
    vox = gt.size
    step = max(1, chunk_voxels)
    idxs = list(range(0, vox, step))

    # Prepare accumulators for streaming metrics needing full arrays
    # Most metrics are decomposable or can operate on full arrays safely.
    # For PSNR/SSIM we compute on full arrays; if memory is tight, fall back to chunked approximations.
    # Here we compute directly on arrays for correctness.
    y_true = _apply_mask(gt, msk)
    y_pred = _apply_mask(pr, msk)

    out: Dict[str, object] = {"case": case, "seq": seq, "method": method}
    # Scalar metrics
    if "mae" in metrics_cfg["compute"]:
        out["mae"] = metric_mae(y_true, y_pred)
    if "mse" in metrics_cfg["compute"]:
        out["mse"] = metric_mse(y_true, y_pred)
    if "rmse" in metrics_cfg["compute"]:
        out["rmse"] = metric_rmse(y_true, y_pred)
    if "ncc" in metrics_cfg["compute"]:
        out["ncc"] = metric_ncc(y_true, y_pred)
    if "psnr" in metrics_cfg["compute"]:
        dr = _data_range(y_true, y_pred, metrics_cfg["psnr"]["data_range"])
        out["psnr"] = metric_psnr(y_true, y_pred, dr)
    if "ssim" in metrics_cfg["compute"]:
        # SSIM needs full 3D arrays, not flattened ones
        dr = _data_range(gt, pr, metrics_cfg["ssim"]["data_range"])
        if msk is not None:
            # Apply mask to full arrays for SSIM computation
            gt_masked = np.where(msk > 0, gt, 0)
            pr_masked = np.where(msk > 0, pr, 0)
            out["ssim"] = metric_ssim(
                gt_masked, pr_masked,
                data_range=dr,
                win_size=metrics_cfg["ssim"]["win_size"],
                gaussian_weights=metrics_cfg["ssim"]["gaussian_weights"],
                sigma=metrics_cfg["ssim"]["sigma"],
                use_sample_covariance=metrics_cfg["ssim"]["use_sample_covariance"],
            )
        else:
            out["ssim"] = metric_ssim(
                gt, pr,
                data_range=dr,
                win_size=metrics_cfg["ssim"]["win_size"],
                gaussian_weights=metrics_cfg["ssim"]["gaussian_weights"],
                sigma=metrics_cfg["ssim"]["sigma"],
                use_sample_covariance=metrics_cfg["ssim"]["use_sample_covariance"],
            )
    return out

# -----------------------------
# Orchestration
# -----------------------------

def _discover_cases(cfg: AnalysisConfig) -> List[str]:
    """
    Build case list from file system or explicit list.
    """
    if cfg.data.cases_list:
        txt = Path(cfg.data.cases_list)
        vals = [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        return vals
    # Infer from GT directory
    exts = (".nii", ".nii.gz", ".npy", ".npz")
    names = set()
    for p in Path(cfg.data.gt_dir).rglob(f"*{cfg.data.file_ext}"):
        base = p.name.replace(cfg.data.file_ext, "")
        # Remove trailing _{seq} if present
        for s in cfg.data.sequences:
            suf = f"_{s}"
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        names.add(base)
    return sorted(names)

def _build_paths(cfg: AnalysisConfig, case: str, seq: str, method: str) -> Optional[Dict[str, Path]]:
    """
    Create file paths for GT and prediction for the triplet. Return None if missing and allowed.
    """
    stem = cfg.data.filename_pattern.format(case=case, seq=seq)
    gt = Path(cfg.data.gt_dir) / f"{stem}{cfg.data.file_ext}"
    pr = Path(cfg.data.pred_dirs[method]) / f"{stem}{cfg.data.file_ext}"
    if not gt.exists() or not pr.exists():
        if cfg.data.allow_missing:
            return None
        missing = "GT" if not gt.exists() else "Prediction"
        raise FileNotFoundError(f"{missing} missing for {case} {seq} {method}: {gt if not gt.exists() else pr}")
    return {"gt": gt, "pred": pr}

def run_analysis(yaml_path: str | Path) -> None:
    """
    Main entry: load config, compute metrics in parallel, then statistics and write outputs.
    """
    cfg = parse_analysis_config(yaml_path)
    rng = np.random.default_rng(cfg.runtime.seed)
    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(cfg)
    methods = list(cfg.data.pred_dirs.keys())
    seqs = list(cfg.data.sequences)

    # Prepare tasks
    tasks: List[Tuple[str, str, str, Dict[str, Path], Optional[Path]]] = []
    for case, seq, method in product(cases, seqs, methods):
        paths = _build_paths(cfg, case, seq, method)
        if paths is None:
            continue
        mask_path = None
        if cfg.data.mask_dir is not None:
            stem = cfg.data.filename_pattern.format(case=case, seq=seq)
            mp = Path(cfg.data.mask_dir) / f"{stem}{cfg.data.file_ext}"
            mask_path = mp if mp.exists() else None
        tasks.append((case, seq, method, paths, mask_path))

    # Parallel execution
    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    metrics_cfg = {
        "compute": list(cfg.metrics.compute),
        "crop_border": int(cfg.metrics.crop_border),
        "ssim": {
            "win_size": cfg.metrics.ssim.win_size,
            "gaussian_weights": cfg.metrics.ssim.gaussian_weights,
            "sigma": cfg.metrics.ssim.sigma,
            "use_sample_covariance": cfg.metrics.ssim.use_sample_covariance,
            "data_range": cfg.metrics.ssim.data_range,
        },
        "psnr": {
            "data_range": cfg.metrics.psnr.data_range,
        },
    }

    with ProcessPoolExecutor(max_workers=cfg.runtime.num_workers) as ex:
        fut2meta = {
            ex.submit(
                _compute_case_metrics,
                case, seq, method, paths,
                cfg.metrics.crop_border,
                cfg.data.norm.kind,
                cfg.data.norm.clip,
                cfg.data.dtype,
                metrics_cfg,
                mask_path,
                cfg.runtime.chunk_voxels,
            ): (case, seq, method)
            for (case, seq, method, paths, mask_path) in tasks
        }
        for fut in as_completed(fut2meta):
            case, seq, method = fut2meta[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                msg = f"{case}:{seq}:{method} -> {e}"
                errors.append(msg)
                if cfg.runtime.fail_fast:
                    raise

    if errors:
        (out_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics computed. Check configuration and paths.")

    # Save per-sample metrics
    if cfg.io.write_csv:
        df.to_csv(out_dir / "metrics_per_sample.csv", index=False)
    if cfg.io.write_json:
        (out_dir / "metrics_per_sample.json").write_text(df.to_json(orient="records"), encoding="utf-8")

    # Aggregate
    metric_columns = [m for m in cfg.metrics.compute if m in df.columns]
    if metric_columns:  # Only aggregate if there are metrics to aggregate
        # Use a more explicit aggregation approach
        agg_dict = {}
        for col in metric_columns:
            agg_dict[col] = ["mean", "std", "median"]
        
        grouped = df.groupby(["seq", "method"]).agg(agg_dict)
        grouped.columns = ["_".join(str(c)).strip() for c in grouped.columns.values]
        grouped = grouped.reset_index()
        if cfg.io.write_csv:
            grouped.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Pairwise statistics per sequence
    comps = cfg.stats.compare_methods or list(combinations(methods, 2))
    stat_rows: List[Dict[str, object]] = []
    for seq in seqs:
        dseq = df[df["seq"] == seq]
        # Align by case for paired tests
        for m1, m2 in comps:
            d1 = dseq[dseq["method"] == m1].set_index("case")
            d2 = dseq[dseq["method"] == m2].set_index("case")
            common = d1.index.intersection(d2.index)
            if len(common) == 0:
                continue
            for metric in cfg.metrics.compute:
                if metric not in d1.columns or metric not in d2.columns:
                    continue
                x = d1.loc[common, metric].to_numpy()
                y = d2.loc[common, metric].to_numpy()

                entry = {"seq": seq, "metric": metric, "m1": m1, "m2": m2, "n": int(len(common))}
                pvals = {}

                if "ttest_paired" in cfg.stats.tests and cfg.stats.paired:
                    tstat, p = stats.ttest_rel(x, y, nan_policy="omit")
                    entry["ttest_paired_t"] = float(tstat)
                    pvals["ttest_paired_p"] = float(p)
                if "wilcoxon" in cfg.stats.tests and cfg.stats.paired:
                    try:
                        result = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided")
                        # Handle both old tuple return and new named tuple return
                        try:
                            # Try new scipy version with named tuple
                            entry["wilcoxon_W"] = float(result.statistic)  # type: ignore
                            pvals["wilcoxon_p"] = float(result.pvalue)  # type: ignore
                        except AttributeError:
                            # Fall back to old scipy version with tuple unpacking
                            wstat, p = result  # type: ignore
                            entry["wilcoxon_W"] = float(wstat)  # type: ignore
                            pvals["wilcoxon_p"] = float(p)  # type: ignore
                    except ValueError:
                        entry["wilcoxon_W"] = float("nan")
                        pvals["wilcoxon_p"] = float("nan")

                if "cohens_dz" in cfg.stats.effect_sizes:
                    entry["cohens_dz"] = cohen_dz(x, y)
                if "cliffs_delta" in cfg.stats.effect_sizes:
                    entry["cliffs_delta"] = cliffs_delta(x, y)

                # Bootstrap CI for mean difference
                if cfg.stats.bootstrap.enabled:
                    diff = x - y
                    nres = cfg.stats.bootstrap.n_resamples
                    ci = cfg.stats.bootstrap.ci
                    idx = np.arange(diff.size)
                    boots = np.mean(diff[r]) if (r := np.random.default_rng().choice(idx, size=(nres, diff.size), replace=True)).size == 0 else np.mean(diff[r], axis=1)
                    lo = float(np.quantile(boots, (1 - ci) / 2))
                    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
                    entry["boot_mean_diff_lo"] = lo
                    entry["boot_mean_diff_hi"] = hi

                entry.update(pvals)
                stat_rows.append(entry)

    stats_df = pd.DataFrame(stat_rows)
    if not stats_df.empty:
        # Adjust p-values per metric and test
        if cfg.stats.multiple_comparison != "none":
            for col in [c for c in stats_df.columns if c.endswith("_p")]:
                mask = stats_df[col].notna()
                padj = _fdr_adjust(stats_df.loc[mask, col].to_numpy(dtype=float), cfg.stats.multiple_comparison)
                stats_df.loc[mask, col.replace("_p", "_p_adj")] = padj

        if cfg.io.write_csv:
            stats_df.to_csv(out_dir / "stats_pairwise.csv", index=False)
        if cfg.io.write_json:
            (out_dir / "stats_pairwise.json").write_text(stats_df.to_json(orient="records"), encoding="utf-8")
