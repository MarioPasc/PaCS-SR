#!/usr/bin/env python3
# analysis.py — SR benchmarking: robust, parallel, reproducible
# Sections:
#   0) Imports and globals
#   1) Configuration, errors, logging
#   2) I/O helpers and sanity checks
#   3) Data shaping
#   4) Linear mixed models (EMM + contrasts)
#   5) Nonparametric confirmation (Friedman + Wilcoxon)
#   6) Radiomics engines (IBSI via PyRadiomics, light FO/GLCM)
#   7) Parallel workers
#   8) Orchestration and CLI

from __future__ import annotations

# ------------------------ 0) Imports and globals ---------------------------
import os, sys, json, math, logging, argparse, pathlib, warnings, gc
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
import patsy  # type: ignore
import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm # type: ignore
from itertools import repeat
import json
from nibabel import filebasedimages as _nib_file_err

# NEW: classify I/O errors we will trap
IO_ERRORS = (EOFError, OSError, _nib_file_err.ImageFileError, ValueError)

import statsmodels.formula.api as smf # type: ignore
from statsmodels.stats.multitest import multipletests # type: ignore

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# Optional IBSI stack
try:
    import SimpleITK as sitk  # type: ignore
    from radiomics import featureextractor  # type: ignore
    _HAS_PYRADIOMICS = True
except Exception:
    _HAS_PYRADIOMICS = False

# --------------------- 1) Configuration, errors, logging -------------------
@dataclass(frozen=True)
class Paths:
    """CLI-resolved paths used across the pipeline."""
    metrics_npz: pathlib.Path
    hr_root: pathlib.Path
    results_root: pathlib.Path
    out_npz: pathlib.Path
    baseline_model: Optional[str] = None

class PipelineError(RuntimeError):
    """User-facing fatal error for the SR stats pipeline."""

PRIMARY_BY_PULSE: Dict[str, str] = {"t1c": "SSIM", "t2w": "SSIM", "t2f": "PSNR", "t1n": "SSIM"}
ROI_LABELS: Tuple[str, ...] = ("all", "core", "edema", "surround")

# IBSI feature allow-list for stability and portability across pyradiomics versions
_IBSI_KEEP = {
    # first-order
    "firstorder_Energy","firstorder_TotalEnergy","firstorder_Entropy","firstorder_Mean",
    "firstorder_Median","firstorder_Variance","firstorder_Skewness","firstorder_Kurtosis",
    # GLCM
    "glcm_Autocorrelation","glcm_ClusterShade","glcm_ClusterProminence","glcm_Contrast",
    "glcm_Correlation","glcm_Dissimilarity","glcm_Idmn","glcm_Idn","glcm_Imc1","glcm_Imc2",
    "glcm_Homogeneity1",
    # GLRLM
    "glrlm_ShortRunEmphasis","glrlm_LongRunEmphasis","glrlm_GrayLevelNonUniformity",
    "glrlm_RunLengthNonUniformity","glrlm_RunPercentage","glrlm_HighGrayLevelRunEmphasis",
    "glrlm_LowGrayLevelRunEmphasis",
    # GLSZM
    "glszm_SmallAreaEmphasis","glszm_LargeAreaEmphasis","glszm_GrayLevelNonUniformity",
    "glszm_ZoneSizeNonUniformity","glszm_ZoneEntropy","glszm_ZoneVariance",
    # GLDM
    "gldm_DependenceNonUniformity","gldm_DependenceEntropy","gldm_LargeDependenceEmphasis",
    "gldm_SmallDependenceEmphasis","gldm_HighGrayLevelEmphasis","gldm_LowGrayLevelEmphasis",
    # NGTDM
    "ngtdm_Coarseness","ngtdm_Contrast","ngtdm_Busyness","ngtdm_Complexity","ngtdm_Strength",
    # Shape
    "shape_VoxelVolume","shape_SurfaceArea","shape_Sphericity","shape_Elongation",
    "shape_Flatness","shape_Compactness2",
}

# Use your SR logger if available; else a local logger
LOG = logging.getLogger("sr.analysis") # type: ignore

def _configure_logging() -> None:
    """Console logger. High signal format for reproducibility."""
    if not LOG.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[SR:analysis] %(levelname)s | %(message)s"))
        LOG.addHandler(h)
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False

# ----------------------- 2) I/O helpers and sanity checks ------------------
def load_metrics_npz(path: pathlib.Path) -> Dict[str, Any]:
    """Load metrics.npz with object arrays decoded to Python containers."""
    d = np.load(path, allow_pickle=True)
    out = {k: (d[k].tolist() if d[k].dtype == object else d[k]) for k in d.files}
    return out

def _strict_load(path: pathlib.Path) -> nib.Nifti1Image:
    """
    Load NIfTI with mmap disabled and force data read+scaling to surface I/O errors.
    Returns a canonicalized image if successful, else raises an IO_ERRORS exception.
    """
    img = nib.load(str(path), mmap=False)
    _ = img.get_fdata(dtype=np.float32)  # force decompress + scale now
    return nib.as_closest_canonical(img)

def to_jsonable(obj: Any) -> Any:
    """Cast numpy/pandas to Python built-ins for JSON serialization."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj

def sanity_check_metrics(metrics: Dict[str, Any],
                         hr_root: pathlib.Path,
                         results_root: pathlib.Path) -> Dict[str, Any]:
    """
    Validate metrics integrity and on-disk volumes. Flags shape mismatches, NaNs,
    out-of-range metrics, incomplete panels, and missing/misaligned NIfTI files.
    """
    report: Dict[str, Any] = {}
    arr = metrics.get("metrics")
    ok = isinstance(arr, np.ndarray) and arr.ndim == 7
    report["shape_ok"] = bool(ok)
    if not ok:
        LOG.error("Bad metrics array. Expected 7-D, got %s", None if arr is None else arr.shape)
        return report

    expected = (len(metrics.get("patient_ids", [])),
                len(metrics.get("pulses", [])),
                len(metrics.get("resolutions_mm", [])),
                len(metrics.get("models", [])),
                len(metrics.get("metric_names", [])),
                len(metrics.get("roi_labels", [])),
                len(metrics.get("stat_names", [])))
    report["observed_shape"] = tuple(arr.shape) # type: ignore
    report["expected_shape"] = expected
    if tuple(arr.shape) != expected: # type: ignore
        LOG.warning("Shape mismatch. observed=%s expected=%s", arr.shape, expected) # type: ignore

    # NaN rate on mean statistic
    stat_names: List[str] = list(metrics.get("stat_names", []))
    mean_idx = stat_names.index("mean") if "mean" in stat_names else 0
    mean_vals = arr[..., mean_idx] # type: ignore
    nan_frac = float(np.isnan(mean_vals).mean()) if mean_vals.size else 1.0
    report["nan_fraction"] = nan_frac
    if nan_frac > 0:
        LOG.info("NaNs in metrics (mean): %.2f%%", 100 * nan_frac)

    # Ranges
    oor: Dict[str, float] = {}
    metric_names = list(metrics.get("metric_names", []))
    for m_i, m_name in enumerate(metric_names):
        vals = mean_vals[..., m_i, :]
        finite = np.isfinite(vals)
        bad = 0
        if m_name.upper() == "SSIM":
            bad = int(((vals[finite] < 0) | (vals[finite] > 1)).sum())
        elif m_name.upper() == "PSNR":
            bad = int(((vals[finite] <= 0) | (vals[finite] > 100)).sum())
        oor[m_name] = float(bad)
    report["out_of_range_counts"] = oor

    # Completeness per (patient, roi)
    df = to_long_df(metrics)
    incomplete: Dict[str, int] = {}
    for pulse in list(metrics.get("pulses", [])):
        primary = PRIMARY_BY_PULSE.get(pulse)
        if primary is None:
            continue
        sub = df[(df["pulse"] == pulse) & (df["metric"] == primary)].copy()
        full_n = sub["model"].nunique() * sub["resolution"].nunique()
        cells = (sub.assign(cell=sub["model"].astype(str) + "|" + sub["resolution"].astype(str))
                    .groupby(["patient", "roi"])["cell"].nunique())
        incomplete[pulse] = int((cells < full_n).sum())
    report["incomplete_panels"] = incomplete

    # Files on disk and shapes
    miss: List[str] = []
    mism: List[str] = []
    patients = list(metrics.get("patient_ids", []))
    pulses = list(metrics.get("pulses", []))
    models = list(metrics.get("models", []))
    resolutions = list(metrics.get("resolutions_mm", []))
    for pid in tqdm(patients, desc="Verifying volumes", unit="patient"):
        seg_p = hr_root / pid / f"{pid}-seg.nii.gz"
        if not seg_p.exists():
            miss.append(str(seg_p))
        hr_imgs: Dict[str, nib.Nifti1Image] = {}
        for pulse in pulses:
            p = hr_root / pid / f"{pid}-{pulse}.nii.gz"
            if not p.exists():
                miss.append(str(p)); continue
            try:
                hr_imgs[pulse] = nib.as_closest_canonical(nib.load(str(p)))
            except Exception:
                miss.append(str(p))
        for pulse in pulses:
            for res in resolutions:
                for model in models:
                    sr = results_root / model / f"{res}mm" / "output_volumes" / f"{pid}-{pulse}.nii.gz"
                    if not sr.exists():
                        miss.append(str(sr)); continue
                    if pulse in hr_imgs:
                        try:
                            sh = nib.load(str(sr)).shape # type: ignore
                            if np.any(np.abs(np.subtract(hr_imgs[pulse].shape, sh)) > 2):
                                mism.append(f"{pid}-{pulse}-{res}-{model}: HR{hr_imgs[pulse].shape} vs SR{sh}")
                        except Exception:
                            mism.append(str(sr))
    report["missing_volumes"] = sorted(set(miss))
    report["mismatched_shapes"] = mism
    if miss: LOG.warning("Missing volumes: %d", len(miss))
    if mism: LOG.warning("Shape mismatches > 2 vox: %d", len(mism))
    return report

# ------------------------------ 3) Data shaping ----------------------------
def to_long_df(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a tidy patient-level frame with the 'mean' stat across slices.
    Returns columns: patient, pulse, resolution, model, metric, roi, value.
    """
    arr: np.ndarray = metrics["metrics"]
    stat_names: List[str] = list(metrics["stat_names"])
    mean_idx = stat_names.index("mean")
    P, PUL, RES, MOD, MET, ROI, _ = arr.shape

    # Cartesian product index
    idx = pd.MultiIndex.from_product(
        [metrics["patient_ids"], metrics["pulses"], metrics["resolutions_mm"],
         metrics["models"], metrics["metric_names"], metrics["roi_labels"]],
        names=["patient", "pulse", "resolution", "model", "metric", "roi"]
    )
    vals = arr[..., mean_idx].reshape(-1)
    df = pd.DataFrame({"value": vals}, index=idx).reset_index()
    df = df[np.isfinite(df["value"])]
    return df

def pack_table(df: pd.DataFrame, label: str) -> np.recarray:
    """Convert mixed-dtype DataFrame to recarray for compact NPZ storage."""
    if df.empty:
        LOG.warning("pack_table[%s]: empty", label)
        return np.recarray(0, dtype=[])
    rec = df.to_records(index=False)
    LOG.debug("pack_table[%s]: dtype=%s", label, rec.dtype)
    return rec

# ---------------- 4) Linear mixed models (EMM + contrasts) -----------------
def fe_series(fit) -> pd.Series:
    """Return named FE parameter vector for MixedLM or OLS-robust fallback."""
    b = fit.fe_params if hasattr(fit, "fe_params") else fit.params
    return b if isinstance(b, pd.Series) else pd.Series(np.asarray(b).ravel(), index=fit.model.exog_names, dtype=float)

def fe_cov(fit, names: List[str]) -> pd.DataFrame:
    """Return FE covariance aligned to `names` order."""
    V = fit.cov_params()
    if isinstance(V, pd.DataFrame):
        return V.reindex(index=names, columns=names)
    V = np.asarray(V)
    p = len(names)
    V = V[:p, :p]
    return pd.DataFrame(V, index=names, columns=names)

def ensure_psd(V_df: pd.DataFrame, eps: float = 1e-9) -> np.ndarray:
    """Eigen-clip to nearest PSD."""
    M = V_df.to_numpy(dtype=float)
    M = 0.5 * (M + M.T)
    w, Q = np.linalg.eigh(np.nan_to_num(M))
    w = np.clip(w, a_min=eps, a_max=None)
    return (Q * w) @ Q.T

def build_fe_X(fit, synth: pd.DataFrame, rhs: str, names: List[str]) -> np.ndarray:
    """DMATRIX for `1 + rhs` aligned to FE parameter names `names`."""
    X = patsy.dmatrix("1 + " + rhs, synth, return_type="dataframe")
    for c in names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[names]
    return X.to_numpy(dtype=float)

def fit_lmm_primary(df: pd.DataFrame, pulse: str, baseline_model: Optional[str] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    MixedLM on the primary endpoint of `pulse`.
    Fixed: C(model)*C(resolution)+C(roi). Random: intercept per subject.
    EMM over ROI and pairwise contrasts vs baseline (default: BSPLINE if present, else first model).
    """
    primary = PRIMARY_BY_PULSE[pulse]
    sub = df[(df["pulse"] == pulse) & (df["metric"] == primary)].copy()
    if sub.empty:
        return pd.DataFrame(), []

    sub["subject"] = sub["patient"].astype("category")
    sub["roi_c"] = sub["roi"].astype("category")
    sub["resolution_c"] = sub["resolution"].astype("category")
    sub["model_c"] = sub["model"].astype("category")

    # keep complete panels
    full_n = sub["model_c"].nunique() * sub["resolution_c"].nunique()
    cnt = (sub.assign(cell=sub["model_c"].astype(str) + "|" + sub["resolution_c"].astype(str))
             .groupby(["patient", "roi"])["cell"].nunique())
    keep = cnt[cnt == full_n].index
    sub = sub.set_index(["patient", "roi"]).loc[keep].reset_index()
    if sub.empty:
        raise PipelineError("No complete (patient, roi) panels for LMM.")

    lev_roi = sub["roi_c"].cat.categories
    lev_mod = sub["model_c"].cat.categories
    lev_res = sub["resolution_c"].cat.categories
    
    # Determine baseline model
    if baseline_model and baseline_model in lev_mod:
        baseline = baseline_model
    elif "BSPLINE" in lev_mod:
        baseline = "BSPLINE"
    else:
        baseline = lev_mod[0]
    LOG.info("Using baseline model: %s (pulse=%s)", baseline, pulse)
    
    fe_rhs = "C(model_c)*C(resolution_c) + C(roi_c)"

    fit = None
    for formula in (f"value ~ {fe_rhs}", "value ~ C(model_c) + C(resolution_c) + C(roi_c)"):
        try:
            m = smf.mixedlm(formula, data=sub, groups=sub["subject"])
            fit = m.fit(method="lbfgs", reml=True)
            break
        except Exception as e:
            LOG.error("MixedLM failed on '%s' → %s", formula, type(e).__name__)
    if fit is None:
        import statsmodels.api as sm # type: ignore
        ols = smf.ols(f"value ~ {fe_rhs}", data=sub).fit()
        fit = ols.get_robustcov_results(cov_type="cluster", groups=sub["subject"])
        class _Wrap:
            fe_params = fit.params # type: ignore
            def cov_params(self): return fit.cov_params()
            class _M: exog_names = fit.model.exog_names # type: ignore
            model = _M()
        fit = _Wrap()
        LOG.info("Fallback to OLS with cluster-robust SEs.")

    b = fe_series(fit); names = list(b.index); V = ensure_psd(fe_cov(fit, names))
    emm_rows: List[Dict[str, Any]] = []
    contrasts: List[Dict[str, Any]] = []
    tmp: List[Tuple[Any, Any, float, float, float, float]] = []
    pvals: List[float] = []

    # EMM per (model,res) marginal over ROI levels
    for r in tqdm(lev_res, desc=f"EMM[{pulse}]", leave=False):
        for m in lev_mod:
            synth = pd.DataFrame({
                "model_c": pd.Categorical([m]*len(lev_roi), categories=lev_mod),
                "resolution_c": pd.Categorical([r]*len(lev_roi), categories=lev_res),
                "roi_c": pd.Categorical(list(lev_roi), categories=lev_roi)
            })
            X = build_fe_X(fit, synth, fe_rhs, names)
            pred = X @ b.values
            mean = float(np.mean(pred))
            row_vars = np.clip(np.einsum("ij,jk,ik->i", X, V, X), 0.0, None)
            se = float(np.sqrt(np.mean(row_vars)))
            zc = stats.norm.ppf(0.975)
            emm_rows.append({"pulse": pulse, "resolution": str(r), "model": str(m),
                             "n_subjects": sub["subject"].nunique(),
                             "mean": mean, "se": se, "lcl": mean - zc*se, "ucl": mean + zc*se})

    # Pairwise vs baseline at each resolution
    for r in tqdm(lev_res, desc=f"Contrasts[{pulse}]", leave=False):
        for m in lev_mod:
            if m == baseline: continue
            synth = pd.DataFrame({
                "model_c": pd.Categorical([m, baseline], categories=lev_mod),
                "resolution_c": pd.Categorical([r, r], categories=lev_res),
                "roi_c": pd.Categorical([lev_roi[0], lev_roi[0]], categories=lev_roi)
            })
            X2 = build_fe_X(fit, synth, fe_rhs, names)
            c = (X2[0] - X2[1]).reshape(-1, 1)
            est = float((c.T @ b.values)[0])
            var = float((c.T @ V @ c)[0, 0]); var = max(var, 0.0)
            se = math.sqrt(var) if var > 0 else float("nan")
            z = est / se if np.isfinite(se) else float("nan")
            p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
            tmp.append((r, m, est, se, z, p)); pvals.append(p)

    padj = multipletests(pvals, alpha=0.05, method="holm")[1].tolist() if pvals else []
    for (r, m, est, se, z, p), p_h in zip(tmp, padj):
        r_val = int(r)
        paired = sub[(sub["resolution"] == r_val) & (sub["model"] == m)]
        base = sub[(sub["resolution"] == r_val) & (sub["model"] == baseline)]
        if paired.empty or base.empty:
            contrasts.append({"pulse": pulse, "resolution": str(r), "model": str(m),
                              "baseline": baseline, "estimate": est, "se": se,
                              "z": z, "p_raw": p, "p_holm": p_h,
                              "hedges_g": np.nan, "g_ci": (np.nan, np.nan), "n_pairs": 0})
            continue
        merged = pd.merge(
            paired[["patient","roi","value"]].rename(columns={"value":"v1"}),
            base[["patient","roi","value"]].rename(columns={"value":"v0"}),
            on=["patient","roi"], how="inner"
        )
        d = (merged["v1"] - merged["v0"]).to_numpy()
        n = d.size
        sd = np.std(d, ddof=1) if n > 1 else np.nan
        dz = np.mean(d) / sd if sd and sd > 0 else np.nan
        J = 1 - 3/(4*n - 1) if n > 1 else np.nan
        g = dz * J if np.isfinite(dz) else np.nan
        if n > 1 and np.isfinite(dz):
            se_dz = math.sqrt((1/n) + (dz**2)/(2*(n-1)))
            zcrit = stats.norm.ppf(0.975)
            g_ci = (g - zcrit*se_dz*J, g + zcrit*se_dz*J)
        else:
            g_ci = (np.nan, np.nan)
        contrasts.append({"pulse": pulse, "resolution": str(r), "model": str(m),
                          "baseline": baseline, "estimate": est, "se": se, "z": z,
                          "p_raw": p, "p_holm": p_h, "hedges_g": g, "g_ci": g_ci, "n_pairs": int(n)})
    return pd.DataFrame(emm_rows), contrasts

# -------- 5) Nonparametric confirmation (Friedman + Wilcoxon) -------------
def holm_adjust(pvals: List[float]) -> List[float]:
    """Holm step-down family-wise adjustment."""
    return multipletests(pvals, alpha=0.05, method="holm")[1].tolist() if pvals else []

def friedman_wilcoxon(df: pd.DataFrame, pulse: str, baseline_model: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Omnibus Friedman per (resolution, roi) and Wilcoxon vs baseline per model."""
    primary = PRIMARY_BY_PULSE[pulse]
    sub = df[(df["pulse"] == pulse) & (df["metric"] == primary)].copy()
    if sub.empty:
        return [], []
    models = sorted(sub["model"].unique().tolist())
    
    # Determine baseline model
    if baseline_model and baseline_model in models:
        baseline = baseline_model
    elif "BSPLINE" in models:
        baseline = "BSPLINE"
    else:
        baseline = models[0]
    LOG.debug("Friedman/Wilcoxon baseline: %s (pulse=%s)", baseline, pulse)
    fr_out: List[Dict[str, Any]] = []
    wi_out: List[Dict[str, Any]] = []

    for res in sorted(sub["resolution"].unique()):
        for roi in ROI_LABELS:
            mask = (sub["resolution"] == res) & (sub["roi"] == roi)
            piv = (sub.loc[mask]
                      .pivot_table(index="patient", columns="model", values="value", aggfunc="mean")
                      .dropna(axis=0, how="any"))
            if piv.shape[0] < 3 or piv.shape[1] < 2:
                continue
            cols = [m for m in models if m in piv.columns]
            try:
                chi2, p = stats.friedmanchisquare(*[piv[m].to_numpy(float) for m in cols])
                fr_out.append({"pulse": pulse, "resolution": int(res), "roi": roi,
                               "n": int(piv.shape[0]), "chi2": float(chi2), "p_raw": float(p)})
            except Exception:
                continue
            pairs, p_raws = [], []
            for m in cols:
                if m == baseline: continue
                x = piv[m].to_numpy(float); y = piv[baseline].to_numpy(float)
                if x.size < 5: continue
                W, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="approx")
                diffs = x - y
                n = diffs.size
                sd = np.std(diffs, ddof=1) if n > 1 else np.nan
                dz = np.mean(diffs) / sd if sd and sd > 0 else np.nan
                pairs.append((m, W, p, n, dz)); p_raws.append(p)
            if not pairs:
                continue
            adj = holm_adjust(p_raws)
            for (m, W, p, n, dz), p_h in zip(pairs, adj):
                sign = np.sign(np.nanmean(piv[m] - piv[baseline]))
                z = stats.norm.ppf(1 - p/2) * sign if np.isfinite(p) and p > 0 else np.nan
                if n > 1 and np.isfinite(dz):
                    se_dz = math.sqrt((1/n) + (dz**2)/(2*(n-1)))
                    zcrit = stats.norm.ppf(0.975)
                    dz_ci = (dz - zcrit*se_dz, dz + zcrit*se_dz)
                else:
                    dz_ci = (np.nan, np.nan)
                wi_out.append({"pulse": pulse, "resolution": int(res), "roi": roi,
                               "model": m, "baseline": baseline, "n": int(n),
                               "W": float(W), "z": float(z), "p_raw": float(p),
                               "p_holm": float(p_h), "dz": float(dz) if np.isfinite(dz) else np.nan,
                               "dz_ci": dz_ci})
    return fr_out, wi_out

# --------------------- 6) Radiomics engines and ICC(2,1) -------------------
def _as_float(img: nib.Nifti1Image) -> np.ndarray:
    return img.get_fdata(dtype=np.float32)

def _err_record(stage: str, paths: Dict[str, str], exc: Exception) -> Dict[str, Any]:
    return {"__error__": True, "stage": stage, "paths": paths, "error": f"{type(exc).__name__}: {exc}"}


def _pad_or_crop_to(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center pad/crop ≤2 voxels per axis to match `target`."""
    dif = np.array(target) - np.array(arr.shape)
    if np.all(dif == 0):
        return arr
    if np.any(np.abs(dif) > 2):
        LOG.error("Refusing pad/crop. Δshape=%s target=%s", dif.tolist(), target); return arr
    out = np.pad(arr,
                 ((max(0, dif[0]//2), max(0, dif[0]-dif[0]//2)),
                  (max(0, dif[1]//2), max(0, dif[1]-dif[1]//2)),
                  (max(0, dif[2]//2), max(0, dif[2]-dif[2]//2))),
                 mode="constant", constant_values=0)
    for ax, d in enumerate(dif):
        if d < 0:
            L = (-d)//2; R = (-d) - L
            sl = [slice(None)]*3; sl[ax] = slice(L, out.shape[ax]-R)
            out = out[tuple(sl)]
    return out

def load_align_triplet(hr_p: pathlib.Path, seg_p: pathlib.Path, sr_p: pathlib.Path
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hr_i = _strict_load(hr_p)
    seg_i = _strict_load(seg_p)
    sr_i = _strict_load(sr_p)
    seg_r = resample_from_to(seg_i, hr_i, order=0)
    sr_r  = resample_from_to(sr_i,  hr_i, order=1)
    hr = _as_float(hr_i); seg = _as_float(seg_r); sr = _as_float(sr_r)
    if seg.shape != hr.shape or sr.shape != hr.shape:
        LOG.warning("Pad/crop to HR grid. HR=%s SEG=%s SR=%s", hr.shape, seg.shape, sr.shape)
        seg = _pad_or_crop_to(seg, hr.shape); sr = _pad_or_crop_to(sr, hr.shape)
    return hr, sr, seg

def roi_mask(seg: np.ndarray, label: Optional[int]) -> np.ndarray:
    """ROI mask. None → union of tumor labels >0."""
    return (seg > 0) if label is None else (seg == label)

def first_order(a: np.ndarray) -> Dict[str, float]:
    """Robust FO inside mask."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: np.nan for k in ["mean","var","skew","kurt","entropy"]}
    hist, _ = np.histogram(a, bins=256, density=True)
    p = hist / max(np.sum(hist), 1.0); p = p[p > 0]
    return {
        "mean": float(np.mean(a)),
        "var": float(np.var(a, ddof=1)) if a.size > 1 else np.nan,
        "skew": float(stats.skew(a, bias=False)) if a.size > 2 else np.nan,
        "kurt": float(stats.kurtosis(a, fisher=True, bias=False)) if a.size > 3 else np.nan,
        "entropy": float(-np.sum(p * np.log2(p)))
    }

def glcm_slice_feats(im2d: np.ndarray, m2d: np.ndarray) -> Dict[str, float]:
    """Texture on axial slice, masked, 32 levels, 1-pixel distance."""
    if not m2d.any():
        return {"contr": np.nan, "homog": np.nan, "corr": np.nan}
    vals = im2d[m2d]
    vmin, vmax = np.percentile(vals, [1, 99]) if vals.size else (np.nan, np.nan)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return {"contr": np.nan, "homog": np.nan, "corr": np.nan}
    q = np.clip(((im2d - vmin) / (vmax - vmin) * 31).astype(np.uint8), 0, 31)
    G = graycomatrix(q, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    return {
        "contr": float(graycoprops(G, "contrast")[0, 0]),
        "homog": float(graycoprops(G, "homogeneity")[0, 0]),
        "corr":  float(graycoprops(G, "correlation")[0, 0]),
    }

def _icc2_1_two_raters(hr: np.ndarray, sr: np.ndarray) -> float:
    """ICC(2,1) closed-form for two 'raters' HR vs SR."""
    X = np.vstack([hr, sr]).T
    X = X[np.all(np.isfinite(X), axis=1)]
    n = X.shape[0]
    if n < 2:
        return np.nan
    grand = X.mean()
    m_rows = X.mean(axis=1, keepdims=True)
    m_cols = X.mean(axis=0, keepdims=True)
    ss_total = ((X - grand)**2).sum()
    ss_rows = 2 * ((m_rows - grand)**2).sum()
    ss_cols = n * ((m_cols - grand)**2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (2 - 1)
    ms_err = ss_err / ((n - 1) * (2 - 1))
    denom = ms_rows + (2 - 1)*ms_err + 2*(ms_cols - ms_err)/n
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float((ms_rows - ms_err) / denom)

# --------------------------- 7) Parallel workers ---------------------------
# PyRadiomics: per-process extractor via initializer to avoid pickling
_PYRAD_SETTINGS: Optional[Dict[str, Any]] = None
_PYRAD_EXTRACTOR: Any = None

def _pyrad_init(settings: Dict[str, Any]) -> None:
    """Process initializer: cache settings and build extractor once."""
    global _PYRAD_SETTINGS, _PYRAD_EXTRACTOR
    _PYRAD_SETTINGS = settings
    if _HAS_PYRADIOMICS:
        _PYRAD_EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(**settings)
        _PYRAD_EXTRACTOR.disableAllFeatures()
        for cls in ("firstorder","glcm","glrlm","glszm","gldm","ngtdm","shape"):
            _PYRAD_EXTRACTOR.enableFeatureClassByName(cls)

def _clean_ibsi(d: Dict[str, Any], prefix: str) -> Dict[str, float]:
    """Subset pyradiomics output to _IBSI_KEEP and add HR_/SR_ prefix."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        if k in _IBSI_KEEP and isinstance(v, (int, float)) and np.isfinite(v):
            out[f"{k}_{prefix}"] = float(v)
    return out

# UPDATED: _worker_ibsi_one — same robust handling
def _worker_ibsi_one(triple: Tuple[pathlib.Path, pathlib.Path, pathlib.Path],
                     pulse: str, res: int, model: str,
                     roi_map: Dict[str, Optional[int]]) -> List[Dict[str, Any]] | Dict[str, Any]:
    hr_p, seg_p, sr_p = triple
    try:
        hr, sr, seg = load_align_triplet(hr_p, seg_p, sr_p)
    except IO_ERRORS as e:
        return _err_record(
            stage="ibsi_load",
            paths={"hr": str(hr_p), "seg": str(seg_p), "sr": str(sr_p),
                   "pulse": pulse, "res": str(res), "model": model},
            exc=e,
        )
    pid = hr_p.parent.name
    rows: List[Dict[str, Any]] = []
    if not _HAS_PYRADIOMICS:
        return rows
    for roi_name, label in roi_map.items():
        m = roi_mask(seg, label)
        if not m.any():
            continue
        try:
            sitk_mask = sitk.GetImageFromArray(m.astype(np.uint8))
            img_hr = sitk.GetImageFromArray(hr.astype(np.float32))
            img_sr = sitk.GetImageFromArray(sr.astype(np.float32))
            res_hr = _PYRAD_EXTRACTOR.execute(img_hr, sitk_mask)
            res_sr = _PYRAD_EXTRACTOR.execute(img_sr, sitk_mask)
        except Exception as e:
            # return a per-ROI error record rather than failing the subject
            rows.append(_err_record(
                stage="ibsi_execute",
                paths={"patient": pid, "roi": roi_name, "pulse": pulse,
                       "res": str(res), "model": model},
                exc=e,
            ))
            continue
        feats = {}
        feats.update(_clean_ibsi(res_hr, "HR"))
        feats.update(_clean_ibsi(res_sr, "SR"))
        if feats:
            row = {"patient": pid, "pulse": pulse, "resolution": int(res),
                   "model": model, "roi": roi_name}
            row.update(feats)
            rows.append(row)
    return rows


def _dispatch_ibsi(args):
    """Unpack tuple and call _worker_ibsi_one."""
    triple, pulse, res, model, roi_map = args
    return _worker_ibsi_one(triple, pulse, res, model, roi_map)


def run_radiomics_ibsi(paths: Paths,
                       pulses: List[str], resolutions: List[int], models: List[str],
                       ibsi_bin_width: float, ibsi_distances: List[int],
                       ibsi_min_roi_vox: int, workers: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    PyRadiomics IBSI features (HR/SR) per ROI. Parallel by subject with safe per-process extractor.
    """
    errors: List[Dict[str, Any]] = []
    if not _HAS_PYRADIOMICS:
        LOG.error("Install 'pyradiomics' + 'SimpleITK' for IBSI features.")
        return {}

    settings = {
        "normalize": False, "resampledPixelSpacing": None, "interpolator": "sitkBSpline",
        "force2D": False, "force2Ddimension": 0, "distances": list(ibsi_distances),
        "symmetricalGLCM": True, "minimumROIDimensions": 2, "minimumROISize": ibsi_min_roi_vox,
        "resegmentRange": None, "additionalInfo": True,
        **({"binWidth": float(ibsi_bin_width)} if ibsi_bin_width and ibsi_bin_width > 0 else {})
    }
    out: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pulses}
    roi_map = {"all": None, "core": 1, "edema": 2, "surround": 3}
    
    ctx = mp.get_context("spawn")
    for pulse in pulses:
        for res in resolutions:
            for model in models:
                pairs = collect_paths(paths.hr_root, paths.results_root, pulse, model, res)
                if not pairs:
                    continue
                with ctx.Pool(processes=max(1, workers),
                              initializer=_pyrad_init, initargs=(settings,)) as pool:
                    tasks = ((tr, pulse, res, model, roi_map) for tr in pairs)
                    it = pool.imap_unordered(
                        _dispatch_ibsi, tasks,
                        chunksize=max(1, len(pairs)//(4*max(1, workers)) or 1),
                    )
                    for rows in tqdm(it, total=len(pairs),
                                     desc=f"IBSI {pulse} {res}mm {model}", unit="subj"):
                        if isinstance(rows, dict) and rows.get("__error__"):
                            errors.append(rows); LOG.warning("IBSI skip: %s", rows["error"]); continue
                        out[pulse].extend(rows)
                gc.collect()
    if errors:
        err_path = paths.out_npz.parent / (paths.out_npz.stem + "_ibsi_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        LOG.info("IBSI errors logged → %s (%d records)", err_path, len(errors))
    return out

def collect_paths(hr_root: pathlib.Path, results_root: pathlib.Path,
                  pulse: str, model: str, res_mm: int
                  ) -> List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]]:
    """Collect (HR, SEG, SR) triplets present in both roots."""
    sr_dir = results_root / model / f"{res_mm}mm" / "output_volumes"
    out: List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = []
    for patient_dir in sorted(hr_root.iterdir()):
        pid = patient_dir.name
        hr_p = patient_dir / f"{pid}-{pulse}.nii.gz"
        seg_p = patient_dir / f"{pid}-seg.nii.gz"
        sr_p = sr_dir / f"{pid}-{pulse}.nii.gz"
        if hr_p.exists() and seg_p.exists() and sr_p.exists():
            out.append((hr_p, seg_p, sr_p))
    return out

def _dispatch_one_subject(args):
    """Unpack tuple and call _one_subject."""
    triple, pulse = args
    return _one_subject(triple, pulse)

def _one_subject(triple: Tuple[pathlib.Path, pathlib.Path, pathlib.Path],
                 pulse: str) -> Tuple[str, Dict[str, Dict[str, float]]] | Dict[str, Any]:
    hr_p, seg_p, sr_p = triple
    try:
        hr, sr, seg = load_align_triplet(hr_p, seg_p, sr_p)
    except IO_ERRORS as e:
        return _err_record(
            stage="icc_light_load",
            paths={"hr": str(hr_p), "seg": str(seg_p), "sr": str(sr_p), "pulse": pulse},
            exc=e,
        )
    pid = hr_p.parent.name
    rows: Dict[str, Dict[str, float]] = {}
    for roi_name, label in {"all": None, "core": 1, "edema": 2, "surround": 3}.items():
        m = roi_mask(seg, label)
        if not m.any():
            continue
        fo_hr = first_order(hr[m]); fo_sr = first_order(sr[m])
        contr, homog, corr = [], [], []
        Z = hr.shape[2]
        for z in range(Z):
            m2d = m[:, :, z]
            if not m2d.any(): continue
            f_hr = glcm_slice_feats(hr[:, :, z], m2d)
            f_sr = glcm_slice_feats(sr[:, :, z], m2d)
            contr.append((f_hr["contr"], f_sr["contr"]))
            homog.append((f_hr["homog"], f_sr["homog"]))
            corr.append((f_hr["corr"],  f_sr["corr"]))
        def _avg(pairs, i):
            a = np.array([p[i] for p in pairs if np.all(np.isfinite(p))], float)
            return float(np.mean(a)) if a.size else np.nan
        feats = {}
        for k in ["mean","var","skew","kurt","entropy"]:
            feats[f"{k}_HR"] = fo_hr[k]; feats[f"{k}_SR"] = fo_sr[k]
        feats["contr_HR"] = _avg(contr,0); feats["contr_SR"] = _avg(contr,1)
        feats["homog_HR"] = _avg(homog,0); feats["homog_SR"] = _avg(homog,1)
        feats["corr_HR"]  = _avg(corr,0);  feats["corr_SR"]  = _avg(corr,1)
        rows[roi_name] = feats
    return pid, rows

# UPDATED: run_icc_light — consume error records, continue, and log to JSON
def run_icc_light(paths: Paths, pulses: List[str], resolutions: List[int],
                  models: List[str], workers: int) -> Dict[str, List[Dict[str, Any]]]:
    icc_out: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pulses}
    errors: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    for pulse in pulses:
        for res in resolutions:
            for model in models:
                pairs = collect_paths(paths.hr_root, paths.results_root, pulse, model, res)
                if not pairs:
                    continue
                per_roi_feats: Dict[str, Dict[str, Dict[str, float]]] = {roi: {} for roi in ROI_LABELS}
                with ctx.Pool(processes=max(1, workers)) as pool:
                    tasks = zip(pairs, repeat(pulse))
                    it = pool.imap_unordered(
                        _dispatch_one_subject,
                        tasks,
                        chunksize=max(1, len(pairs)//(4*max(1, workers)) or 1),
                    )
                    for res_item in tqdm(it, total=len(pairs),
                                         desc=f"Radiomics {pulse} {res}mm {model}", unit="subj"):
                        if isinstance(res_item, dict) and res_item.get("__error__"):
                            errors.append(res_item)
                            LOG.warning("Skipped subject due to I/O: %s", res_item["error"])
                            continue
                        pid, roi_feats = res_item
                        for roi in ROI_LABELS:
                            if roi in roi_feats:
                                per_roi_feats[roi][pid] = roi_feats[roi]
                # compute ICC as before ...
                for roi in ROI_LABELS:
                    if not per_roi_feats[roi]:
                        continue
                    keys = sorted(per_roi_feats[roi].keys())
                    feat_names = [k[:-3] for k in per_roi_feats[roi][keys[0]].keys() if k.endswith("_HR")]
                    for feat in feat_names:
                        hr_vals = np.array([per_roi_feats[roi][k][f"{feat}_HR"] for k in keys], float)
                        sr_vals = np.array([per_roi_feats[roi][k][f"{feat}_SR"] for k in keys], float)
                        mask = np.isfinite(hr_vals) & np.isfinite(sr_vals)
                        if mask.sum() < 2: continue
                        icc = _icc2_1_two_raters(hr_vals[mask], sr_vals[mask])
                        icc_out[pulse].append({"pulse": pulse, "resolution": int(res), "model": model,
                                               "roi": roi, "feature": feat, "n": int(mask.sum()),
                                               "ICC2_1": float(icc)})
                gc.collect()
    # persist error ledger next to main NPZ
    try:
        err_path = paths.out_npz.parent / (paths.out_npz.stem + "_radiomics_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        if errors:
            LOG.info("Radiomics errors logged → %s (%d records)", err_path, len(errors))
    except Exception:
        pass
    return icc_out


# ------------------------- 8) Orchestration and CLI ------------------------
def main() -> None:
    _configure_logging()
    # avoid BLAS oversubscription inside forked workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_npz", type=pathlib.Path, required=True)
    ap.add_argument("--hr_root", type=pathlib.Path, required=True)
    ap.add_argument("--results_root", type=pathlib.Path, required=True)
    ap.add_argument("--out_npz", type=pathlib.Path, required=True)
    ap.add_argument("--baseline-model", type=str, default=None,
                    help="Baseline model for comparisons (default: BSPLINE if present, else first model alphabetically)")

    # Radiomics NPZ
    ap.add_argument("--radiomics_out", type=pathlib.Path, required=False,
                    help="If set, save per-subject IBSI radiomics (HR/SR) to a second NPZ.")
    ap.add_argument("--ibsi_bin_width", type=float, default=25.0,
                    help="IBSI fixed bin width. Set ≤0 to skip binWidth and let pyradiomics decide.")
    ap.add_argument("--ibsi_distances", type=int, nargs="+", default=[1, 2, 3],
                    help="Distances for GLCM/GLRLM.")
    ap.add_argument("--ibsi_min_roi_vox", type=int, default=500,
                    help="Skip IBSI if ROI voxels < threshold.")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Process workers for radiomics stages.")
    args = ap.parse_args()
    paths = Paths(args.metrics_npz, args.hr_root, args.results_root, args.out_npz, 
                  baseline_model=args.baseline_model)

    LOG.info("SR statistics analysis started.")

    # 1) Load and sanity report
    metr = load_metrics_npz(paths.metrics_npz)
    try:
        report = sanity_check_metrics(metr, paths.hr_root, paths.results_root)
        rep_path = paths.out_npz.parent / (paths.out_npz.stem + "_sanity_report.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOG.info("Sanity report → %s", rep_path)
    except Exception as e:
        LOG.error("Sanity check failed: %s", e)

    # 2) Tidy frame
    df = to_long_df(metr)
    pulses = list(metr["pulses"]); resolutions = list(metr["resolutions_mm"]); models = list(metr["models"])
    LOG.info("Dataset: n=%d rows | pulses=%s | resolutions=%s | models=%s",
             len(df), pulses, resolutions, models)

    if not paths.out_npz.exists():
        # 3) LMM per pulse
        lmm_emm_all: Dict[str, Any] = {}
        lmm_contr_all: Dict[str, Any] = {}
        for p in tqdm(pulses, desc="LMM (primary)", unit="pulse"):
            emm_table, contrasts = fit_lmm_primary(df, p, baseline_model=paths.baseline_model)
            lmm_emm_all[p] = {"emm_table": pack_table(emm_table, f"EMM[{p}]")}
            lmm_contr_all[p] = {"contrasts": contrasts}

        # 4) Nonparametric confirmation
        friedman_all: Dict[str, Any] = {}
        wilcoxon_all: Dict[str, Any] = {}
        for p in tqdm(pulses, desc="Nonparametric tests", unit="pulse"):
            fr, wi = friedman_wilcoxon(df, p, baseline_model=paths.baseline_model)
            friedman_all[p] = {"by_res_roi": fr}
            wilcoxon_all[p] = {"pairs": wi}

        # 5) Radiomic stability ICC (light FO/GLCM, fast)
        icc_all = run_icc_light(paths, pulses, resolutions, models, workers=args.workers)

        LOG.debug(f"ICC results: {icc_all}")
        
        # 6) Persist main NPZ for plotting
        meta = {
            "pulses": [str(x) for x in pulses],
            "resolutions_mm": [int(x) for x in resolutions],
            "models": [str(x) for x in models],
            "metric_names": [str(x) for x in metr["metric_names"]],
            "roi_labels": [str(x) for x in metr["roi_labels"]],
            "primary_by_pulse": {str(k): str(v) for k, v in PRIMARY_BY_PULSE.items()},
            "baseline_model": str(paths.baseline_model) if paths.baseline_model else None,
        }
        meta_json = json.dumps(meta, ensure_ascii=False)
        np.savez_compressed(
            paths.out_npz,
            meta=meta_json,
            lmm_emm=np.array(lmm_emm_all, dtype=object),
            lmm_contrasts=np.array(lmm_contr_all, dtype=object),
            friedman=np.array(friedman_all, dtype=object),
            wilcoxon=np.array(wilcoxon_all, dtype=object),
            icc=np.array(icc_all, dtype=object),
        )
        LOG.info("Saved primary stats NPZ → %s", paths.out_npz)
    else:
        LOG.info("Primary stats NPZ already exists → %s", paths.out_npz)

    # 7) Optional: IBSI radiomics NPZ (heavier; parallelized; per-subject rows)
    if args.radiomics_out is not None:
        ibsi = run_radiomics_ibsi(paths, pulses, resolutions, models,
                                  ibsi_bin_width=args.ibsi_bin_width,
                                  ibsi_distances=args.ibsi_distances,
                                  ibsi_min_roi_vox=args.ibsi_min_roi_vox,
                                  workers=args.workers)
        meta_r = {
            "pulses": [str(x) for x in pulses],
            "resolutions_mm": [int(x) for x in resolutions],
            "models": [str(x) for x in models],
            "roi_labels": list(ROI_LABELS),
            "ibsi_features": sorted(_IBSI_KEEP),
            "ibsi_bin_width": int(args.ibsi_bin_width),
            "ibsi_distances": list(args.ibsi_distances),
            "ibsi_min_roi_vox": int(args.ibsi_min_roi_vox),
        }
        np.savez_compressed(args.radiomics_out,
                            meta=json.dumps(meta_r, ensure_ascii=False),
                            radiomics=np.array(ibsi, dtype=object))
        LOG.info("Saved IBSI radiomics NPZ → %s", args.radiomics_out)

    LOG.info("Done.")

if __name__ == "__main__":
    main()
