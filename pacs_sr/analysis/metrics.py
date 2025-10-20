#!/usr/bin/env python3
"""
metrics.py
==========

Quantify SR reconstruction quality using four metrics:

* **PSNR** – peak-signal-to-noise ratio
* **SSIM** – structural similarity (Gaussian, masked)
* **Bhattacharyya coefficient** – histogram overlap
* **LPIPS** – learned perceptual image patch similarity (AlexNet variant)

Folder layout (example)
-----------------------
  HR_ROOT/
      BraTS-MEN-00018-000/
          BraTS-MEN-00018-000-t1c.nii.gz
          BraTS-MEN-00018-000-t2w.nii.gz
          BraTS-MEN-00018-000-seg.nii.gz
      ...
  RESULTS_ROOT/
      SMORE/
          3mm/output_volumes/*.nii.gz
          5mm/output_volumes/*.nii.gz
          7mm/output_volumes/*.nii.gz
      BSpline/
          3mm/output_volumes/*.nii.gz
          ...

Output
------
`metrics.npz` containing

    metrics      (P, 3, 3, M, 4, 4, 2) float64   # NaN for missing cases
    patient_ids  list[str]
    pulses       ["t1c", "t2w", "t2f"]
    resolutions  [3, 5, 7]                       # mm
    models       list[str]
    metric_names ["PSNR", "SSIM", "BC", "LPIPS"]
    roi_labels   ["all", "core", "edema", "surround"]
    stat_names   ["mean", "std"]
"""

from __future__ import annotations

# ----------------------------------------------------------------- imports
import argparse
import multiprocessing.dummy as mp_threads
import multiprocessing as mp
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, cast

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from pacs_sr.utils.imio import load_lps
import nibabel as nib
# progress bar -------------------------------------------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # fallback: no-op tqdm
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(total or 0)
# -------- deep-perceptual metric ------------------------------------------
try:
    import torch
    import lpips                            # pip install lpips
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LPIPS_MODEL = lpips.LPIPS(net="alex").to(_DEVICE).eval()
    _LPIPS_MODEL.requires_grad_(False)
    _USE_THREADS = bool(_DEVICE.type == "cuda")
except ModuleNotFoundError:                 # library missing → metric = NaN
    _LPIPS_MODEL = None
    _DEVICE = None  # type: ignore[assignment]
    _USE_THREADS = False


# ------------------------------------------------------------- constants ---
ROI_LABELS   = ("all", "core", "edema", "surround")         # 0,1,2,3
METRIC_NAMES = ("PSNR", "SSIM")
STAT_NAMES   = ("mean", "std")                              # NEW axis length = 2
PULSES       = ("t1c", "t1n", "t2w", "t2f")
RESOLUTIONS  = (3, 5, 7)                                    # mm
HR_RE        = re.compile(r"^(?P<pid>[^/]+)-(?P<pulse>t1c|t1n|t2w|t2f)\.nii\.gz$")

# ---------------------------------------------------------- dataclasses ---
@dataclass(frozen=True)
class VolumePaths:
    hr: pathlib.Path
    seg: pathlib.Path
    sr: pathlib.Path


# ----------------------------------------------------------- utils --------
def read_image(path: pathlib.Path) -> sitk.Image:
    """SimpleITK reader with float64 output."""
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, sitk.sitkFloat64)


def sitk_to_np(img: sitk.Image) -> np.ndarray:
    """Preserve ordering (z, y, x) for NumPy."""
    arr = sitk.GetArrayFromImage(img)        # returns z,y,x
    return np.asanyarray(arr, dtype=np.float64)

def unify_shapes(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Pad every 3-D array with zeros so that all have the same shape.
    Any pair of arrays may differ by at most 2 voxels along each axis,
    mirroring `match_slices`'s safety rule.
    """
    shapes = np.array([a.shape for a in arrays])
    if np.any(np.ptp(shapes, axis=0) > 3):
        raise ValueError("Volumes differ by >3 voxels; cannot auto-align.")

    target = shapes.max(axis=0)
    padded: list[np.ndarray] = []
    for a in arrays:
        pad = [(0, t - s) for s, t in zip(a.shape, target)]
        padded.append(np.pad(a, pad, constant_values=0))
    return tuple(padded)

def match_slices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If *a* and *b* differ by ≤2 slices along *any* axis, pad the smaller
    one with zeros so both arrays share the same shape.
    """
    if a.shape == b.shape:
        return a, b
    diffs = np.subtract(a.shape, b.shape)
    if np.any(np.abs(diffs) > 2):
        raise ValueError(f"Images differ by >2 voxels {a.shape} vs {b.shape}")
    # pad smaller
    pad_a = tuple((0, max(0, -d)) for d in diffs)
    pad_b = tuple((0, max(0,  d)) for d in diffs)
    return np.pad(a, pad_a, constant_values=0), np.pad(b, pad_b, constant_values=0)


def exclude_z_slices(arr: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """
    Return *arr* with the Z-slices listed in *idx* removed.

    Any index < 0 or ≥ arr.shape[0] is silently ignored to avoid IndexError.
    """
    if not idx:                                   # empty / None → no-op
        return arr

    z = arr.shape[0]
    # Direct conversion without intermediate list - more efficient
    idx_arr = np.asarray(idx, dtype=np.intp)  # Use native int type for indexing
    in_bounds = (idx_arr >= 0) & (idx_arr < z)
    
    if not np.all(in_bounds):               # tell the user once
        bad = idx_arr[~in_bounds]
        LOGGER.debug("Ignoring %d slice indices outside [0,%d]: %s",
                     bad.size, z - 1, bad.tolist())
        idx_arr = idx_arr[in_bounds]

    # More efficient: use np.delete for known indices
    if idx_arr.size == 0:
        return arr
    return np.delete(arr, idx_arr, axis=0)


# --------------------------------------- metric helpers (robust to NaN/Inf)
def _finite(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return only finite-valued voxels of both arrays (element-wise AND)."""
    # Check if we even need to filter (common case: all finite)
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return a, b
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def psnr(hr: np.ndarray, sr: np.ndarray, data_range: float | None = None) -> float:
    hr, sr = _finite(hr, sr)
    if data_range is None:
        data_range = np.ptp(hr)
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:                                            # SAFE-SSIM
        return np.inf
    if data_range == 0:
        return np.nan
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


# SKIMAGE-SSIM ────────────────────────────────────────────────────────────
def safe_ssim_slice(hr2d: np.ndarray,
                    sr2d: np.ndarray,
                    mask2d: np.ndarray) -> float:
    """
    SSIM on a *single axial slice* using scikit-image.

    * Chooses the largest odd ``win_size`` ≤ min(height, width).
    * When the side is < 3 pixels the function returns **NaN** instead of
      raising ``ValueError``.
    """
    if not mask2d.any():                         # nothing to compare
        return np.nan

    h, w = hr2d.shape
    m = min(h, w)

    if m < 3:                                   # too small for any legal window
        LOGGER.debug("ROI too small for SSIM (%dx%d) → NaN", h, w)
        return np.nan

    ws = 7                                      # choose an odd win_size
    if m < 7:
        ws = m if m % 2 == 1 else m - 1

    try:
        return float(
            ssim(hr2d,
                 sr2d,
                 data_range=float(np.ptp(hr2d)),
                 gaussian_weights=True,
                 win_size=ws,
                 mask=mask2d)                   # ← only voxels inside ROI
        )
    except ValueError as err:
        LOGGER.debug("SSIM failed (%s) → NaN", err)
        return np.nan


def bhattacharyya(hr: np.ndarray, sr: np.ndarray, bins: int = 256,
                  return_distance: bool = True) -> float:
    """
    Bhattacharyya coefficient *or* distance between two 1-D distributions.

    Parameters
    ----------
    hr, sr : ndarray
        Samples (any shape); NaN / Inf are ignored.
    bins : int
        Number of histogram bins.
    return_distance : bool, default False
        If True, return  `-ln(BC)`  (Bhattacharyya distance).
        Otherwise return the coefficient.

    Returns
    -------
    float
        BC in [0, 1]  or  D_B ≥ 0.
    """
    hr, sr = _finite(hr, sr)
    if hr.size == 0 or sr.size == 0:
        return np.nan

    # Use common range for both histograms - more accurate comparison
    vmin = min(hr.min(), sr.min())
    vmax = max(hr.max(), sr.max())
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    
    # Compute both histograms with same bins
    h_cnt, _ = np.histogram(hr, bins=bin_edges, density=False)
    s_cnt, _ = np.histogram(sr, bins=bin_edges, density=False)

    # convert to probabilities (sum = 1) - avoid division if sum is 0
    h_sum = h_cnt.sum()
    s_sum = s_cnt.sum()
    if h_sum == 0 or s_sum == 0:
        return np.nan
    
    p = h_cnt / h_sum
    q = s_cnt / s_sum

    # Compute BC directly without intermediate array
    bc = np.sqrt(p * q).sum()                # coefficient in [0,1]

    if return_distance:
        # guard against log(0)
        return -np.log(max(bc, 1e-12))
    else:
        return bc



# ----------------- LPIPS (slice-wise, ROI-aware via masking to zeros) -----
def lpips_slice(hr2d: np.ndarray,
                sr2d: np.ndarray,
                mask2d: np.ndarray) -> float:
    """
    Compute LPIPS on an axial slice.

    * Voxels **outside** the ROI are zeroed in both images.
    * Intensities are linearly mapped to the range [-1, 1] using the
      *joint* min-max of the two masked images.
    """
    if _LPIPS_MODEL is None:                  # library unavailable
        return np.nan
    if not mask2d.any():
        return np.nan

    # apply mask – everything else set to zero to mimic background
    hr_masked = hr2d * mask2d
    sr_masked = sr2d * mask2d

    vmin = min(np.nanmin(hr_masked), np.nanmin(sr_masked))
    vmax = max(np.nanmax(hr_masked), np.nanmax(sr_masked))
    if vmax - vmin == 0.0:                    # uniform slice
        return np.nan

    # scale to [-1, 1]
    def _scale(img: np.ndarray) -> np.ndarray:
        return (img - vmin) / (vmax - vmin) * 2.0 - 1.0

    hr_scaled = _scale(hr_masked)
    sr_scaled = _scale(sr_masked)

    # shape: 1x3xHxW (RGB replicated)
    hr_t = torch.from_numpy(hr_scaled).float().unsqueeze(0).repeat(3, 1, 1)
    sr_t = torch.from_numpy(sr_scaled).float().unsqueeze(0).repeat(3, 1, 1)
    hr_t, sr_t = hr_t.unsqueeze(0).to(_DEVICE), sr_t.unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        d = _LPIPS_MODEL(hr_t, sr_t).item()   # higher = more different
    return d


# ------------------------------------------------------ per-ROI metrics ---
def roi_mask(seg: np.ndarray, label: int | None) -> np.ndarray:
    """Return boolean mask selecting ROI `label`; `None` → whole image."""
    if label is None:
        return np.ones_like(seg, dtype=bool)
    return seg == label


def compute_metrics(hr: np.ndarray, sr: np.ndarray,
                    seg: np.ndarray) -> np.ndarray:
    """
    **Slice-wise metrics** – returns array (4 metrics, 4 ROIs, 2 stats).

    stats[0] = mean across retained slices  
    stats[1] = std  across retained slices
    """
    out = np.full((len(METRIC_NAMES),
                   len(ROI_LABELS),
                   len(STAT_NAMES)),
                  np.nan, dtype=np.float64)

    z_slices = hr.shape[0]
    for ridx, label in enumerate((None, 1, 2, 3)):
        psnr_vals, ssim_vals, bc_vals, lpips_vals = [], [], [], []

        for z in range(z_slices):
            roi = roi_mask(seg[z], label)
            if not roi.any():
                continue

            #  PSNR & Bhattacharyya ───────────────────────────── 1-D voxels
            h_vec, s_vec = hr[z][roi], sr[z][roi]
            psnr_vals.append(psnr(h_vec, s_vec))
            bc_vals.append(bhattacharyya(h_vec, s_vec))

            #  SSIM ──────────────────────────────────────────── 2-D masked
            ssim_vals.append(safe_ssim_slice(hr[z], sr[z], roi))

            #  LPIPS ─────────────────────────────────────────── deep metric
            lpips_vals.append(lpips_slice(hr[z], sr[z], roi))

        # write mean/std if we have at least one valid slice
        for vals, midx in zip(
            (psnr_vals, ssim_vals, bc_vals, lpips_vals),
            range(len(METRIC_NAMES))
        ):
            if vals:                                     # non-empty list
                out[midx, ridx, 0] = np.nanmean(vals)
                out[midx, ridx, 1] = np.nanstd(vals)

    return out                                           # (4, 4, 2)


# ----------------------------------------------------- filesystem walker --
def collect_paths(hr_root: pathlib.Path,
                  results_root: pathlib.Path,
                  pulse: str,
                  model: str,
                  resolution_mm: int) -> List[VolumePaths]:
    """
    Collect all VolumePaths for every patient that has both HR and SR volumes.
    Returns a list instead of generator for better multiprocessing performance.
    """
    res_dir = results_root / model / f"{resolution_mm}mm" / "output_volumes"
    LOGGER.info("Collecting paths for %s | %d mm | %s", model, resolution_mm, pulse)
    LOGGER.info("res_dir = %s", res_dir)
    
    paths = []
    for patient_dir in sorted(hr_root.iterdir()):
        pid = patient_dir.name
        hr_path = patient_dir / f"{pid}-{pulse}.nii.gz"
        seg_path = patient_dir / f"{pid}-seg.nii.gz"
        sr_path = res_dir / f"{pid}-{pulse}.nii.gz"
        if hr_path.exists() and seg_path.exists() and sr_path.exists():
            paths.append(VolumePaths(hr=hr_path, seg=seg_path, sr=sr_path))
    
    return paths


# ------------------------------------------------ worker (per patient) ----
def process_patient(vpaths: VolumePaths,
                    exclude: Sequence[int]) -> np.ndarray:
    """
    Load HR / SR / SEG volumes for a single patient, align them,
    compute all metrics, and return an array of shape
        (len(METRIC_NAMES), len(ROI_LABELS), len(STAT_NAMES))

    Any geometry mismatch > 2 voxels at any stage → patient is skipped
    (logged) and NaNs are returned, so the main loop continues.
    """
    try:
        
        # ---- reference (HR) ------------------------------------------------
        hr_img = nib.load(str(vpaths.hr))          # keep for geometry
        hr_arr = load_lps(vpaths.hr, dtype=np.float32)  # LPS, no resampling
        
        
        # ---- moving volumes, resampled onto HR grid ------------------------
        sr_arr = load_lps(vpaths.sr, like=cast(nib.Nifti1Image, hr_img), order=1)   # linear
        seg_arr = load_lps(vpaths.seg, like=cast(nib.Nifti1Image, hr_img), order=0) # nearest
        
        # ---------- geometry alignment ------------------------------------
        # (1) HR ↔ SR : allow ≤ 2-voxel padding on each axis
        hr_arr, sr_arr = match_slices(hr_arr, sr_arr)

        # (2) bring the SEG volume to the *same* shape  (≤ 2 voxels tolerance)
        try:
            hr_arr, sr_arr, seg_arr = unify_shapes(hr_arr, sr_arr, seg_arr)
        except ValueError as geo_err:
            raise ValueError(f"SEG/HR size mismatch – {geo_err}") from None

        # ---------- axial slice exclusion ---------------------------------
        # Only exclude if necessary - avoid unnecessary array operations
        if exclude:
            hr_arr  = exclude_z_slices(hr_arr,  exclude)
            sr_arr  = exclude_z_slices(sr_arr,  exclude)
            seg_arr = exclude_z_slices(seg_arr, exclude)

        # ---------- metrics -----------------------------------------------
        return compute_metrics(hr_arr, sr_arr, seg_arr)

    except Exception as err:
        # Skip this patient but never kill the pool
        LOGGER.warning("Skipping patient %s – %s",
                       vpaths.hr.parent.name, err)
        return np.full((len(METRIC_NAMES),
                        len(ROI_LABELS),
                        len(STAT_NAMES)),
                       np.nan, dtype=np.float64)


# Small adapter to allow ordered imap with a single-argument function
def _process_patient_args(arg: Tuple[VolumePaths, Sequence[int]]) -> np.ndarray:
    vp, exclude = arg
    return process_patient(vp, exclude)




# ------------------------------------------------------------- main -------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/Bhattacharyya/LPIPS for SR volumes.")
    ap.add_argument("--hr_root", type=pathlib.Path,
                    default=pathlib.Path(
                        "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution"))
    ap.add_argument("--results_root", type=pathlib.Path,
                    default=pathlib.Path(
                        "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/models"))

    # -------- pulse --------------------------------------------------------
    ap.add_argument("--pulse", choices=("t1c", "t2w", "t2f", "t1n", "all"),
                    default="all",
                    help="Evaluate given pulse or all (default)")

    # -------- slice window -------------------------------------------------
    ap.add_argument("--slice-window", nargs=2, type=int, metavar=("MIN", "MAX"),
                    default=(15, 140),
                    help="Only consider slices MIN..MAX inclusive "
                         "(default 15-140); outside range is ignored.")

    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(
                        "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/metrics/metrics.npz"))
    ap.add_argument("--workers", type=int, default=mp.cpu_count() - 1)
    args = ap.parse_args()

    # discover models dynamically
    model_dirs = [d.name for d in args.results_root.iterdir() if d.is_dir()]
    model_dirs.sort()

    # convert tuple → range of indices to drop
    lo, hi = args.slice_window
    exclude = list(range(0, lo))               # [0, lo-1]
    exclude += list(range(hi + 1, 10 ** 6))    # (hi, ∞) – upper bound trimmed later

    # ---------------------------------------------------------------- engine
    patients = sorted([d.name for d in args.hr_root.iterdir()])
    # SLICE-WISE – added STAT dimension
    metrics_arr = np.full((len(patients),           # P
                           len(PULSES),             # 3
                           len(RESOLUTIONS),        # 3
                           len(model_dirs),
                           len(METRIC_NAMES),       # 4
                           len(ROI_LABELS),         # 4
                           len(STAT_NAMES)),        # 2
                          np.nan, dtype=np.float64)

    patient_idx = {pid: i for i, pid in enumerate(patients)}
    pulse_list = PULSES if args.pulse == "all" else [args.pulse]

    for m_idx, model in enumerate(model_dirs):
        for pulse in pulse_list:
            for r_idx, res in enumerate(RESOLUTIONS):

                items = list(collect_paths(args.hr_root,
                                          args.results_root,
                                          pulse,
                                          model,
                                          res))
                if not items:
                    continue
                LOGGER.info("Model %s | %d patients | 1x1x%d mm", model, len(items), res)

                PoolClass = mp_threads.Pool if _USE_THREADS else mp.Pool

                # Cap workers to number of items to avoid idle processes
                workers = max(1, min(args.workers, len(items)))
                # Reasonable chunk size to reduce scheduling overhead
                chunksize = max(1, len(items) // (workers * 4) or 1)

                pulse_idx = PULSES.index(pulse)
                with PoolClass(workers) as pool:
                    # Use imap instead of imap for ordered results matching items list
                    iterator = pool.imap(
                        _process_patient_args,
                        [(vp, exclude) for vp in items],
                        chunksize=chunksize,
                    )

                    # Progress bar per (model, pulse, resolution)
                    for idx, metr in enumerate(
                        tqdm(iterator, total=len(items), desc=f"{model} | {pulse} | {res}mm", 
                             unit="patient", smoothing=0.1)
                    ):
                        vp = items[idx]
                        p = patient_idx[vp.hr.parent.name]
                        metrics_arr[p, pulse_idx, r_idx, m_idx, :, :, :] = metr
                        # Reduce log volume: info every 10 patients
                        if (idx + 1) % 10 == 0 or (idx + 1) == len(items):
                            LOGGER.info(
                                "[%s %s %dmm] %d/%d processed",
                                model,
                                pulse,
                                res,
                                idx + 1,
                                len(items),
                            )

    # Save with compression to reduce file size
    np.savez_compressed(args.out,
             metrics=metrics_arr,
             patient_ids=patients,
             pulses=PULSES,
             resolutions_mm=RESOLUTIONS,
             models=model_dirs,
             metric_names=METRIC_NAMES,
             roi_labels=ROI_LABELS,
             stat_names=STAT_NAMES)

    LOGGER.info("Saved metrics to %s", args.out)

"""
metrics_arr.shape
# (P, 3, 3, M, 4, 4, 2)
#  ↑  ↑  ↑  ↑  ↑  ↑  └── mean / std   (STAT_NAMES)
#  |  |  |  |  |  └──── ROI_LABELS
#  |  |  |  |  └────── METRIC_NAMES
#  |  |  |  └───────── model index
#  |  |  └──────────── resolution {3,5,7} mm
#  |  └─────────────── pulse {t1c,t1n,t2w,t2f}
#  └────────────────── patient
"""

if __name__ == "__main__":
    main()
