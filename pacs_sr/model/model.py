from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from pacs_sr.config.config import PacsSRConfig
from pacs_sr.utils.patches import region_labels, region_adjacency_from_labels
from pacs_sr.utils.zscore import zscore, inverse_zscore, ZScoreParams
from pacs_sr.utils.registration import register_and_mask, apply_brain_mask
from pacs_sr.model.metrics import psnr, ssim3d_slicewise, mae, mse
from pacs_sr.model.regularization import laplacian_smooth_weights
from pacs_sr.utils.logger import PacsSRLogger, setup_experiment_logger
from pacs_sr.utils.weight_maps import weights_dict_to_npz

Array = np.ndarray

@dataclass(frozen=True)
class RegionStats:
    Q: Array  # (M,M)
    B: Array  # (M,)
    vox: int  # voxel count

def _load_nii(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))

def _get_data32(img: nib.Nifti1Image) -> Array:
    return img.get_fdata(dtype=np.float32)

def _patient_paths(entry: Dict, models: Tuple[str, ...], spacing: str, pulse: str) -> Tuple[List[Path], Path]:
    sr_paths = [Path(entry[m][spacing][pulse]) for m in models]
    hr_path = Path(entry["HR"][spacing][pulse])
    return sr_paths, hr_path

def _compute_edge_weights(hr: Array, lam_edge: float, edge_power: float) -> Array:
    if lam_edge <= 0:
        return np.ones_like(hr, dtype=np.float32)
    # Sobel gradients per axis
    from scipy.ndimage import sobel
    gx = sobel(hr, axis=0, mode="nearest")
    gy = sobel(hr, axis=1, mode="nearest")
    gz = sobel(hr, axis=2, mode="nearest")
    g = np.sqrt(gx*gx + gy*gy + gz*gz)
    gmax = float(np.max(g)) + 1e-6
    w = 1.0 + lam_edge * (g / gmax) ** edge_power
    return w.astype(np.float32)

def _accumulate_region_stats_for_patient(
    entry: Dict,
    models: Tuple[str, ...],
    spacing: str,
    pulse: str,
    labels: Array,
    cfg: PacsSRConfig,
) -> Dict[int, RegionStats]:
    """
    Build Q_r = sum x x^T and B_r = sum x y within each region for a single patient.
    """
    sr_paths, hr_path = _patient_paths(entry, models, spacing, pulse)
    hr_img = _load_nii(hr_path)
    hr = _get_data32(hr_img)

    # Preprocessing: register to atlas and apply brain mask (before normalization)
    if cfg.use_registration and cfg.atlas_dir is not None:
        hr, _ = register_and_mask(hr, hr_img.affine, pulse, cfg.atlas_dir)

    # Foreground mask from HR > 0 (after registration if enabled)
    mask = hr > 0

    # z-score normalization under mask
    if cfg.normalize == "zscore":
        hr, _ = zscore(hr, mask=mask)

    # weights for edge emphasis
    wv = _compute_edge_weights(hr, cfg.lambda_edge, cfg.edge_power) * mask.astype(np.float32)

    # load expert predictions and apply same preprocessing
    X = []
    for p in sr_paths:
        xi_img = _load_nii(p)
        xi = _get_data32(xi_img)

        # Preprocessing: register expert predictions (before normalization)
        if cfg.use_registration and cfg.atlas_dir is not None:
            xi, _ = register_and_mask(xi, xi_img.affine, pulse, cfg.atlas_dir)

        # z-score normalization with same mask as HR
        if cfg.normalize == "zscore":
            xi, _ = zscore(xi, mask=mask)

        X.append(xi.astype(np.float32))
    X = np.stack(X, axis=0)  # (M, Z, Y, X)
    M = X.shape[0]

    # iterate regions
    stats: Dict[int, RegionStats] = {}
    for rid in np.unique(labels):
        rid = int(rid)
        sel = labels == rid
        selm = sel & mask
        nvox = int(np.count_nonzero(selm))
        if nvox == 0:
            continue
        # gather per-expert vectors
        x = np.stack([xi[selm] for xi in X], axis=0)  # (M, n)
        y = hr[selm]                                   # (n,)
        ww = wv[selm]                                  # (n,)
        # weighted normal equations
        xw = x * ww  # (M,n)
        Q = xw @ x.T  # (M,M)
        B = xw @ y    # (M,)
        prev = stats.get(rid)
        if prev is None:
            stats[rid] = RegionStats(Q=Q, B=B, vox=nvox)
        else:
            stats[rid] = RegionStats(Q=prev.Q + Q, B=prev.B + B, vox=prev.vox + nvox)
    return stats

def _reduce_region_stats(all_stats: List[Dict[int, RegionStats]], M: int) -> Tuple[List[int], Array, Array, Array]:
    """
    Reduce a list of per-patient dicts into a dense (R,M,M) and (R,M).
    Returns:
        region_ids, Qs, Bs, counts
    """
    # collect unique region ids
    keys = set()
    for d in all_stats:
        keys.update(d.keys())
    region_ids = sorted(int(k) for k in keys)
    R = len(region_ids)
    Qs = np.zeros((R, M, M), dtype=np.float64)
    Bs = np.zeros((R, M), dtype=np.float64)
    counts = np.zeros((R,), dtype=np.int64)
    idx = {rid: i for i, rid in enumerate(region_ids)}
    for d in all_stats:
        for rid, st in d.items():
            i = idx[int(rid)]
            Qs[i] += st.Q
            Bs[i] += st.B
            counts[i] += st.vox
    return region_ids, Qs, Bs, counts

def _solve_simplex_qp(Q: Array, B: Array, lambda_ridge: float, simplex: bool) -> Array:
    """
    Solve min_w w^T(Q+λI)w - 2 b^T w with optional simplex constraints.
    """
    M = Q.shape[0]
    Qr = Q + lambda_ridge * np.eye(M, dtype=Q.dtype)
    def fun(w: Array) -> float:
        return float(w @ Qr @ w - 2.0 * B @ w)
    def jac(w: Array) -> Array:
        return (2.0 * (Qr @ w) - 2.0 * B).astype(np.float64)
    w0 = np.ones(M, dtype=np.float64) / M
    if simplex:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": lambda w: w}  # w_i >= 0
        ]
        res = minimize(fun, w0, jac=jac, constraints=cons, method="SLSQP", options={"maxiter": 200, "ftol": 1e-12})
        w = res.x
        w[w < 0] = 0.0
        s = w.sum()
        w = (np.ones_like(w)/M) if s <= 0 else (w / s)
    else:
        # unconstrained quadratic; closed form w = Q^{-1} B
        try:
            w = np.linalg.solve(Qr, B)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(Qr) @ B
    return w.astype(np.float32)

class PatchwiseConvexStacker:
    """
    Patchwise convex stacker for 3D MRI SR.
    Learn one weight vector per 3D tile. Blending is linear and interpretable.
    """
    def __init__(self, cfg: PacsSRConfig, logger: Optional[PacsSRLogger] = None) -> None:
        self.cfg = cfg
        self.weights_: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}  # key=(spacing,pulse) -> {rid: w}
        self.labels_cache_: Dict[Tuple[int,int,int,int,int], np.ndarray] = {}
        self.region_ids_cache_: Dict[Tuple[int,int,int,int,int], List[int]] = {}
        self.mask_cache_: Dict[str, np.ndarray] = {}  # cache for brain masks

        # Initialize logger
        if logger is None:
            out_dir = Path(cfg.out_root) / cfg.experiment_name
            log_file = out_dir / f"{cfg.experiment_name}_training.log" if cfg.log_to_file else None
            self.logger = PacsSRLogger(name=f"PaCS-SR.{cfg.experiment_name}", log_file=log_file)
            # Set log level
            import logging as stdlib_logging
            level = getattr(stdlib_logging, cfg.log_level.upper(), stdlib_logging.INFO)
            self.logger.logger.setLevel(level)
        else:
            self.logger = logger

        # Log session header and configuration
        self.logger.log_session_header()
        self.logger.log_config(cfg)

    def _labels_for_shape(self, shape: Tuple[int,int,int]) -> Tuple[np.ndarray, List[int]]:
        key = (shape[0], shape[1], shape[2], self.cfg.patch_size, self.cfg.stride)
        if key not in self.labels_cache_:
            labels = region_labels(shape, self.cfg.patch_size, self.cfg.stride)
            rids = sorted(int(r) for r in np.unique(labels))
            self.labels_cache_[key] = labels
            self.region_ids_cache_[key] = rids
        return self.labels_cache_[key], self.region_ids_cache_[key]

    def fit_one(self, manifest: Dict, spacing: str, pulse: str) -> Dict[int, np.ndarray]:
        """
        Fit the regional weights on training patients for one (spacing, pulse).
        """
        entries = list(manifest["train"].values())
        n_patients = len(entries)

        # Determine volume shape for labels
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            # If using registration, get shape from atlas
            from pacs_sr.utils.registration import get_atlas_path
            import ants
            atlas_path = get_atlas_path(self.cfg.atlas_dir, pulse)
            atlas_img = ants.image_read(str(atlas_path))
            shape = atlas_img.numpy().shape
        else:
            # Otherwise, peek one HR to build labels
            _, hrp = next(iter((_patient_paths(e, self.cfg.models, spacing, pulse) for e in entries)))
            shape = _load_nii(hrp).shape[:3]

        labels, rids = self._labels_for_shape(shape)
        n_regions = len(rids)
        M = len(self.cfg.models)

        # Log training start
        self.logger.log_training_start(spacing, pulse, n_patients, n_regions)

        # Accumulate per-patient stats in parallel with progress bar
        self.logger.info("Accumulating statistics from training patients...")
        stats_list = Parallel(n_jobs=self.cfg.num_workers, prefer="threads")(
            delayed(_accumulate_region_stats_for_patient)(e, self.cfg.models, spacing, pulse, labels, self.cfg)
            for e in tqdm(entries, desc=f"Processing patients ({spacing}, {pulse})", unit="patient")
        )
        region_ids, Qs, Bs, counts = _reduce_region_stats(stats_list, M)

        # solve per-region
        self.logger.info("\nOptimizing weights per region...")
        W = np.zeros((len(region_ids), M), dtype=np.float32)
        for i, rid in enumerate(region_ids):
            if counts[i] == 0:
                W[i] = np.ones(M, dtype=np.float32) / M
                continue
            w = _solve_simplex_qp(Qs[i], Bs[i], self.cfg.lambda_ridge, self.cfg.simplex)
            W[i] = w.astype(np.float32)

            # Log region optimization based on configured frequency
            if i % self.cfg.log_region_freq == 0 or i == len(region_ids) - 1:
                obj_val = float(w @ Qs[i] @ w - 2.0 * Bs[i] @ w)
                self.logger.log_region_optimization(rid, w.tolist(), obj_val)

        # optional Laplacian smoothing
        if self.cfg.laplacian_tau > 0:
            self.logger.info(f"\nApplying Laplacian smoothing (tau={self.cfg.laplacian_tau})...")
            adj = region_adjacency_from_labels(labels)
            W = laplacian_smooth_weights(W, adj, tau=self.cfg.laplacian_tau, n_iter=1)

        # store
        wdict = {int(rid): W[i] for i, rid in enumerate(region_ids)}
        self.weights_[(spacing, pulse)] = wdict

        # Log training summary
        self.logger.log_training_summary(n_regions, spacing, pulse)

        return wdict

    def blend_entry(self, entry: Dict, spacing: str, pulse: str) -> Tuple[np.ndarray, nib.Nifti1Image, np.ndarray, ZScoreParams]:
        """
        Apply learned regional weights to one patient and return blended image (native scale), ref image, and HR target.
        """
        sr_paths, hr_path = _patient_paths(entry, self.cfg.models, spacing, pulse)
        hr_img = _load_nii(hr_path)
        hr = _get_data32(hr_img)

        # Preprocessing: register HR to atlas if enabled (before normalization)
        hr_affine = hr_img.affine
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            hr, hr_affine = register_and_mask(hr, hr_img.affine, pulse, self.cfg.atlas_dir)

        # Determine labels shape (atlas shape if registration enabled)
        labels, rids = self._labels_for_shape(hr.shape[:3])
        wdict = self.weights_[(spacing, pulse)]

        # normalization params for HR
        mask = hr > 0
        if self.cfg.normalize == "zscore":
            hr_n, hr_z = zscore(hr, mask=mask)
        else:
            hr_n, hr_z = hr, ZScoreParams(mean=0.0, std=1.0)

        # load experts and apply same preprocessing
        X = []
        for p in sr_paths:
            xi_img = _load_nii(p)
            xi = _get_data32(xi_img)

            # Preprocessing: register expert predictions
            if self.cfg.use_registration and self.cfg.atlas_dir is not None:
                xi, _ = register_and_mask(xi, xi_img.affine, pulse, self.cfg.atlas_dir)

            # Normalize with same mask as HR
            if self.cfg.normalize == "zscore":
                xi, _ = zscore(xi, mask=mask)
            X.append(xi.astype(np.float32))
        X = np.stack(X, axis=0)  # (M, Z, Y, X)

        # Perform blending (cross product of weights and expert opinions)
        blended_n = np.zeros_like(hr_n, dtype=np.float32)
        for rid in rids:
            sel = labels == rid
            w = wdict.get(int(rid))
            if w is None:
                w = np.ones((X.shape[0],), dtype=np.float32) / X.shape[0]
            # linear blend inside tile
            tile = np.tensordot(w.astype(np.float32), X[:, sel], axes=(0,0))
            blended_n[sel] = tile.astype(np.float32)

        # Postprocessing: registration already applied during preprocessing
        # No need to register again since all volumes are already in atlas space

        # Step 1: invert normalization to native scale
        blended_native = inverse_zscore(blended_n, hr_z) if self.cfg.normalize == "zscore" else blended_n

        # Step 2: apply brain mask to set out-of-brain voxels to 0 (fixes tiling artifacts)
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            blended_native = apply_brain_mask(blended_native, pulse, self.cfg.atlas_dir, self.mask_cache_)

        # Create reference image with potentially updated affine
        ref_img = nib.Nifti1Image(blended_native, hr_affine, hr_img.header)

        return blended_native, ref_img, hr, hr_z

    def evaluate_split(self, manifest: Dict, spacing: str, pulse: str) -> Dict[str, float]:
        """
        Evaluate on train and test sets. Write outputs if requested.
        """
        # Base directory for this spacing
        base_dir = Path(self.cfg.out_root) / f"{self.cfg.experiment_name}" / spacing
        # Directory for metrics and weights (per pulse)
        out_dir = base_dir / pulse
        out_dir.mkdir(parents=True, exist_ok=True)
        # Directory for output volumes (shared across pulses)
        output_volumes_dir = base_dir / "output_volumes"
        output_volumes_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # Get weight dictionary and volume shape for NPZ saving
        wdict = self.weights_[(spacing, pulse)]

        # Train set evaluation (optional, can be disabled for speed)
        if self.cfg.evaluate_train:
            train_entries = list(manifest["train"].items())
            n_train = len(train_entries)
            self.logger.log_evaluation_start(spacing, pulse, "train", n_train)

            train_metrics = []
            for idx, (pid, entry) in enumerate(tqdm(train_entries, desc=f"Evaluating train ({spacing}, {pulse})", unit="patient", disable=False)):
                pred, ref_img, hr, _ = self.blend_entry(entry, spacing, pulse)
                mask = hr > 0
                metrics = {
                    "mse": mse(pred, hr, mask),
                    "mae": mae(pred, hr, mask),
                    "psnr": psnr(pred, hr, data_range=float(hr.max() - hr.min()), mask=mask),
                    "ssim": ssim3d_slicewise(pred, hr, mask=mask, axis=self.cfg.ssim_axis),
                }
                train_metrics.append(metrics)

                # Log per-patient metrics (less verbose)
                if idx % 10 == 0 or idx == n_train - 1:
                    self.logger.log_patient_metrics(pid, metrics, idx, n_train)

                # Save blended prediction
                if self.cfg.save_blends:
                    blend_path = out_dir / f"{pid}_blend_train.nii.gz"
                    nib.save(nib.Nifti1Image(pred.astype(np.float32), ref_img.affine, ref_img.header), blend_path)

            # Aggregate train metrics
            results["train"] = {k: float(np.mean([m[k] for m in train_metrics])) for k in train_metrics[0].keys()}
            self.logger.log_aggregate_metrics("train", results["train"], spacing, pulse)
        else:
            self.logger.info(f"Skipping train evaluation (evaluate_train=False)")

        # Test set evaluation
        test_entries = list(manifest["test"].items())
        n_test = len(test_entries)
        self.logger.log_evaluation_start(spacing, pulse, "test", n_test)

        test_metrics = []
        volume_shape = None  # Will be set from first test patient

        for idx, (pid, entry) in enumerate(tqdm(test_entries, desc=f"Evaluating test ({spacing}, {pulse})", unit="patient", disable=False)):
            pred, ref_img, hr, _ = self.blend_entry(entry, spacing, pulse)
            mask = hr > 0

            # Capture volume shape for weight map saving
            if volume_shape is None:
                volume_shape = hr.shape[:3]

            metrics = {
                "mse": mse(pred, hr, mask),
                "mae": mae(pred, hr, mask),
                "psnr": psnr(pred, hr, data_range=float(hr.max() - hr.min()), mask=mask),
                "ssim": ssim3d_slicewise(pred, hr, mask=mask, axis=self.cfg.ssim_axis),
            }
            test_metrics.append(metrics)

            # Log per-patient metrics (every 5 patients or last)
            if idx % 5 == 0 or idx == n_test - 1:
                self.logger.log_patient_metrics(pid, metrics, idx, n_test)

            # Save blended prediction to output_volumes directory
            if self.cfg.save_blends:
                blend_path = output_volumes_dir / f"{pid}-{pulse}.nii.gz"
                nib.save(nib.Nifti1Image(pred.astype(np.float32), ref_img.affine, ref_img.header), blend_path)
                self.logger.log_blend_saving(blend_path, pid)

            # Save weight maps in NPZ format for test patients
            if self.cfg.save_weight_volumes and volume_shape is not None:
                weight_npz_path = out_dir / f"{pid}_weights_test.npz"
                weights_dict_to_npz(
                    weights_dict=wdict,
                    volume_shape=volume_shape,
                    patch_size=self.cfg.patch_size,
                    stride=self.cfg.stride,
                    model_names=list(self.cfg.models),
                    save_path=weight_npz_path,
                    patient_id=pid,
                    split="test",
                    spacing=spacing,
                    pulse=pulse,
                    include_analysis=True
                )
                self.logger.log_weight_saving(weight_npz_path, format="npz")

        # Aggregate test metrics
        results["test"] = {k: float(np.mean([m[k] for m in test_metrics])) for k in test_metrics[0].keys()}
        self.logger.log_aggregate_metrics("test", results["test"], spacing, pulse)

        # Save weights dictionary as JSON (for compatibility)
        weights_json_path = out_dir / "weights.json"
        with open(weights_json_path, "w") as f:
            json.dump({str(r): w.tolist() for r, w in wdict.items()}, f, indent=2)
        self.logger.info(f"\nSaved weight dictionary (JSON): {weights_json_path}")

        # Save metrics summary
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved metrics summary: {metrics_path}")

        return results
