from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy import ndimage as ndi
from scipy.signal import windows as sigwin
from tqdm import tqdm

from pacs_sr.config.config import PacsSRConfig
from pacs_sr.data.hdf5_io import (
    blend_key,
    expert_h5_path,
    expert_key,
    hr_key,
    read_volume,
    results_h5_path,
    write_volume,
)
from pacs_sr.utils.patches import (
    region_labels,
    region_adjacency_from_labels,
    tile_boxes,
)
from pacs_sr.utils.parallel import create_backend
from pacs_sr.utils.zscore import zscore, inverse_zscore, ZScoreParams
from pacs_sr.utils.registration import register_and_mask, apply_brain_mask
from pacs_sr.model.metrics import psnr, ssim3d_slicewise, mae, mse
from pacs_sr.model.regularization import laplacian_smooth_weights
from pacs_sr.utils.logger import PacsSRLogger
from pacs_sr.utils.weight_maps import weights_dict_to_npz

Array = np.ndarray


@dataclass(frozen=True)
class RegionStats:
    Q: Array  # (M,M)
    B: Array  # (M,)
    vox: int  # voxel count


def _load_patient_data(
    patient_id: str,
    models: Tuple[str, ...],
    spacing: str,
    pulse: str,
    source_h5: Path,
    experts_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Load HR volume, affine, and all expert volumes from HDF5.

    Args:
        patient_id: Patient identifier string.
        models: Tuple of model names.
        spacing: Spacing string.
        pulse: Pulse sequence.
        source_h5: Path to source_data.h5.
        experts_dir: Directory containing {model}.h5 files.

    Returns:
        Tuple of (hr_data, affine, expert_volumes).
    """
    hr, affine = read_volume(source_h5, hr_key(patient_id, pulse))
    experts = []
    for m in models:
        exp, _ = read_volume(
            expert_h5_path(experts_dir, m),
            expert_key(spacing, patient_id, pulse),
        )
        experts.append(exp)
    return hr, affine, experts


def _grad3d(vol: Array, op: str) -> tuple[Array, Array, Array]:
    """
    3D gradients per axis. Supports 'sobel', 'prewitt', 'scharr'.
    Returns (gx, gy, gz), each float32.
    """
    op = op.lower()
    v = vol.astype(np.float32, copy=False)

    if op == "sobel":
        gx = ndi.sobel(v, axis=2, mode="nearest")
        gy = ndi.sobel(v, axis=1, mode="nearest")
        gz = ndi.sobel(v, axis=0, mode="nearest")
        return gx, gy, gz

    if op == "prewitt":
        gx = ndi.prewitt(v, axis=2, mode="nearest")
        gy = ndi.prewitt(v, axis=1, mode="nearest")
        gz = ndi.prewitt(v, axis=0, mode="nearest")
        return gx, gy, gz

    if op == "scharr":
        # 3D Scharr via separable outer products:
        # derivative d=[1,0,-1] along target axis; smoothing s=[3,10,3] along the others
        d = np.array([1, 0, -1], dtype=np.float32)
        s = np.array([3, 10, 3], dtype=np.float32)

        def kx():
            # x-derivative: d(x) ⊗ s(y) ⊗ s(z)
            return np.einsum("z,y,x->zyx", s, s, d)

        def ky():
            # y-derivative: s(x) ⊗ d(y) ⊗ s(z)
            return np.einsum("z,y,x->zyx", s, d, s)

        def kz():
            # z-derivative: d(z) ⊗ s(y) ⊗ s(x) but axes order is (z,y,x)
            return np.einsum("z,y,x->zyx", d, s, s)

        gx = ndi.convolve(v, kx(), mode="nearest")
        gy = ndi.convolve(v, ky(), mode="nearest")
        gz = ndi.convolve(v, kz(), mode="nearest")
        return gx, gy, gz

    raise ValueError(f"Unknown grad_operator '{op}'")


def _tile_window(sz: tuple[int, int, int], kind: str) -> Array:
    """
    3D separable window. Use periodic Hann (sym=False) for COLA at 50% overlap.
    """
    if kind == "hann":
        wz = sigwin.hann(sz[0], sym=False).astype(np.float32)
        wy = sigwin.hann(sz[1], sym=False).astype(np.float32)
        wx = sigwin.hann(sz[2], sym=False).astype(np.float32)
        return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(
            np.float32
        )
    return np.ones(sz, dtype=np.float32)


def _compute_edge_weights(hr: Array, lam_edge: float, edge_power: float) -> Array:
    if lam_edge <= 0:
        return np.ones_like(hr, dtype=np.float32)
    # Sobel gradients per axis
    gx = ndi.sobel(hr, axis=0, mode="nearest")
    gy = ndi.sobel(hr, axis=1, mode="nearest")
    gz = ndi.sobel(hr, axis=2, mode="nearest")
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    gmax = float(np.max(g)) + 1e-6
    w = 1.0 + lam_edge * (g / gmax) ** edge_power
    return w.astype(np.float32)


def _accumulate_region_stats_for_patient_safe(
    patient_id: str,
    models: Tuple[str, ...],
    spacing: str,
    pulse: str,
    labels: Array,
    cfg: PacsSRConfig,
    source_h5: Path,
    experts_dir: Path,
) -> Optional[Dict[int, RegionStats]]:
    """Wrapper that safely handles errors by returning None instead of crashing."""
    try:
        return _accumulate_region_stats_for_patient(
            patient_id, models, spacing, pulse, labels, cfg, source_h5, experts_dir
        )
    except Exception:
        return None


def _accumulate_region_stats_for_patient(
    patient_id: str,
    models: Tuple[str, ...],
    spacing: str,
    pulse: str,
    labels: Array,
    cfg: PacsSRConfig,
    source_h5: Path,
    experts_dir: Path,
) -> Dict[int, RegionStats]:
    """Build Q_r = sum x x^T and B_r = sum x y within each region for a single patient."""
    hr, hr_affine, expert_volumes = _load_patient_data(
        patient_id, models, spacing, pulse, source_h5, experts_dir
    )

    # Preprocessing: register to atlas and apply brain mask (before normalization)
    if cfg.use_registration and cfg.atlas_dir is not None:
        hr, _ = register_and_mask(hr, hr_affine, pulse, cfg.atlas_dir)
        for i, xi in enumerate(expert_volumes):
            xi, _ = register_and_mask(xi, hr_affine, pulse, cfg.atlas_dir)
            expert_volumes[i] = xi

    # Find maximum common shape across HR and all experts (to preserve all information)
    all_shapes = [hr.shape] + [xi.shape for xi in expert_volumes]
    max_shape = tuple(max(shapes[i] for shapes in all_shapes) for i in range(3))

    # Resize HR to max shape using bicubic interpolation if needed
    if hr.shape != max_shape:
        from scipy.ndimage import zoom

        zoom_factors = tuple(max_shape[i] / hr.shape[i] for i in range(3))
        hr = zoom(hr, zoom_factors, order=3)  # order=3 for bicubic
        # Also resize labels to match
        labels = zoom(labels, zoom_factors, order=0).astype(
            np.int32
        )  # order=0 for nearest-neighbor on labels

    # Foreground mask from HR > 0 (after cropping and registration)
    mask = hr > 0

    # z-score normalization under mask
    if cfg.normalize == "zscore":
        hr, _ = zscore(hr, mask=mask)

    # weights for edge emphasis
    wv = _compute_edge_weights(hr, cfg.lambda_edge, cfg.edge_power) * mask.astype(
        np.float32
    )

    # compute HR gradients after normalization
    hr_gx = hr_gy = hr_gz = None
    if cfg.lambda_grad > 0:
        hr_gx, hr_gy, hr_gz = _grad3d(hr, cfg.grad_operator)

    # Process expert predictions with common shape
    X = []
    GX = []
    GY = []
    GZ = []
    for xi in expert_volumes:
        # Resize to max shape using bicubic interpolation if needed
        if xi.shape != max_shape:
            from scipy.ndimage import zoom

            zoom_factors = tuple(max_shape[i] / xi.shape[i] for i in range(3))
            xi = zoom(xi, zoom_factors, order=3)  # order=3 for bicubic

        # z-score normalization with same mask as HR
        if cfg.normalize == "zscore":
            xi, _ = zscore(xi, mask=mask)

        X.append(xi.astype(np.float32))
        if cfg.lambda_grad > 0:
            gx, gy, gz = _grad3d(xi, cfg.grad_operator)
            GX.append(gx)
            GY.append(gy)
            GZ.append(gz)

    X = np.stack(X, axis=0)  # (M, Z, Y, X)
    if cfg.lambda_grad > 0:
        GX = np.stack(GX, axis=0)  # (M, Z, Y, X)
        GY = np.stack(GY, axis=0)
        GZ = np.stack(GZ, axis=0)
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
        y = hr[selm]  # (n,)
        ww = wv[selm]  # (n,)
        # weighted normal equations
        xw = x * ww  # (M,n)
        Q = xw @ x.T  # (M,M)
        B = xw @ y  # (M,)

        # add gradient terms if enabled
        if cfg.lambda_grad > 0:
            # expert gradients restricted to region
            gx = np.stack([gi[selm] for gi in GX], axis=0)  # (M, n)
            gy = np.stack([gi[selm] for gi in GY], axis=0)
            gz = np.stack([gi[selm] for gi in GZ], axis=0)
            # target gradients
            ygx = hr_gx[selm]
            ygy = hr_gy[selm]
            ygz = hr_gz[selm]
            # weight by ww then accumulate axiswise
            lam = float(cfg.lambda_grad)
            for G, ygrad in ((gx, ygx), (gy, ygy), (gz, ygz)):
                Gw = G * ww  # (M, n)
                Q += lam * (Gw @ G.T)  # add gradient-domain quadratic
                B += lam * (Gw @ ygrad)

        prev = stats.get(rid)
        if prev is None:
            stats[rid] = RegionStats(Q=Q, B=B, vox=nvox)
        else:
            stats[rid] = RegionStats(Q=prev.Q + Q, B=prev.B + B, vox=prev.vox + nvox)
    return stats


def _reduce_region_stats(
    all_stats: List[Dict[int, RegionStats]], M: int
) -> Tuple[List[int], Array, Array, Array]:
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
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones_like(w),
            },
            {"type": "ineq", "fun": lambda w: w},  # w_i >= 0
        ]
        res = minimize(
            fun,
            w0,
            jac=jac,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-12},
        )
        w = res.x
        w[w < 0] = 0.0
        s = w.sum()
        w = (np.ones_like(w) / M) if s <= 0 else (w / s)
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

    def __init__(
        self,
        cfg: PacsSRConfig,
        source_h5: Path,
        experts_dir: Path,
        fold_num: Optional[int] = None,
        logger: Optional[PacsSRLogger] = None,
    ) -> None:
        self.cfg = cfg
        self.source_h5 = source_h5
        self.experts_dir = experts_dir
        self.fold_num = fold_num  # Store fold number for directory structure
        self.weights_: Dict[
            Tuple[str, str], Dict[int, np.ndarray]
        ] = {}  # key=(spacing,pulse) -> {rid: w}
        self.labels_cache_: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
        self.region_ids_cache_: Dict[Tuple[int, int, int, int, int], List[int]] = {}
        self.mask_cache_: Dict[str, np.ndarray] = {}  # cache for brain masks

        # Initialize logger
        if logger is None:
            out_dir = Path(cfg.out_root) / cfg.experiment_name
            log_file = (
                out_dir / f"{cfg.experiment_name}_training.log"
                if cfg.log_to_file
                else None
            )
            self.logger = PacsSRLogger(
                name=f"PaCS-SR.{cfg.experiment_name}", log_file=log_file
            )
            # Set log level
            import logging as stdlib_logging

            level = getattr(stdlib_logging, cfg.log_level.upper(), stdlib_logging.INFO)
            self.logger.logger.setLevel(level)
        else:
            self.logger = logger

        # Log session header and configuration
        self.logger.log_session_header()
        self.logger.log_config(cfg)

    def _labels_for_shape(
        self, shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, List[int]]:
        key = (shape[0], shape[1], shape[2], self.cfg.patch_size, self.cfg.stride)
        if key not in self.labels_cache_:
            labels = region_labels(shape, self.cfg.patch_size, self.cfg.stride)
            rids = sorted(int(r) for r in np.unique(labels))
            self.labels_cache_[key] = labels
            self.region_ids_cache_[key] = rids
        return self.labels_cache_[key], self.region_ids_cache_[key]

    def fit_one(
        self, manifest: Dict, spacing: str, pulse: str
    ) -> Dict[int, np.ndarray]:
        """Fit the regional weights on training patients for one (spacing, pulse)."""
        patient_ids = list(manifest["train"])
        n_patients = len(patient_ids)

        # Determine volume shape for labels
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            from pacs_sr.utils.registration import get_atlas_path
            import ants

            atlas_path = get_atlas_path(self.cfg.atlas_dir, pulse)
            atlas_img = ants.image_read(str(atlas_path))
            shape = atlas_img.numpy().shape
        else:
            # Peek one HR to build labels
            hr_peek, _ = read_volume(self.source_h5, hr_key(patient_ids[0], pulse))
            shape = hr_peek.shape[:3]

        labels, rids = self._labels_for_shape(shape)
        n_regions = len(rids)
        M = len(self.cfg.models)

        # Log training start
        self.logger.log_training_start(spacing, pulse, n_patients, n_regions)

        # Accumulate per-patient stats in parallel
        self.logger.info("Accumulating statistics from training patients...")

        # Memory-aware worker count: limit parallelism to avoid OOM
        #
        # Memory requirements per worker (empirically measured):
        # - Python interpreter + libraries: ~250 MB
        # - Volume data (HR + 4 experts @ ~100MB each): ~500 MB
        # - ANTs registration temp data (if enabled): +1-2 GB
        # - Processing buffers (gradients, statistics): ~300 MB
        # - Safety margin for peaks: +200 MB
        #
        # Total per worker:
        # - Without registration: ~1.25 GB
        # - With registration: ~2.5-3 GB
        #
        # SLURM allocation: 16 GB, but ~1-2 GB used by system/base process
        # Safe usable: ~14 GB

        try:
            import psutil

            # Get available memory at this moment
            mem = psutil.virtual_memory()
            available_memory_gb = mem.available / (1024**3)
            total_memory_gb = mem.total / (1024**3)

            # Estimate memory per worker based on whether registration is enabled
            if self.cfg.use_registration and self.cfg.atlas_dir is not None:
                gb_per_worker = 2.5  # ANTs registration is memory-intensive
                self.logger.info(
                    "ANTs registration enabled - using conservative memory estimate (2.5 GB/worker)"
                )
            else:
                gb_per_worker = (
                    1.5  # More conservative than 1 GB to account for overhead
                )

            # Use only 50% of available memory (very conservative) to leave headroom
            usable_memory_gb = available_memory_gb * 0.5
            memory_based_workers = max(1, int(usable_memory_gb / gb_per_worker))

            self.logger.info(
                f"Memory info: {available_memory_gb:.1f} GB available / {total_memory_gb:.1f} GB total, "
                f"using {usable_memory_gb:.1f} GB for workers ({gb_per_worker} GB/worker)"
            )

        except ImportError:
            # Fallback: very conservative estimate if psutil not available
            self.logger.warning(
                "psutil not available, using conservative worker estimate"
            )
            if self.cfg.use_registration and self.cfg.atlas_dir is not None:
                memory_based_workers = 4  # Very safe for registration
            else:
                memory_based_workers = 8  # Safe default without registration

        # Allow manual override via environment variable for debugging/tuning
        import os

        if "PACS_SR_MAX_WORKERS" in os.environ:
            try:
                manual_max_workers = int(os.environ["PACS_SR_MAX_WORKERS"])
                memory_based_workers = min(memory_based_workers, manual_max_workers)
                self.logger.info(
                    f"Manual worker limit applied: PACS_SR_MAX_WORKERS={manual_max_workers}"
                )
            except ValueError:
                pass

        effective_workers = min(self.cfg.num_workers, memory_based_workers, n_patients)

        if effective_workers < self.cfg.num_workers:
            try:
                mem_info = f"available memory: {available_memory_gb:.1f}GB"
            except NameError:
                mem_info = "using conservative estimate"
            self.logger.warning(
                f"Reducing workers from {self.cfg.num_workers} to {effective_workers} "
                f"to avoid OOM ({mem_info})"
            )

        # Create parallel backend with memory-safe worker count
        backend = create_backend(
            backend=self.cfg.parallel_backend, n_jobs=effective_workers, verbose=0
        )

        # Prepare argument tuples for parallel processing
        tasks = [
            (
                pid,
                self.cfg.models,
                spacing,
                pulse,
                labels,
                self.cfg,
                self.source_h5,
                self.experts_dir,
            )
            for pid in patient_ids
        ]

        # Execute with OOM protection: try parallel first, fallback to serial on OOM
        try:
            # Execute parallel processing with or without tqdm
            if self.cfg.disable_tqdm:
                # SLURM-friendly: parallel processing with periodic logging
                import time
                from joblib import delayed

                self.logger.info(
                    f"Processing {n_patients} patients in parallel "
                    f"(backend={self.cfg.parallel_backend}, workers={effective_workers})..."
                )
                patient_start_time = time.time()

                # Use 'auto' batch_size - joblib will adapt based on task completion time
                # This prevents pre-loading too many tasks and causing memory spikes
                with backend.parallel_context(batch_size="auto") as parallel:
                    stats_list_raw = parallel(
                        delayed(_accumulate_region_stats_for_patient_safe)(*args)
                        for args in tasks
                    )

                elapsed = time.time() - patient_start_time
                avg_time = elapsed / n_patients if n_patients > 0 else 0
                self.logger.info(
                    f"Completed {n_patients} patients in {elapsed:.1f}s "
                    f"(avg: {avg_time:.2f}s/patient, throughput: {n_patients / (elapsed / 60):.1f} patients/min)"
                )
            else:
                # Interactive mode: parallel processing with tqdm progress bar
                import time
                from joblib import delayed

                with backend.parallel_context(batch_size="auto") as parallel:
                    stats_list_raw = parallel(
                        delayed(_accumulate_region_stats_for_patient_safe)(*args)
                        for args in tqdm(
                            tasks,
                            desc=f"Processing patients ({spacing}, {pulse})",
                            unit="patient",
                        )
                    )

        except (MemoryError, Exception) as e:
            # Fallback to serial processing on OOM or other parallel execution errors
            error_type = type(e).__name__
            error_msg = str(e)

            # Provide detailed diagnostics
            try:
                import psutil

                mem = psutil.virtual_memory()
                mem_diag = (
                    f"Memory at failure: {mem.used / (1024**3):.1f}GB used / "
                    f"{mem.total / (1024**3):.1f}GB total ({mem.percent:.1f}% utilization)"
                )
            except:
                mem_diag = "Memory diagnostics unavailable"

            self.logger.warning(
                f"Parallel processing failed ({error_type}). {mem_diag}\n"
                f"Error details: {error_msg[:200]}\n"
                f"Falling back to serial processing to ensure completion..."
            )

            import time

            patient_start_time = time.time()
            stats_list_raw = []

            for idx, args in enumerate(tasks):
                try:
                    result = _accumulate_region_stats_for_patient_safe(*args)
                    stats_list_raw.append(result)

                    # Log progress periodically
                    if self.cfg.disable_tqdm and (
                        (idx + 1) % 5 == 0 or idx == n_patients - 1
                    ):
                        elapsed = time.time() - patient_start_time
                        avg_time = elapsed / (idx + 1)
                        eta = avg_time * (n_patients - idx - 1)
                        self.logger.info(
                            f"  Serial progress: {idx + 1}/{n_patients} ({(idx + 1) / n_patients * 100:.1f}%) | "
                            f"Avg: {avg_time:.1f}s/patient | ETA: {eta / 60:.1f}min"
                        )
                except Exception as patient_err:
                    self.logger.error(f"Failed to process patient {idx}: {patient_err}")
                    stats_list_raw.append(None)

            elapsed = time.time() - patient_start_time
            self.logger.info(f"Serial processing completed in {elapsed:.1f}s")

        # Filter out None results (corrupted patients)
        stats_list = [s for s in stats_list_raw if s is not None]
        n_skipped = len(stats_list_raw) - len(stats_list)
        if n_skipped > 0:
            self.logger.warning(
                f"Skipped {n_skipped} corrupted patient(s) during processing"
            )

        # Cleanup backend resources
        backend.cleanup()

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
            self.logger.info(
                f"\nApplying Laplacian smoothing (tau={self.cfg.laplacian_tau})..."
            )
            adj = region_adjacency_from_labels(labels)
            # Convert adjacency from region IDs to array indices
            rid_to_idx = {rid: i for i, rid in enumerate(region_ids)}
            adj_indexed = {
                rid_to_idx[rid]: {rid_to_idx[n] for n in neighbors if n in rid_to_idx}
                for rid, neighbors in adj.items()
                if rid in rid_to_idx
            }
            W = laplacian_smooth_weights(
                W, adj_indexed, tau=self.cfg.laplacian_tau, n_iter=1
            )

        # store
        wdict = {int(rid): W[i] for i, rid in enumerate(region_ids)}
        self.weights_[(spacing, pulse)] = wdict

        # Log training summary
        self.logger.log_training_summary(n_regions, spacing, pulse)

        return wdict

    def blend_entry(
        self, patient_id: str, spacing: str, pulse: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ZScoreParams]:
        """Apply learned regional weights to one patient.

        Returns:
            Tuple of (blended_native, hr_affine, hr, hr_zscore_params).
        """
        hr, hr_affine, expert_volumes = _load_patient_data(
            patient_id,
            self.cfg.models,
            spacing,
            pulse,
            self.source_h5,
            self.experts_dir,
        )

        # Preprocessing: register HR to atlas if enabled (before normalization)
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            hr, hr_affine = register_and_mask(hr, hr_affine, pulse, self.cfg.atlas_dir)
            for i, xi in enumerate(expert_volumes):
                xi, _ = register_and_mask(xi, hr_affine, pulse, self.cfg.atlas_dir)
                expert_volumes[i] = xi

        # Determine labels shape (atlas shape if registration enabled)
        labels, rids = self._labels_for_shape(hr.shape[:3])
        wdict = self.weights_[(spacing, pulse)]

        # Find maximum common shape across HR and all experts (to preserve all information)
        all_shapes = [hr.shape] + [xi.shape for xi in expert_volumes]
        max_shape = tuple(max(shapes[i] for shapes in all_shapes) for i in range(3))

        # Resize HR and labels to max shape using bicubic interpolation if needed
        if hr.shape != max_shape:
            from scipy.ndimage import zoom

            zoom_factors = tuple(max_shape[i] / hr.shape[i] for i in range(3))
            hr = zoom(hr, zoom_factors, order=3)  # order=3 for bicubic
            # Also resize labels to match
            labels = zoom(labels, zoom_factors, order=0).astype(
                np.int32
            )  # order=0 for nearest-neighbor on labels

        # normalization params for HR
        mask = hr > 0
        if self.cfg.normalize == "zscore":
            hr_n, hr_z = zscore(hr, mask=mask)
        else:
            hr_n, hr_z = hr, ZScoreParams(mean=0.0, std=1.0)

        # Process expert predictions with common shape
        X = []
        for xi in expert_volumes:
            # Resize to max shape using bicubic interpolation if needed
            if xi.shape != max_shape:
                from scipy.ndimage import zoom

                zoom_factors = tuple(max_shape[i] / xi.shape[i] for i in range(3))
                xi = zoom(xi, zoom_factors, order=3)  # order=3 for bicubic

            # Normalize with same mask as HR
            if self.cfg.normalize == "zscore":
                xi, _ = zscore(xi, mask=mask)
            X.append(xi.astype(np.float32))
        X = np.stack(X, axis=0)  # (M, Z, Y, X)

        # Choose mixing window; force flat if no overlap
        win_kind = self.cfg.mixing_window
        if self.cfg.stride >= self.cfg.patch_size:
            win_kind = "flat"

        # Perform blending with overlap-add using deterministic tile grid
        accum = np.zeros_like(hr_n, dtype=np.float32)
        weight = np.zeros_like(hr_n, dtype=np.float32)

        # Iterate exact tile boxes; DO NOT derive from labels for overlap cases
        for rid, (z0, z1, y0, y1, x0, x1) in tile_boxes(
            hr.shape[:3], self.cfg.patch_size, self.cfg.stride
        ):
            w = wdict.get(int(rid))
            if w is None:
                w = np.ones((X.shape[0],), dtype=np.float32) / X.shape[0]

            sub = X[:, z0:z1, y0:y1, x0:x1]  # (M, dz, dy, dx)
            tile_vals = np.tensordot(
                w.astype(np.float32), sub.reshape(sub.shape[0], -1), axes=(0, 0)
            )
            tile_vals = tile_vals.reshape(sub.shape[1:])  # (dz, dy, dx)

            w3 = _tile_window(tile_vals.shape, win_kind)
            accum[z0:z1, y0:y1, x0:x1] += w3 * tile_vals
            weight[z0:z1, y0:y1, x0:x1] += w3

        # finalize
        eps = 1e-8
        blended_n = accum / (weight + eps)

        # Postprocessing: registration already applied during preprocessing
        # No need to register again since all volumes are already in atlas space

        # Step 1: invert normalization to native scale
        blended_native = (
            inverse_zscore(blended_n, hr_z)
            if self.cfg.normalize == "zscore"
            else blended_n
        )

        # Step 2: apply brain mask to set out-of-brain voxels to 0 (fixes tiling artifacts)
        if self.cfg.use_registration and self.cfg.atlas_dir is not None:
            # Use atlas-based brain mask (registered to atlas space)
            blended_native = apply_brain_mask(
                blended_native, pulse, self.cfg.atlas_dir, self.mask_cache_
            )
        else:
            # Without registration, compute brain mask from HR volume in native space
            # This removes blending artifacts outside the brain region
            from pacs_sr.utils.compute_brain_mask import largest_cc_mask

            brain_mask = largest_cc_mask(hr, thr=0.1, connectivity=26)
            blended_native = blended_native * brain_mask.astype(np.float32)

        return blended_native, hr_affine, hr, hr_z

    def _eval_patient(
        self, pid: str, spacing: str, pulse: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Blend one patient and compute metrics.

        Returns:
            Tuple of (pred, affine, hr, metrics_dict).
        """
        pred, affine, hr, _ = self.blend_entry(pid, spacing, pulse)
        mask = hr > 0
        m = {
            "mse": mse(pred, hr, mask),
            "mae": mae(pred, hr, mask),
            "psnr": psnr(pred, hr, data_range=float(hr.max() - hr.min()), mask=mask),
            "ssim": ssim3d_slicewise(pred, hr, mask=mask, axis=self.cfg.ssim_axis),
        }
        return pred, affine, hr, m

    def evaluate_split(
        self, manifest: Dict, spacing: str, pulse: str
    ) -> Dict[str, float]:
        """Evaluate on train and test sets. Write outputs to HDF5 if requested.

        Blends are saved to results_fold_{N}.h5 via write_volume.
        Small metadata (metrics.json, weights.json) stays as files.
        """
        # Results HDF5 for this fold
        res_h5 = results_h5_path(
            self.cfg.out_root, self.cfg.experiment_name, self.fold_num or 0
        )

        # Directory for small metadata files (metrics.json, weights.json)
        base_dir = Path(self.cfg.out_root) / self.cfg.experiment_name / spacing
        if self.fold_num is not None:
            model_data_dir = base_dir / "model_data" / f"fold_{self.fold_num}" / pulse
        else:
            model_data_dir = base_dir / "model_data" / pulse
        model_data_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        wdict = self.weights_[(spacing, pulse)]

        # --- Train set evaluation (optional) ---
        if self.cfg.evaluate_train:
            train_pids = list(manifest["train"])
            n_train = len(train_pids)
            self.logger.log_evaluation_start(spacing, pulse, "train", n_train)

            train_metrics = []
            iterator = (
                train_pids
                if self.cfg.disable_tqdm
                else tqdm(
                    train_pids,
                    desc=f"Evaluating train ({spacing}, {pulse})",
                    unit="patient",
                )
            )
            import time

            eval_start_time = time.time()

            for idx, pid in enumerate(iterator):
                patient_start = time.time()
                try:
                    pred, affine, hr, metrics = self._eval_patient(pid, spacing, pulse)
                    train_metrics.append(metrics)
                    patient_time = time.time() - patient_start

                    if idx % 10 == 0 or idx == n_train - 1:
                        self.logger.log_patient_metrics(
                            pid, metrics, idx, n_train, elapsed_time=patient_time
                        )

                    if self.cfg.save_blends:
                        write_volume(
                            res_h5, blend_key(spacing, pid, pulse), pred, affine
                        )
                except Exception as err:
                    self.logger.warning(f"  Skipping train patient {pid}: {err}")
                    continue

            if train_metrics:
                results["train"] = {
                    k: float(np.mean([m[k] for m in train_metrics]))
                    for k in train_metrics[0]
                }
                self.logger.log_aggregate_metrics(
                    "train", results["train"], spacing, pulse
                )
            else:
                self.logger.warning("No train metrics to aggregate (empty train set)")
        else:
            self.logger.info("Skipping train evaluation (evaluate_train=False)")

        # --- Test set evaluation ---
        test_pids = list(manifest["test"])
        n_test = len(test_pids)
        self.logger.log_evaluation_start(spacing, pulse, "test", n_test)

        test_metrics = []
        volume_shape = None
        iterator = (
            test_pids
            if self.cfg.disable_tqdm
            else tqdm(
                test_pids, desc=f"Evaluating test ({spacing}, {pulse})", unit="patient"
            )
        )
        import time

        eval_start_time = time.time()

        for idx, pid in enumerate(iterator):
            patient_start = time.time()
            try:
                pred, affine, hr, metrics = self._eval_patient(pid, spacing, pulse)
                test_metrics.append(metrics)
                patient_time = time.time() - patient_start

                if volume_shape is None:
                    volume_shape = hr.shape[:3]

                # Log progress
                if idx % 5 == 0 or idx == n_test - 1:
                    if self.cfg.disable_tqdm:
                        elapsed = time.time() - eval_start_time
                        avg_time = elapsed / (idx + 1)
                        eta = avg_time * (n_test - idx - 1)
                        self.logger.info(
                            f"  [{idx + 1:3d}/{n_test:3d}] ({(idx + 1) / n_test * 100:5.1f}%) {pid}: "
                            f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} | "
                            f"Time: {patient_time:.1f}s | Avg: {avg_time:.1f}s | ETA: {eta / 60:.1f}min"
                        )
                    else:
                        self.logger.log_patient_metrics(pid, metrics, idx, n_test)

                # Save blend to HDF5
                if self.cfg.save_blends:
                    write_volume(res_h5, blend_key(spacing, pid, pulse), pred, affine)
                    self.logger.log_blend_saving(res_h5, pid)

                # Save weight maps (still uses NPZ for analysis compatibility)
                if self.cfg.save_weight_volumes and volume_shape is not None:
                    weight_npz_path = model_data_dir / f"{pid}_weights_test.npz"
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
                        include_analysis=True,
                    )
                    self.logger.log_weight_saving(weight_npz_path, format="npz")
            except Exception as err:
                self.logger.warning(f"  Skipping test patient {pid}: {err}")
                continue

        # Aggregate test metrics
        if test_metrics:
            results["test"] = {
                k: float(np.mean([m[k] for m in test_metrics])) for k in test_metrics[0]
            }
            self.logger.log_aggregate_metrics("test", results["test"], spacing, pulse)
        else:
            self.logger.warning("No test metrics to aggregate (empty test set)")
            results["test"] = {}

        # Save weights dictionary as JSON (small, human-readable)
        weights_json_path = model_data_dir / "weights.json"
        with open(weights_json_path, "w") as f:
            json.dump({str(r): w.tolist() for r, w in wdict.items()}, f, indent=2)
        self.logger.info(f"\nSaved weight dictionary (JSON): {weights_json_path}")

        # Save metrics summary
        metrics_path = model_data_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved metrics summary: {metrics_path}")

        return results
