#!/usr/bin/env python3
"""Generate cubic B-spline interpolated volumes from LR to HR resolution.

For each (patient, spacing, pulse):
  1. Load LR volume from source_data.h5
  2. Apply scipy.ndimage.zoom with order=3 (cubic B-spline) along z-axis only
  3. Save to bspline.h5 with HR affine

Computation (zoom) is parallelized across workers; HDF5 writes are sequential
because HDF5 does not support concurrent writers.

Usage:
    python scripts/generate_bspline.py --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import zoom
from tqdm import tqdm

from pacs_sr.data.hdf5_io import (
    expert_h5_path,
    expert_key,
    has_key,
    hr_key,
    list_groups,
    lr_key,
    read_volume,
    write_volume,
)

LOG = logging.getLogger(__name__)


def _compute_one(
    patient_id: str,
    spacing: str,
    pulse: str,
    source_h5: Path,
    bspline_h5: Path,
) -> Tuple[str, str, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Compute B-spline interpolation for one (patient, spacing, pulse).

    Returns the result in memory so the caller can write it sequentially.

    Args:
        patient_id: Patient ID string.
        spacing: Spacing string (e.g., "3mm").
        pulse: Pulse sequence (e.g., "t1c").
        source_h5: Path to source_data.h5 with HR and LR volumes.
        bspline_h5: Path to bspline.h5 (checked for skip).

    Returns:
        (status_msg, out_key, data_tuple) where data_tuple is
        (sr_data, hr_affine) or None if skipped/missing.
    """
    out_key = expert_key(spacing, patient_id, pulse)

    if has_key(bspline_h5, out_key):
        return (f"SKIP {patient_id}-{pulse} ({spacing}): already exists", out_key, None)

    lr_k = lr_key(spacing, patient_id, pulse)
    hr_k = hr_key(patient_id, pulse)

    if not has_key(source_h5, lr_k):
        return (
            f"MISS {patient_id}-{pulse} ({spacing}): LR key not found",
            out_key,
            None,
        )

    if not has_key(source_h5, hr_k):
        return (
            f"MISS {patient_id}-{pulse} ({spacing}): HR key not found",
            out_key,
            None,
        )

    # Load volumes (read-only, safe for concurrent access)
    lr_data, _ = read_volume(source_h5, lr_k)
    hr_data, hr_affine = read_volume(source_h5, hr_k)
    hr_shape = hr_data.shape[:3]
    del hr_data  # free memory, we only need the shape and affine

    # Compute zoom factors
    zoom_factors = tuple(hr_shape[i] / lr_data.shape[i] for i in range(3))

    # Apply cubic B-spline interpolation (the expensive part)
    sr_data = zoom(lr_data, zoom_factors, order=3)

    # Ensure exact shape match (rounding can cause off-by-one)
    if sr_data.shape != hr_shape:
        sr_data = sr_data[: hr_shape[0], : hr_shape[1], : hr_shape[2]]

    msg = f"OK   {patient_id}-{pulse} ({spacing}): {lr_data.shape} -> {sr_data.shape}"
    return (msg, out_key, (sr_data.astype(np.float32), hr_affine))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate BSPLINE expert outputs")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Tasks per batch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[BSPLINE] %(levelname)s | %(message)s"
    )

    from pacs_sr.config.config import load_full_config

    full = load_full_config(args.config)
    data = full.data

    source_h5 = data.source_h5
    bspline_h5 = expert_h5_path(data.experts_dir, "BSPLINE")

    # Ensure experts directory exists
    bspline_h5.parent.mkdir(parents=True, exist_ok=True)

    # Discover patients from HR group in source_data.h5
    patients = list_groups(source_h5, "high_resolution")
    LOG.info("Found %d patients in %s", len(patients), source_h5)

    spacings = list(data.spacings)
    pulses = list(data.pulses)

    # Build task list
    tasks = []
    for patient_id in patients:
        for spacing in spacings:
            for pulse in pulses:
                tasks.append((patient_id, spacing, pulse, source_h5, bspline_h5))

    LOG.info(
        "Processing %d tasks (%d patients x %d spacings x %d pulses)",
        len(tasks),
        len(patients),
        len(spacings),
        len(pulses),
    )

    # Process in batches: parallel compute, sequential write
    # This avoids concurrent HDF5 writes while keeping zoom parallelized.
    n_ok = 0
    n_skip = 0
    n_miss = 0
    batch_size = args.batch_size

    for batch_start in tqdm(
        range(0, len(tasks), batch_size),
        desc="Batches",
        total=(len(tasks) + batch_size - 1) // batch_size,
    ):
        batch = tasks[batch_start : batch_start + batch_size]

        # Parallel compute
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_compute_one)(*task) for task in batch
        )

        # Sequential write
        for msg, out_key, data_tuple in results:
            if data_tuple is not None:
                sr_data, hr_affine = data_tuple
                write_volume(bspline_h5, out_key, sr_data, hr_affine)
                n_ok += 1
            elif msg.startswith("SKIP"):
                n_skip += 1
            else:
                n_miss += 1
                LOG.warning(msg)

    LOG.info("Done: %d OK, %d skipped, %d missing", n_ok, n_skip, n_miss)


if __name__ == "__main__":
    main()
