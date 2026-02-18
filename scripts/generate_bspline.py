#!/usr/bin/env python3
"""Generate cubic B-spline interpolated volumes from LR to HR resolution.

For each (patient, spacing, pulse):
  1. Load LR volume from source_data.h5
  2. Apply scipy.ndimage.zoom with order=3 (cubic B-spline) along z-axis only
  3. Save to bspline.h5 with HR affine

All HDF5 access happens in the main process (sequential). Only the zoom
computation is dispatched to parallel workers.

Usage:
    python scripts/generate_bspline.py --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

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


def _zoom_one(
    lr_data: np.ndarray,
    hr_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Apply cubic B-spline zoom. Pure compute, no HDF5 access.

    Args:
        lr_data: Low-resolution volume.
        hr_shape: Target high-resolution shape.

    Returns:
        Zoomed volume as float32.
    """
    zoom_factors = tuple(hr_shape[i] / lr_data.shape[i] for i in range(3))
    sr_data = zoom(lr_data, zoom_factors, order=3)

    # Ensure exact shape match (rounding can cause off-by-one)
    if sr_data.shape != hr_shape:
        sr_data = sr_data[: hr_shape[0], : hr_shape[1], : hr_shape[2]]

    return sr_data.astype(np.float32)


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

    # Build task list, filtering skips and misses in the main process
    tasks = []  # (out_key, lr_data, hr_shape, hr_affine)
    n_skip = 0
    n_miss = 0

    LOG.info("Scanning for tasks to process...")
    for patient_id in patients:
        for spacing in spacings:
            for pulse in pulses:
                out_key = expert_key(spacing, patient_id, pulse)

                if has_key(bspline_h5, out_key):
                    n_skip += 1
                    continue

                lr_k = lr_key(spacing, patient_id, pulse)
                hr_k = hr_key(patient_id, pulse)

                if not has_key(source_h5, lr_k):
                    LOG.warning(
                        "MISS %s-%s (%s): LR key not found", patient_id, pulse, spacing
                    )
                    n_miss += 1
                    continue

                if not has_key(source_h5, hr_k):
                    LOG.warning(
                        "MISS %s-%s (%s): HR key not found", patient_id, pulse, spacing
                    )
                    n_miss += 1
                    continue

                # Read data in main process (sequential, safe)
                lr_data, _ = read_volume(source_h5, lr_k)
                hr_data, hr_affine = read_volume(source_h5, hr_k)
                hr_shape = hr_data.shape[:3]
                del hr_data

                tasks.append((out_key, lr_data, hr_shape, hr_affine))

    total = len(tasks) + n_skip + n_miss
    LOG.info(
        "Total %d: %d to process, %d skipped, %d missing",
        total,
        len(tasks),
        n_skip,
        n_miss,
    )

    if not tasks:
        LOG.info("Nothing to do.")
        return

    # Process in batches: parallel zoom, sequential write
    n_ok = 0
    batch_size = args.batch_size

    for batch_start in tqdm(
        range(0, len(tasks), batch_size),
        desc="Batches",
        total=(len(tasks) + batch_size - 1) // batch_size,
    ):
        batch = tasks[batch_start : batch_start + batch_size]

        # Parallel compute (pure numpy/scipy, no HDF5)
        sr_volumes = Parallel(n_jobs=args.n_jobs)(
            delayed(_zoom_one)(lr_data, hr_shape) for _, lr_data, hr_shape, _ in batch
        )

        # Sequential write to bspline.h5
        for (out_key, _, _, hr_affine), sr_data in zip(batch, sr_volumes):
            write_volume(bspline_h5, out_key, sr_data, hr_affine)
            n_ok += 1

    LOG.info("Done: %d OK, %d skipped, %d missing", n_ok, n_skip, n_miss)


if __name__ == "__main__":
    main()
