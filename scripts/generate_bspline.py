#!/usr/bin/env python3
"""Generate cubic B-spline interpolated volumes from LR to HR resolution.

For each (patient, spacing, pulse):
  1. Load LR volume from source_data.h5
  2. Apply scipy.ndimage.zoom with order=3 (cubic B-spline) along z-axis only
  3. Save to bspline.h5 with HR affine

Usage:
    python scripts/generate_bspline.py --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import zoom

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


def _process_one(
    patient_id: str,
    spacing: str,
    pulse: str,
    source_h5: Path,
    bspline_h5: Path,
) -> str:
    """Interpolate one (patient, spacing, pulse) volume.

    Args:
        patient_id: Patient ID string.
        spacing: Spacing string (e.g., "3mm").
        pulse: Pulse sequence (e.g., "t1c").
        source_h5: Path to source_data.h5 with HR and LR volumes.
        bspline_h5: Path to bspline.h5 for writing output.

    Returns:
        Status message string.
    """
    out_key = expert_key(spacing, patient_id, pulse)

    if has_key(bspline_h5, out_key):
        return f"SKIP {patient_id}-{pulse} ({spacing}): already exists"

    lr_k = lr_key(spacing, patient_id, pulse)
    hr_k = hr_key(patient_id, pulse)

    if not has_key(source_h5, lr_k):
        return f"MISS {patient_id}-{pulse} ({spacing}): LR key not found"

    if not has_key(source_h5, hr_k):
        return f"MISS {patient_id}-{pulse} ({spacing}): HR key not found"

    # Load volumes
    lr_data, _ = read_volume(source_h5, lr_k)
    hr_data, hr_affine = read_volume(source_h5, hr_k)
    hr_shape = hr_data.shape[:3]

    # Compute zoom factors: only z-axis differs for anisotropic data
    zoom_factors = tuple(hr_shape[i] / lr_data.shape[i] for i in range(3))

    # Apply cubic B-spline interpolation
    sr_data = zoom(lr_data, zoom_factors, order=3)

    # Ensure exact shape match (rounding can cause off-by-one)
    if sr_data.shape != hr_shape:
        sr_data = sr_data[: hr_shape[0], : hr_shape[1], : hr_shape[2]]

    # Save with HR affine
    write_volume(bspline_h5, out_key, sr_data.astype(np.float32), hr_affine)

    return f"OK   {patient_id}-{pulse} ({spacing}): {lr_data.shape} -> {sr_data.shape}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate BSPLINE expert outputs")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[BSPLINE] %(levelname)s | %(message)s"
    )

    from pacs_sr.config.config import load_full_config

    full = load_full_config(args.config)
    data = full.data

    source_h5 = data.source_h5
    bspline_h5 = expert_h5_path(data.experts_dir, "BSPLINE")

    # Ensure experts directory and output file parent exist
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

    # Execute in parallel
    # Note: h5py with gzip compression is thread-safe for writing to different keys,
    # but joblib uses processes by default which is safer.
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_process_one)(*task) for task in tasks
    )

    # Report summary
    n_ok = sum(1 for r in results if r.startswith("OK"))
    n_skip = sum(1 for r in results if r.startswith("SKIP"))
    n_miss = sum(1 for r in results if r.startswith("MISS"))
    LOG.info("Done: %d OK, %d skipped, %d missing", n_ok, n_skip, n_miss)

    # Log any missing
    for r in results:
        if r.startswith("MISS"):
            LOG.warning(r)


if __name__ == "__main__":
    main()
