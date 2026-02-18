#!/usr/bin/env python3
"""Convert NIfTI directory structure to HDF5 source_data.h5.

Reads from the existing NIfTI layout and writes a single HDF5 file containing
all high-resolution and low-resolution volumes with affine matrices as attributes.

Usage:
    python scripts/convert_nifti_to_hdf5.py \
        --hr-root /media/.../high_resolution \
        --lr-root /media/.../low_resolution \
        --output /media/.../source_data.h5 \
        --spacings 3mm 5mm 7mm \
        --pulses t1c t2w t2f
        
  ~/.conda/envs/pacs/bin/python scripts/convert_nifti_to_hdf5.py \
      --hr-root /media/mpascual/PortableSSD/BraTS_GLI/LowRes_HighRes/high_resolution \
      --lr-root /media/mpascual/PortableSSD/BraTS_GLI/LowRes_HighRes/low_resolution \
      --output /media/mpascual/PortableSSD/BraTS_GLI/LowRes_HighRes/source_data.h5 \
      --spacings 3mm 5mm 7mm \
      --pulses t1c t2w t2f
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from pacs_sr.data.hdf5_io import has_key, hr_key, lr_key, write_volume

LOG = logging.getLogger(__name__)


def _discover_patients(root: Path) -> list[str]:
    """List patient directories under *root*."""
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def convert_hr(
    hr_root: Path,
    output_h5: Path,
    pulses: list[str],
) -> int:
    """Convert high-resolution NIfTI volumes to HDF5.

    Args:
        hr_root: Directory containing {patient_id}/{patient_id}-{pulse}.nii.gz.
        output_h5: Output HDF5 file path.
        pulses: List of pulse sequences.

    Returns:
        Number of datasets written.
    """
    patients = _discover_patients(hr_root)
    LOG.info("HR: found %d patients in %s", len(patients), hr_root)

    written = 0
    for patient_id in tqdm(patients, desc="HR volumes"):
        for pulse in pulses:
            nii_path = hr_root / patient_id / f"{patient_id}-{pulse}.nii.gz"
            key = hr_key(patient_id, pulse)

            if has_key(output_h5, key):
                continue

            if not nii_path.exists():
                LOG.debug("Missing HR: %s", nii_path)
                continue

            img = nib.load(str(nii_path))
            data = img.get_fdata(dtype=np.float32)
            write_volume(output_h5, key, data, img.affine)
            written += 1

    return written


def convert_lr(
    lr_root: Path,
    output_h5: Path,
    spacings: list[str],
    pulses: list[str],
) -> int:
    """Convert low-resolution NIfTI volumes to HDF5.

    Args:
        lr_root: Directory containing {spacing}/{patient_id}/{patient_id}-{pulse}.nii.gz.
        output_h5: Output HDF5 file path.
        spacings: List of spacing strings.
        pulses: List of pulse sequences.

    Returns:
        Number of datasets written.
    """
    written = 0
    for spacing in spacings:
        spacing_dir = lr_root / spacing
        if not spacing_dir.exists():
            LOG.warning("LR spacing directory missing: %s", spacing_dir)
            continue

        patients = _discover_patients(spacing_dir)
        LOG.info("LR %s: found %d patients", spacing, len(patients))

        for patient_id in tqdm(patients, desc=f"LR {spacing}"):
            for pulse in pulses:
                nii_path = spacing_dir / patient_id / f"{patient_id}-{pulse}.nii.gz"
                key = lr_key(spacing, patient_id, pulse)

                if has_key(output_h5, key):
                    continue

                if not nii_path.exists():
                    LOG.debug("Missing LR: %s", nii_path)
                    continue

                img = nib.load(str(nii_path))
                data = img.get_fdata(dtype=np.float32)
                write_volume(output_h5, key, data, img.affine)
                written += 1

    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert NIfTI directory structure to HDF5 source_data.h5"
    )
    parser.add_argument(
        "--hr-root", type=Path, required=True, help="High-resolution NIfTI root"
    )
    parser.add_argument(
        "--lr-root", type=Path, required=True, help="Low-resolution NIfTI root"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output HDF5 file path"
    )
    parser.add_argument(
        "--spacings",
        nargs="+",
        default=["3mm", "5mm", "7mm"],
        help="Spacing strings (default: 3mm 5mm 7mm)",
    )
    parser.add_argument(
        "--pulses",
        nargs="+",
        default=["t1c", "t2w", "t2f"],
        help="Pulse sequences (default: t1c t2w t2f)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[convert] %(levelname)s | %(message)s"
    )

    # Validate inputs
    if not args.hr_root.is_dir():
        LOG.error("HR root not found: %s", args.hr_root)
        sys.exit(1)
    if not args.lr_root.is_dir():
        LOG.error("LR root not found: %s", args.lr_root)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Output: %s", args.output)

    # Convert HR
    hr_count = convert_hr(args.hr_root, args.output, args.pulses)
    LOG.info("HR datasets written: %d", hr_count)

    # Convert LR
    lr_count = convert_lr(args.lr_root, args.output, args.spacings, args.pulses)
    LOG.info("LR datasets written: %d", lr_count)

    LOG.info("Total datasets written: %d", hr_count + lr_count)
    LOG.info("Done. File: %s", args.output)


if __name__ == "__main__":
    main()
