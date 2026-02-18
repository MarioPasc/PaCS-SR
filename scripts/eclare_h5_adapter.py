#!/usr/bin/env python3
"""ECLARE HDF5 adapter: extract LR from HDF5, run ECLARE, ingest result back.

For a single (patient, spacing, pulse):
  1. Extract LR volume from source_data.h5 to a temp NIfTI file
  2. Run `run-eclare` CLI on the temp NIfTI
  3. Load the ECLARE output and write it back into eclare.h5

Usage:
    python scripts/eclare_h5_adapter.py \
        --source-h5 path/to/source_data.h5 \
        --expert-h5 path/to/eclare.h5 \
        --patient-id BraTS-GLI-00033-100 \
        --spacing 3mm --pulse t1c \
        --thickness 3 --gpu-id 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from pacs_sr.data.hdf5_io import (
    expert_key,
    has_key,
    lr_key,
    read_volume,
    write_volume,
)


def run_adapter(
    source_h5: Path,
    expert_h5: Path,
    patient_id: str,
    spacing: str,
    pulse: str,
    thickness: int,
    gpu_id: int,
) -> bool:
    """Extract LR → run ECLARE → ingest back into expert HDF5.

    Args:
        source_h5: Path to source_data.h5.
        expert_h5: Path to eclare.h5 (created if not exists).
        patient_id: Patient ID string.
        spacing: Spacing label (e.g., "3mm").
        pulse: Pulse sequence (e.g., "t1c").
        thickness: Relative slice thickness for ECLARE.
        gpu_id: GPU device index.

    Returns:
        True if successful, False otherwise.
    """
    out_key = expert_key(spacing, patient_id, pulse)

    # Skip if already done
    if has_key(expert_h5, out_key):
        print(f"SKIP {patient_id}-{pulse} ({spacing}): already in {expert_h5}")
        return True

    lr_k = lr_key(spacing, patient_id, pulse)
    if not has_key(source_h5, lr_k):
        print(f"MISS {patient_id}-{pulse} ({spacing}): LR key not found in {source_h5}")
        return False

    tmp_dir = tempfile.mkdtemp(prefix="eclare_")
    try:
        # 1. Extract LR volume to temp NIfTI
        lr_data, lr_affine = read_volume(source_h5, lr_k)
        tmp_input = Path(tmp_dir) / f"{patient_id}-{pulse}.nii.gz"
        nib.save(nib.Nifti1Image(lr_data, lr_affine), str(tmp_input))

        # 2. Run ECLARE
        cmd = [
            "run-eclare",
            "--in-fpath",
            str(tmp_input),
            "--out-dir",
            tmp_dir,
            "--gpu-id",
            str(gpu_id),
            "--relative-slice-thickness",
            str(thickness),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: ECLARE failed for {patient_id}-{pulse} ({spacing})")
            print(f"  stderr: {result.stderr[:500]}")
            return False

        # 3. Find ECLARE output (appends _eclare suffix by default)
        eclare_output = Path(tmp_dir) / f"{patient_id}-{pulse}_eclare.nii.gz"
        if not eclare_output.exists():
            # Try glob in case naming differs
            candidates = list(Path(tmp_dir).glob("*.nii.gz"))
            # Exclude the input file
            candidates = [c for c in candidates if c != tmp_input]
            if candidates:
                eclare_output = candidates[0]
            else:
                print(f"ERROR: ECLARE produced no output for {patient_id}-{pulse}")
                return False

        # 4. Load and ingest into HDF5
        eclare_img = nib.load(str(eclare_output))
        eclare_data = eclare_img.get_fdata(dtype=np.float32)
        eclare_affine = eclare_img.affine

        expert_h5.parent.mkdir(parents=True, exist_ok=True)
        write_volume(expert_h5, out_key, eclare_data, eclare_affine)
        print(f"OK   {patient_id}-{pulse} ({spacing}): {eclare_data.shape}")
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ECLARE HDF5 adapter: extract → run-eclare → ingest"
    )
    parser.add_argument(
        "--source-h5", type=Path, required=True, help="Path to source_data.h5"
    )
    parser.add_argument(
        "--expert-h5", type=Path, required=True, help="Path to eclare.h5"
    )
    parser.add_argument("--patient-id", type=str, required=True, help="Patient ID")
    parser.add_argument(
        "--spacing", type=str, required=True, help="Spacing label (e.g., '3mm')"
    )
    parser.add_argument(
        "--pulse", type=str, required=True, help="Pulse sequence (e.g., 't1c')"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        required=True,
        help="Relative slice thickness for ECLARE",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    ok = run_adapter(
        source_h5=args.source_h5,
        expert_h5=args.expert_h5,
        patient_id=args.patient_id,
        spacing=args.spacing,
        pulse=args.pulse,
        thickness=args.thickness,
        gpu_id=args.gpu_id,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
