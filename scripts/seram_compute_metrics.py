#!/usr/bin/env python3
"""Batch metric computation for SERAM experiments.

Computes 3D-MS-SSIM and MMD-MF for all (patient, spacing, pulse, model)
combinations and outputs a single CSV file.

Reads all volumes from HDF5 files:
  - GT from source_data.h5
  - Expert SR from {model}.h5
  - PaCS-SR blends from results_fold_{N}.h5

Usage:
    python scripts/seram_compute_metrics.py --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from joblib import Parallel, delayed

from pacs_sr.data.hdf5_io import (
    blend_key,
    expert_h5_path,
    expert_key,
    has_key,
    hr_key,
    read_volume,
    results_h5_path,
)
from pacs_sr.model.metrics_3d import ms_ssim_3d
from pacs_sr.model.metrics_mmd_mf import mmd_mf

LOG = logging.getLogger(__name__)


def _find_blend_h5(
    patient_id: str,
    spacing: str,
    pulse: str,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
) -> Optional[Path]:
    """Find which fold's results HDF5 contains a blend for a given patient.

    Args:
        patient_id: Patient ID.
        spacing: Spacing string.
        pulse: Pulse sequence.
        out_root: Output root directory.
        experiment_name: Experiment name.
        n_folds: Number of folds to search.

    Returns:
        Path to the results HDF5 file containing the blend, or None.
    """
    key = blend_key(spacing, patient_id, pulse)
    for fold_num in range(1, n_folds + 1):
        h5_path = results_h5_path(out_root, experiment_name, fold_num)
        if h5_path.exists() and has_key(h5_path, key):
            return h5_path
    return None


def _compute_patient_metrics(
    patient_id: str,
    spacing: str,
    pulse: str,
    model: str,
    source_h5: Path,
    experts_dir: Path,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
) -> Optional[Dict]:
    """Compute metrics for a single (patient, spacing, pulse, model) tuple.

    Args:
        patient_id: Patient identifier.
        spacing: Spacing string.
        pulse: Pulse sequence.
        model: Model name (expert name or "PACS_SR").
        source_h5: Path to source_data.h5.
        experts_dir: Directory containing {model}.h5 files.
        out_root: Output root for PaCS-SR results.
        experiment_name: Experiment name.
        n_folds: Number of folds.

    Returns:
        Dictionary with metric values, or None on failure.
    """
    try:
        # Load GT
        gt_k = hr_key(patient_id, pulse)
        if not has_key(source_h5, gt_k):
            LOG.debug("Missing GT: %s", gt_k)
            return None
        gt_vol, _ = read_volume(source_h5, gt_k)

        # Load SR prediction
        if model == "PACS_SR":
            blend_h5 = _find_blend_h5(
                patient_id, spacing, pulse, out_root, experiment_name, n_folds
            )
            if blend_h5 is None:
                LOG.debug("Missing blend for %s %s %s", patient_id, spacing, pulse)
                return None
            sr_vol, _ = read_volume(blend_h5, blend_key(spacing, patient_id, pulse))
        else:
            model_h5 = expert_h5_path(experts_dir, model)
            exp_k = expert_key(spacing, patient_id, pulse)
            if not has_key(model_h5, exp_k):
                LOG.debug("Missing SR: %s in %s", exp_k, model_h5)
                return None
            sr_vol, _ = read_volume(model_h5, exp_k)

        # Ensure shape match
        if sr_vol.shape != gt_vol.shape:
            LOG.warning(
                "Shape mismatch for %s %s %s %s: SR=%s GT=%s",
                patient_id,
                spacing,
                pulse,
                model,
                sr_vol.shape,
                gt_vol.shape,
            )
            min_shape = tuple(min(s, g) for s, g in zip(sr_vol.shape, gt_vol.shape))
            sr_vol = sr_vol[: min_shape[0], : min_shape[1], : min_shape[2]]
            gt_vol = gt_vol[: min_shape[0], : min_shape[1], : min_shape[2]]

        # Brain mask from GT
        mask = gt_vol > 0

        # Compute metrics
        ms_ssim_val = ms_ssim_3d(sr_vol, gt_vol, mask=mask)
        mmd_mf_val = mmd_mf(sr_vol, gt_vol, mask=mask)

        return {
            "patient": patient_id,
            "spacing": spacing,
            "pulse": pulse,
            "model": model,
            "ms_ssim_3d": ms_ssim_val,
            "mmd_mf": mmd_mf_val,
        }
    except Exception as exc:
        LOG.error("Failed %s %s %s %s: %s", patient_id, spacing, pulse, model, exc)
        return None


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SERAM batch metric computation")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Override manifest path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[SERAM-metrics] %(levelname)s | %(message)s"
    )

    from pacs_sr.config.config import load_full_config

    full = load_full_config(args.config)
    data = full.data
    pacs_sr = full.pacs_sr

    # Load manifest to get test patients per fold
    manifest_path = args.manifest or data.out
    LOG.info("Loading manifest: %s", manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Collect unique test patients across all folds (avoid duplicate computation)
    test_patients: set[str] = set()
    for fold_data in manifest["folds"]:
        test_patients.update(fold_data["test"])
    test_patients_sorted = sorted(test_patients)
    n_folds = len(manifest["folds"])
    LOG.info(
        "Found %d unique test patients across %d folds",
        len(test_patients_sorted),
        n_folds,
    )

    spacings = list(pacs_sr.spacings)
    pulses = list(pacs_sr.pulses)
    expert_models = list(pacs_sr.models)
    all_models = expert_models + ["PACS_SR"]

    # Build task list
    tasks = []
    for patient_id in test_patients_sorted:
        for spacing in spacings:
            for pulse in pulses:
                for model in all_models:
                    tasks.append(
                        (
                            patient_id,
                            spacing,
                            pulse,
                            model,
                            data.source_h5,
                            data.experts_dir,
                            Path(pacs_sr.out_root),
                            pacs_sr.experiment_name,
                            n_folds,
                        )
                    )

    LOG.info("Computing metrics for %d tasks", len(tasks))

    # Execute in parallel
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_compute_patient_metrics)(*task) for task in tasks
    )

    # Filter None results and build DataFrame
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)

    # Save CSV
    out_path = Path(pacs_sr.out_root) / pacs_sr.experiment_name / "seram_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.info("Saved %d rows to %s", len(df), out_path)

    # Print summary
    if not df.empty:
        print("\nMetrics Summary (mean +/- std):")
        print("=" * 80)
        for model in all_models:
            model_df = df[df["model"] == model]
            if model_df.empty:
                continue
            ms_ssim_mean = model_df["ms_ssim_3d"].mean()
            ms_ssim_std = model_df["ms_ssim_3d"].std()
            mmd_mean = model_df["mmd_mf"].mean()
            mmd_std = model_df["mmd_mf"].std()
            print(
                f"  {model:12s}: MS-SSIM={ms_ssim_mean:.4f}+/-{ms_ssim_std:.4f}  "
                f"MMD-MF={mmd_mean:.6f}+/-{mmd_std:.6f}"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
