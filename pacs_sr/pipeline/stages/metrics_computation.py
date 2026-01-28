"""
Metrics Computation Stage
=========================

Computes all metrics (PSNR, SSIM, LPIPS, KID) for PaCS-SR and expert models.
Saves comprehensive CSVs organized by spacing/pulse for downstream visualization.

This stage is separate from visualization to allow metric refinement without
recomputing expensive perceptual metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple

import numpy as np

from .base import PipelineStage, StageResult

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


class MetricsComputationStage(PipelineStage):
    """
    Metrics computation stage: Compute comprehensive metrics for all methods.

    Outputs:
    - metrics_by_case.csv: Per-patient metrics with columns:
        method, spacing, pulse, patient_id, psnr, ssim, lpips, kid
    - metrics_summary.csv: Aggregated metrics (mean, std) per method/spacing/pulse
    - metrics_for_pareto.csv: Formatted for Pareto frontier analysis
    """

    @property
    def name(self) -> str:
        return "metrics_computation"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute metrics computation stage."""
        self.log(context, "Starting metrics computation stage...")

        # Collect PaCS-SR metrics from training outputs
        self.log(context, "Collecting PaCS-SR metrics...")
        pacs_sr_metrics = self._collect_pacs_sr_metrics(context, checkpoint)

        if not pacs_sr_metrics:
            return StageResult.fail("No PaCS-SR metrics found")

        self.log(context, f"Collected {len(pacs_sr_metrics)} PaCS-SR metric entries")

        # Compute expert model metrics for comparison
        self.log(context, "Computing expert model metrics...")
        expert_metrics = self._compute_expert_metrics(context)

        self.log(context, f"Collected {len(expert_metrics)} expert metric entries")

        # Combine all metrics
        all_metrics = pacs_sr_metrics + expert_metrics

        # Save comprehensive CSVs
        self._save_metrics_by_case(context, all_metrics)
        self._save_metrics_summary(context, all_metrics)
        self._save_metrics_for_pareto(context, all_metrics)

        # Save per-spacing-pulse breakdown
        self._save_metrics_by_condition(context, all_metrics)

        self.log(context, "Metrics computation completed successfully")
        return StageResult.ok(
            f"Computed metrics for {len(all_metrics)} entries",
            data={"n_entries": len(all_metrics)}
        )

    def _collect_pacs_sr_metrics(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> List[Dict[str, Any]]:
        """Collect PaCS-SR metrics from training outputs."""
        metrics_list = []

        for fold, spacing, pulse in context.get_training_tasks():
            task_id = context.task_id(fold, spacing, pulse)

            # Try to load from metrics.json file
            metrics_path = (
                context.training_dir / spacing / "model_data" / f"fold_{fold}" / pulse / "metrics.json"
            )

            if not metrics_path.exists():
                # Also try checkpoint
                task_metrics = checkpoint.get_task_metrics(task_id)
                if task_metrics:
                    test_metrics = task_metrics.get("test", {})
                    if test_metrics:
                        # Per-patient metrics from checkpoint (aggregated)
                        metrics_list.append({
                            "method": "PACS_SR",
                            "fold": fold,
                            "spacing": spacing,
                            "pulse": pulse,
                            "patient_id": f"fold_{fold}_aggregate",
                            "psnr": test_metrics.get("psnr"),
                            "ssim": test_metrics.get("ssim"),
                            "lpips": test_metrics.get("lpips"),
                            "kid": test_metrics.get("kid"),
                            "mae": test_metrics.get("mae"),
                            "mse": test_metrics.get("mse"),
                        })
                continue

            try:
                with open(metrics_path, "r") as f:
                    file_metrics = json.load(f)

                # Check for per-patient metrics
                test_data = file_metrics.get("test", {})
                if isinstance(test_data, dict):
                    # Could be per-patient or aggregate
                    if "per_patient" in test_data:
                        for patient_id, patient_metrics in test_data["per_patient"].items():
                            metrics_list.append({
                                "method": "PACS_SR",
                                "fold": fold,
                                "spacing": spacing,
                                "pulse": pulse,
                                "patient_id": patient_id,
                                "psnr": patient_metrics.get("psnr"),
                                "ssim": patient_metrics.get("ssim"),
                                "lpips": patient_metrics.get("lpips"),
                                "kid": patient_metrics.get("kid"),
                                "mae": patient_metrics.get("mae"),
                                "mse": patient_metrics.get("mse"),
                            })
                    else:
                        # Aggregate metrics
                        metrics_list.append({
                            "method": "PACS_SR",
                            "fold": fold,
                            "spacing": spacing,
                            "pulse": pulse,
                            "patient_id": f"fold_{fold}_aggregate",
                            "psnr": test_data.get("psnr"),
                            "ssim": test_data.get("ssim"),
                            "lpips": test_data.get("lpips"),
                            "kid": test_data.get("kid"),
                            "mae": test_data.get("mae"),
                            "mse": test_data.get("mse"),
                        })

            except Exception as e:
                self.log(context, f"Error loading {metrics_path}: {e}", "warning")
                continue

        return metrics_list

    def _compute_expert_metrics(
        self,
        context: "PipelineContext",
    ) -> List[Dict[str, Any]]:
        """
        Compute metrics for expert models (BSPLINE, ECLARE, SMORE, etc.).

        This enables Pareto frontier comparison between PaCS-SR and individual experts.
        """
        metrics_list = []

        config = context.config
        models_root = Path(config.data.models_root)
        hr_root = Path(config.data.hr_root)
        manifest = context.manifest

        if manifest is None:
            self.log(context, "No manifest loaded, skipping expert metrics", "warning")
            return metrics_list

        # Import metric functions
        try:
            from pacs_sr.model.metrics import psnr, ssim3d_slicewise, mae, mse, kid_slicewise
            import nibabel as nib
        except ImportError as e:
            self.log(context, f"Cannot compute expert metrics: {e}", "warning")
            return metrics_list

        # Try to import LPIPS
        try:
            from pacs_sr.analysis.metrics import lpips_slice
            has_lpips = True
        except ImportError:
            has_lpips = False
            self.log(context, "LPIPS not available", "warning")

        expert_models = list(config.pacs_sr.models)

        # Process each fold's test set
        for fold_idx, fold_data in enumerate(manifest["folds"]):
            fold = fold_idx + 1
            test_patients = fold_data["test"]

            for spacing in context.spacings:
                for pulse in context.pulses:
                    for patient_id, patient_data in test_patients.items():
                        # Get HR path
                        hr_path = None
                        if "HR" in patient_data and spacing in patient_data["HR"]:
                            hr_path = patient_data["HR"][spacing].get(pulse)

                        if hr_path is None or not Path(hr_path).exists():
                            continue

                        try:
                            hr = nib.load(hr_path).get_fdata().astype(np.float32)
                            mask = hr > 0
                            data_range = float(hr.max() - hr.min())
                        except Exception:
                            continue

                        # Compute metrics for each expert
                        for expert in expert_models:
                            if expert not in patient_data:
                                continue
                            if spacing not in patient_data[expert]:
                                continue
                            if pulse not in patient_data[expert][spacing]:
                                continue

                            sr_path = patient_data[expert][spacing][pulse]
                            if not Path(sr_path).exists():
                                continue

                            try:
                                sr = nib.load(sr_path).get_fdata().astype(np.float32)

                                # Ensure shapes match
                                if sr.shape != hr.shape:
                                    continue

                                # Compute metrics
                                entry = {
                                    "method": expert,
                                    "fold": fold,
                                    "spacing": spacing,
                                    "pulse": pulse,
                                    "patient_id": patient_id,
                                    "psnr": psnr(sr, hr, data_range=data_range, mask=mask),
                                    "ssim": ssim3d_slicewise(sr, hr, mask=mask),
                                    "mae": mae(sr, hr, mask=mask),
                                    "mse": mse(sr, hr, mask=mask),
                                    "lpips": None,
                                    "kid": None,
                                }

                                # Compute perceptual metrics if enabled
                                if config.pacs_sr.compute_kid:
                                    entry["kid"] = kid_slicewise(sr, hr, mask=mask)

                                if config.pacs_sr.compute_lpips and has_lpips:
                                    # Compute slice-wise LPIPS
                                    lpips_vals = []
                                    for z in range(sr.shape[0]):
                                        if mask[z].any():
                                            lp = lpips_slice(hr[z], sr[z], mask[z])
                                            if not np.isnan(lp):
                                                lpips_vals.append(lp)
                                    entry["lpips"] = float(np.mean(lpips_vals)) if lpips_vals else None

                                metrics_list.append(entry)

                            except Exception as e:
                                continue

        return metrics_list

    def _save_metrics_by_case(
        self,
        context: "PipelineContext",
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        """Save per-case metrics to CSV."""
        import csv

        path = context.analysis_dir / "metrics_by_case.csv"
        context.analysis_dir.mkdir(parents=True, exist_ok=True)

        columns = ["method", "fold", "spacing", "pulse", "patient_id",
                   "psnr", "ssim", "lpips", "kid", "mae", "mse"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for entry in all_metrics:
                writer.writerow(entry)

        self.log(context, f"Saved metrics by case: {path}")

    def _save_metrics_summary(
        self,
        context: "PipelineContext",
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        """Save aggregated metrics summary."""
        import csv

        # Group by method, spacing, pulse
        groups = {}
        for entry in all_metrics:
            key = (entry["method"], entry["spacing"], entry["pulse"])
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)

        path = context.analysis_dir / "metrics_summary.csv"

        columns = ["method", "spacing", "pulse", "n",
                   "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
                   "lpips_mean", "lpips_std", "kid_mean", "kid_std"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for (method, spacing, pulse), entries in sorted(groups.items()):
                row = {
                    "method": method,
                    "spacing": spacing,
                    "pulse": pulse,
                    "n": len(entries),
                }

                for metric in ["psnr", "ssim", "lpips", "kid"]:
                    values = [e[metric] for e in entries if e.get(metric) is not None]
                    if values:
                        row[f"{metric}_mean"] = float(np.mean(values))
                        row[f"{metric}_std"] = float(np.std(values))
                    else:
                        row[f"{metric}_mean"] = None
                        row[f"{metric}_std"] = None

                writer.writerow(row)

        self.log(context, f"Saved metrics summary: {path}")

    def _save_metrics_for_pareto(
        self,
        context: "PipelineContext",
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        """Save metrics formatted for Pareto frontier analysis."""
        import csv

        # Group by method, spacing, pulse and compute means
        groups = {}
        for entry in all_metrics:
            key = (entry["method"], entry["spacing"], entry["pulse"])
            if key not in groups:
                groups[key] = {"lpips": [], "kid": [], "psnr": [], "ssim": []}

            for m in ["lpips", "kid", "psnr", "ssim"]:
                if entry.get(m) is not None:
                    groups[key][m].append(entry[m])

        path = context.analysis_dir / "metrics_for_pareto.csv"

        columns = ["method", "spacing", "pulse",
                   "lpips_mean", "kid_mean", "psnr_mean", "ssim_mean",
                   "is_pacs_sr"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for (method, spacing, pulse), values in sorted(groups.items()):
                row = {
                    "method": method,
                    "spacing": spacing,
                    "pulse": pulse,
                    "is_pacs_sr": 1 if method == "PACS_SR" else 0,
                }

                for m in ["lpips", "kid", "psnr", "ssim"]:
                    row[f"{m}_mean"] = float(np.mean(values[m])) if values[m] else None

                writer.writerow(row)

        self.log(context, f"Saved Pareto analysis data: {path}")

    def _save_metrics_by_condition(
        self,
        context: "PipelineContext",
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        """Save separate CSVs for each spacing/pulse combination."""
        import csv

        metrics_dir = context.analysis_dir / "by_condition"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Group by spacing and pulse
        by_spacing = {}
        by_pulse = {}

        for entry in all_metrics:
            spacing = entry["spacing"]
            pulse = entry["pulse"]

            if spacing not in by_spacing:
                by_spacing[spacing] = []
            by_spacing[spacing].append(entry)

            if pulse not in by_pulse:
                by_pulse[pulse] = []
            by_pulse[pulse].append(entry)

        columns = ["method", "fold", "patient_id", "psnr", "ssim", "lpips", "kid"]

        # Save by spacing
        for spacing, entries in by_spacing.items():
            path = metrics_dir / f"metrics_{spacing}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pulse"] + columns, extrasaction="ignore")
                writer.writeheader()
                for entry in entries:
                    row = {"pulse": entry["pulse"], **entry}
                    writer.writerow(row)

        # Save by pulse (sequence)
        for pulse, entries in by_pulse.items():
            path = metrics_dir / f"metrics_{pulse}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["spacing"] + columns, extrasaction="ignore")
                writer.writeheader()
                for entry in entries:
                    row = {"spacing": entry["spacing"], **entry}
                    writer.writerow(row)

        self.log(context, f"Saved condition-specific metrics to: {metrics_dir}")
