"""
Enhanced logging utilities for PaCS-SR training and evaluation.

This module provides comprehensive logging capabilities including:
- Session logging (configuration, training start/end)
- Real-time training progress tracking
- Validation and test metrics reporting
- Formatted output for better readability
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class PacsSRLogger:
    """Enhanced logger for PaCS-SR training and evaluation."""

    def __init__(self, name: str = "PaCS-SR", log_file: Optional[Path] = None):
        """
        Initialize the PaCS-SR logger.

        Args:
            name: Logger name
            log_file: Optional path to log file (if None, only console logging)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Track timing
        self.timers = {}

    def log_session_header(self):
        """Log a formatted session header."""
        self.logger.info("=" * 80)
        self.logger.info("PaCS-SR: Patchwise Convex Stacking for Super-Resolution")
        self.logger.info("=" * 80)
        self.logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_config(self, config: Any):
        """
        Log configuration parameters.

        Args:
            config: PacsSRConfig object or dictionary
        """
        self.logger.info("-" * 80)
        self.logger.info("Configuration Parameters:")
        self.logger.info("-" * 80)

        if hasattr(config, "__dict__"):
            config_dict = vars(config)
        else:
            config_dict = config

        # Group parameters by category
        categories = {
            "Experiment": ["experiment_name"],
            "Tile Geometry": ["patch_size", "stride"],
            "Optimization": ["simplex", "lambda_ridge", "laplacian_tau"],
            "Edge Weighting": ["lambda_edge", "edge_power"],
            "Normalization": ["normalize"],
            "Compute": ["num_workers", "parallel_backend", "device"],
            "Metrics": ["compute_lpips", "ssim_axis"],
            "Saving": ["save_weight_volumes", "save_blends"],
            "Data": ["spacings", "pulses", "models"],
            "Paths": ["cv_json", "out_root"]
        }

        for category, keys in categories.items():
            self.logger.info(f"\n  [{category}]")
            for key in keys:
                if key in config_dict:
                    value = config_dict[key]
                    # Format lists nicely
                    if isinstance(value, list):
                        value_str = ", ".join(map(str, value))
                        self.logger.info(f"    {key:20s}: [{value_str}]")
                    else:
                        self.logger.info(f"    {key:20s}: {value}")

        self.logger.info("-" * 80)

    def log_training_start(self, spacing: str, pulse: str, n_patients: int, n_regions: int):
        """
        Log the start of training for a specific configuration.

        Args:
            spacing: Spacing identifier (e.g., "3mm")
            pulse: Pulse sequence (e.g., "t1c")
            n_patients: Number of training patients
            n_regions: Number of regions/tiles
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"TRAINING: {spacing} | {pulse}")
        self.logger.info("=" * 80)
        self.logger.info(f"Training patients: {n_patients}")
        self.logger.info(f"Number of regions: {n_regions}")
        self.logger.info("-" * 80)

        # Start timer
        self.timers[f"{spacing}_{pulse}_train"] = time.time()

    def log_patient_progress(self, patient_id: str, idx: int, total: int, elapsed_time: Optional[float] = None):
        """
        Log progress during patient processing.

        Args:
            patient_id: Patient identifier
            idx: Current patient index (0-based)
            total: Total number of patients
            elapsed_time: Optional time taken for this patient (in seconds)
        """
        progress_pct = (idx + 1) / total * 100
        msg = f"  [{idx+1:3d}/{total:3d}] ({progress_pct:5.1f}%) Processing: {patient_id}"
        if elapsed_time is not None:
            msg += f" | Time: {elapsed_time:.2f}s"
        self.logger.info(msg)

    def log_region_optimization(self, region_id: int, weights: list, objective_value: Optional[float] = None):
        """
        Log per-region optimization results.

        Args:
            region_id: Region identifier
            weights: Optimized weight vector
            objective_value: Final objective function value (optional)
        """
        weights_str = ", ".join([f"{w:.4f}" for w in weights])
        msg = f"    Region {region_id:4d}: weights=[{weights_str}]"
        if objective_value is not None:
            msg += f" | objective={objective_value:.6e}"
        self.logger.info(msg)

    def log_training_summary(self, n_regions: int, spacing: str, pulse: str):
        """
        Log summary after training completion.

        Args:
            n_regions: Number of regions optimized
            spacing: Spacing identifier
            pulse: Pulse sequence
        """
        timer_key = f"{spacing}_{pulse}_train"
        if timer_key in self.timers:
            elapsed = time.time() - self.timers[timer_key]
            self.logger.info("-" * 80)
            self.logger.info(f"Training completed: {n_regions} regions optimized in {elapsed:.2f}s")
            self.logger.info(f"Average time per region: {elapsed/n_regions:.4f}s")
            del self.timers[timer_key]

    def log_evaluation_start(self, spacing: str, pulse: str, split: str, n_patients: int):
        """
        Log the start of evaluation.

        Args:
            spacing: Spacing identifier
            pulse: Pulse sequence
            split: "train" or "test"
            n_patients: Number of patients to evaluate
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"EVALUATION ({split.upper()}): {spacing} | {pulse}")
        self.logger.info("=" * 80)
        self.logger.info(f"Evaluating {n_patients} {split} patients")
        self.logger.info("-" * 80)

        # Start timer
        self.timers[f"{spacing}_{pulse}_eval_{split}"] = time.time()

    def log_patient_metrics(self, patient_id: str, metrics: Dict[str, float], idx: int, total: int, elapsed_time: Optional[float] = None):
        """
        Log per-patient evaluation metrics.

        Args:
            patient_id: Patient identifier
            metrics: Dictionary of metric values
            idx: Current patient index
            total: Total patients
            elapsed_time: Optional time taken for this patient (in seconds)
        """
        progress_pct = (idx + 1) / total * 100
        metrics_str = " | ".join([f"{k.upper()}={v:.4f}" for k, v in metrics.items()])
        msg = f"  [{idx+1:3d}/{total:3d}] ({progress_pct:5.1f}%) {patient_id}: {metrics_str}"
        if elapsed_time is not None:
            msg += f" | Time: {elapsed_time:.2f}s"
        self.logger.info(msg)

    def log_aggregate_metrics(self, split: str, metrics: Dict[str, float], spacing: str, pulse: str):
        """
        Log aggregated metrics across all patients.

        Args:
            split: "train" or "test"
            metrics: Dictionary of aggregated metric values
            spacing: Spacing identifier
            pulse: Pulse sequence
        """
        timer_key = f"{spacing}_{pulse}_eval_{split}"
        elapsed = None
        if timer_key in self.timers:
            elapsed = time.time() - self.timers[timer_key]
            del self.timers[timer_key]

        self.logger.info("-" * 80)
        self.logger.info(f"Aggregate Metrics ({split.upper()}):")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name.upper():10s}: {value:.6f}")

        if elapsed is not None:
            self.logger.info(f"\nEvaluation time: {elapsed:.2f}s")
        self.logger.info("=" * 80)

    def log_weight_saving(self, path: Path, format: str = "npz"):
        """
        Log weight map saving.

        Args:
            path: Path where weights were saved
            format: File format (npz, json, etc.)
        """
        self.logger.info(f"Saved weight maps ({format.upper()}): {path}")

    def log_blend_saving(self, path: Path, patient_id: str):
        """
        Log blended prediction saving.

        Args:
            path: Path where blend was saved
            patient_id: Patient identifier
        """
        self.logger.info(f"  Saved blend for {patient_id}: {path}")

    def log_session_end(self):
        """Log session end."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def info(self, msg: str):
        """Convenience method for logging info messages."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Convenience method for logging warnings."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Convenience method for logging errors."""
        self.logger.error(msg)


def setup_experiment_logger(out_dir: Path, experiment_name: str) -> PacsSRLogger:
    """
    Create a logger for an experiment with file output.

    Args:
        out_dir: Output directory for logs
        experiment_name: Name of the experiment

    Returns:
        Configured PacsSRLogger instance
    """
    log_file = out_dir / f"{experiment_name}_training.log"
    return PacsSRLogger(name=f"PaCS-SR.{experiment_name}", log_file=log_file)
