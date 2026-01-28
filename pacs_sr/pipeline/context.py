"""
Pipeline Context
================

Shared execution context for all pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pacs_sr.config.config import FullConfig


@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline stages.

    Contains all configuration, paths, and runtime state needed
    for pipeline execution.
    """

    # Configuration
    config: FullConfig

    # Experiment identity
    experiment_name: str
    experiment_dir: Path
    started_at: datetime = field(default_factory=datetime.now)

    # Execution scope
    folds: Tuple[int, ...] = ()
    spacings: Tuple[str, ...] = ()
    pulses: Tuple[str, ...] = ()

    # Runtime state
    manifest: Optional[Dict[str, Any]] = None
    logger: Optional[logging.Logger] = None

    # Options
    resume: bool = True
    dry_run: bool = False
    verbose: bool = False

    @property
    def training_dir(self) -> Path:
        """Directory for training outputs."""
        return self.experiment_dir / "training"

    @property
    def predictions_dir(self) -> Path:
        """Directory for blended predictions."""
        return self.experiment_dir / "predictions"

    @property
    def analysis_dir(self) -> Path:
        """Directory for analysis outputs."""
        return self.experiment_dir / "analysis"

    @property
    def figures_dir(self) -> Path:
        """Directory for generated figures."""
        return self.experiment_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.experiment_dir / "logs"

    @property
    def config_path(self) -> Path:
        """Path to frozen config copy."""
        return self.experiment_dir / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        """Path to manifest file."""
        return self.experiment_dir / "manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        """Path to checkpoint state file."""
        return self.experiment_dir / "pipeline_state.json"

    def get_training_task_dir(self, fold: int, spacing: str, pulse: str) -> Path:
        """Get directory for a specific training task."""
        return self.training_dir / f"fold_{fold}" / spacing / pulse

    def get_prediction_dir(self, spacing: str) -> Path:
        """Get directory for predictions at a specific spacing."""
        return self.predictions_dir / spacing

    def get_training_tasks(self) -> List[Tuple[int, str, str]]:
        """
        Get list of all training tasks as (fold, spacing, pulse) tuples.

        Returns:
            List of (fold, spacing, pulse) tuples
        """
        tasks = []
        for fold in self.folds:
            for spacing in self.spacings:
                for pulse in self.pulses:
                    tasks.append((fold, spacing, pulse))
        return tasks

    def task_id(self, fold: int, spacing: str, pulse: str) -> str:
        """Generate unique task ID for checkpoint tracking."""
        return f"fold_{fold}_{spacing}_{pulse}"

    def log(self, message: str, level: str = "info") -> None:
        """Log a message through the context logger."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def create_directories(self) -> None:
        """Create all output directories."""
        dirs = [
            self.experiment_dir,
            self.training_dir,
            self.predictions_dir,
            self.analysis_dir,
            self.figures_dir,
            self.figures_dir / "metrics",
            self.figures_dir / "weights",
            self.figures_dir / "patient_examples",
            self.logs_dir,
        ]

        # Create spacing-specific prediction directories
        for spacing in self.spacings:
            dirs.append(self.get_prediction_dir(spacing))

        # Create training task directories
        for fold, spacing, pulse in self.get_training_tasks():
            dirs.append(self.get_training_task_dir(fold, spacing, pulse))

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
