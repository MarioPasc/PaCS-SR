"""
Pipeline Orchestrator
=====================

Main orchestration class that coordinates all pipeline stages.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from pacs_sr.config.config import FullConfig, load_full_config

from .checkpoint import CheckpointManager, TaskStatus
from .context import PipelineContext


class PipelineOrchestrator:
    """
    Orchestrates the complete PaCS-SR pipeline.

    Coordinates execution of all stages with checkpoint/resume support:
    1. Setup - Validate config, create directories
    2. Manifest - Build/load K-fold manifest
    3. Training - Train across folds/spacings/pulses
    4. Analysis - Aggregate metrics, statistical tests
    5. Visualization - Generate publication figures
    6. Report - Optional HTML summary
    """

    def __init__(
        self,
        config_path: Path,
        output_root: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        timestamp_suffix: bool = True,
        resume: bool = True,
        folds: Optional[List[int]] = None,
        spacings: Optional[List[str]] = None,
        pulses: Optional[List[str]] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config_path: Path to YAML configuration file
            output_root: Override output root directory
            experiment_name: Override experiment name
            timestamp_suffix: Append timestamp to experiment directory
            resume: Resume from checkpoint if available
            folds: Specific folds to run (default: all)
            spacings: Specific spacings to run (default: all)
            pulses: Specific pulses to run (default: all)
            dry_run: Validate config and show plan without executing
            verbose: Enable verbose logging
        """
        self.config_path = Path(config_path)
        self.resume = resume
        self.dry_run = dry_run
        self.verbose = verbose

        # Load configuration
        self.full_config = load_full_config(self.config_path)

        # Determine experiment settings
        self._experiment_name = experiment_name or self._get_experiment_name()
        self._output_root = output_root or self._get_output_root()
        self._timestamp_suffix = timestamp_suffix

        # Determine scope
        self._folds = tuple(folds) if folds else self._get_default_folds()
        self._spacings = tuple(spacings) if spacings else tuple(self.full_config.pacs_sr.spacings)
        self._pulses = tuple(pulses) if pulses else tuple(self.full_config.pacs_sr.pulses)

        # Will be initialized in run()
        self.context: Optional[PipelineContext] = None
        self.checkpoint: Optional[CheckpointManager] = None
        self.logger: Optional[logging.Logger] = None
        self._stages: List = []

    def _get_experiment_name(self) -> str:
        """Get experiment name from config or generate default."""
        # Check for pipeline section in config
        if hasattr(self.full_config, 'pipeline') and self.full_config.pipeline:
            return self.full_config.pipeline.experiment_name
        return self.full_config.pacs_sr.experiment_name

    def _get_output_root(self) -> Path:
        """Get output root from config or default."""
        if hasattr(self.full_config, 'pipeline') and self.full_config.pipeline:
            return Path(self.full_config.pipeline.output_root)
        return Path(self.full_config.pacs_sr.out_root).parent

    def _get_default_folds(self) -> tuple:
        """Get default folds from config."""
        kfolds = self.full_config.data.kfolds
        return tuple(range(1, kfolds + 1))

    def _create_experiment_dir(self) -> Path:
        """Create experiment directory with optional timestamp."""
        if self._timestamp_suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{self._experiment_name}_{timestamp}"
        else:
            dir_name = self._experiment_name

        experiment_dir = self._output_root / dir_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def _find_existing_experiment(self) -> Optional[Path]:
        """Find existing experiment directory to resume."""
        if not self._output_root.exists():
            return None

        # Look for directories matching experiment name pattern
        pattern = f"{self._experiment_name}_*" if self._timestamp_suffix else self._experiment_name
        matching_dirs = sorted(self._output_root.glob(pattern), reverse=True)

        for exp_dir in matching_dirs:
            checkpoint_path = exp_dir / "pipeline_state.json"
            if checkpoint_path.exists():
                return exp_dir

        return None

    def _setup_logger(self, log_dir: Path) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger("pacs_sr.pipeline")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        console.setFormatter(console_fmt)
        logger.addHandler(console)

        # File handler
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        return logger

    def _initialize_stages(self) -> List:
        """Initialize pipeline stages."""
        from .stages import (
            SetupStage,
            ManifestStage,
            TrainingStage,
            AnalysisStage,
            VisualizationStage,
        )

        return [
            SetupStage(),
            ManifestStage(),
            TrainingStage(),
            AnalysisStage(),
            VisualizationStage(),
        ]

    def _create_context(self, experiment_dir: Path) -> PipelineContext:
        """Create pipeline execution context."""
        return PipelineContext(
            config=self.full_config,
            experiment_name=self._experiment_name,
            experiment_dir=experiment_dir,
            folds=self._folds,
            spacings=self._spacings,
            pulses=self._pulses,
            resume=self.resume,
            dry_run=self.dry_run,
            verbose=self.verbose,
            logger=self.logger,
        )

    def _copy_config(self, experiment_dir: Path) -> None:
        """Copy config file to experiment directory."""
        dest = experiment_dir / "config.yaml"
        if not dest.exists():
            shutil.copy2(self.config_path, dest)

    def run(self) -> bool:
        """
        Execute the pipeline.

        Returns:
            True if pipeline completed successfully, False otherwise
        """
        # Determine experiment directory
        if self.resume:
            experiment_dir = self._find_existing_experiment()
            if experiment_dir:
                print(f"Resuming experiment: {experiment_dir}")
            else:
                experiment_dir = self._create_experiment_dir()
                print(f"Starting new experiment: {experiment_dir}")
        else:
            experiment_dir = self._create_experiment_dir()
            print(f"Starting new experiment: {experiment_dir}")

        # Setup logging
        self.logger = self._setup_logger(experiment_dir / "logs")
        self.logger.info(f"Pipeline started: {self._experiment_name}")
        self.logger.info(f"Output directory: {experiment_dir}")

        # Create context
        self.context = self._create_context(experiment_dir)

        # Initialize checkpoint
        self.checkpoint = CheckpointManager(self.context.checkpoint_path)

        if self.resume and self.checkpoint.load():
            self.logger.info("Loaded checkpoint from previous run")
            progress = self.checkpoint.get_training_progress()
            self.logger.info(
                f"Training progress: {progress['completed']}/{progress['total']} tasks completed"
            )
        else:
            # Initialize new checkpoint
            task_ids = [
                self.context.task_id(fold, spacing, pulse)
                for fold, spacing, pulse in self.context.get_training_tasks()
            ]
            config_hash = CheckpointManager.compute_config_hash(
                yaml.safe_load(self.config_path.read_text())
            )
            self.checkpoint.initialize(self._experiment_name, config_hash, task_ids)
            self.logger.info(f"Initialized checkpoint with {len(task_ids)} training tasks")

        # Copy config
        self._copy_config(experiment_dir)

        # Dry run mode
        if self.dry_run:
            self._print_execution_plan()
            return True

        # Initialize and run stages
        self._stages = self._initialize_stages()

        success = True
        for stage in self._stages:
            stage_name = stage.name

            # Check if stage should be skipped
            if self.checkpoint.is_stage_completed(stage_name):
                self.logger.info(f"Stage '{stage_name}' already completed, skipping")
                continue

            self.logger.info(f"Starting stage: {stage_name}")
            self.checkpoint.mark_stage_started(stage_name)

            try:
                result = stage.run(self.context, self.checkpoint)

                if result.success:
                    self.checkpoint.mark_stage_completed(stage_name)
                    self.logger.info(f"Stage '{stage_name}' completed successfully")
                else:
                    self.checkpoint.mark_stage_failed(stage_name, result.message)
                    self.logger.error(f"Stage '{stage_name}' failed: {result.message}")
                    success = False
                    break

            except Exception as e:
                self.checkpoint.mark_stage_failed(stage_name, str(e))
                self.logger.exception(f"Stage '{stage_name}' failed with exception")
                success = False
                break

        # Final summary
        if success:
            self.logger.info("Pipeline completed successfully")
            self._print_summary()
        else:
            self.logger.error("Pipeline failed - check logs for details")
            self.logger.info("Run with --resume to continue from checkpoint")

        return success

    def _print_execution_plan(self) -> None:
        """Print execution plan for dry run."""
        print("\n" + "=" * 60)
        print("EXECUTION PLAN (Dry Run)")
        print("=" * 60)

        print(f"\nExperiment: {self._experiment_name}")
        print(f"Output: {self.context.experiment_dir}")

        print(f"\nScope:")
        print(f"  Folds: {list(self._folds)}")
        print(f"  Spacings: {list(self._spacings)}")
        print(f"  Pulses: {list(self._pulses)}")

        total_tasks = len(self._folds) * len(self._spacings) * len(self._pulses)
        print(f"\nTraining tasks: {total_tasks}")

        if self.checkpoint:
            progress = self.checkpoint.get_training_progress()
            print(f"  Already completed: {progress['completed']}")
            print(f"  Remaining: {progress['pending']}")

        print("\nStages to run:")
        stages = ["setup", "manifest", "training", "analysis", "visualization"]
        for stage in stages:
            if self.checkpoint and self.checkpoint.is_stage_completed(stage):
                print(f"  [SKIP] {stage}")
            else:
                print(f"  [RUN]  {stage}")

        print("\n" + "=" * 60)

    def _print_summary(self) -> None:
        """Print pipeline completion summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)

        print(f"\nExperiment: {self._experiment_name}")
        print(f"Output: {self.context.experiment_dir}")

        # Training progress
        if self.checkpoint:
            progress = self.checkpoint.get_training_progress()
            print(f"\nTraining: {progress['completed']}/{progress['total']} tasks")
            if progress['failed'] > 0:
                print(f"  Failed: {progress['failed']}")

        # Output locations
        print(f"\nOutputs:")
        print(f"  Training: {self.context.training_dir}")
        print(f"  Predictions: {self.context.predictions_dir}")
        print(f"  Analysis: {self.context.analysis_dir}")
        print(f"  Figures: {self.context.figures_dir}")

        print("\n" + "=" * 60)
