"""
Setup Stage
===========

Validates configuration and creates output directory structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import PipelineStage, StageResult

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


class SetupStage(PipelineStage):
    """
    Setup stage: Validate configuration and create directories.

    Tasks:
    - Validate configuration completeness
    - Check data paths exist
    - Create output directory structure
    - Log configuration summary
    """

    @property
    def name(self) -> str:
        return "setup"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute setup stage."""
        self.log(context, "Starting setup...")

        # Validate configuration
        validation_result = self._validate_config(context)
        if not validation_result.success:
            return validation_result

        # Create directory structure
        self.log(context, "Creating directory structure...")
        context.create_directories()

        # Log configuration summary
        self._log_config_summary(context)

        self.log(context, "Setup completed successfully")
        return StageResult.ok("Setup completed")

    def _validate_config(self, context: "PipelineContext") -> StageResult:
        """Validate configuration completeness."""
        config = context.config

        # Check required data paths
        data_config = config.data

        # Check models root
        models_root = Path(data_config.models_root)
        if not models_root.exists():
            return StageResult.fail(f"Models root not found: {models_root}")

        # Check HR root
        hr_root = Path(data_config.hr_root)
        if not hr_root.exists():
            return StageResult.fail(f"HR root not found: {hr_root}")

        # Check that at least some expert directories exist
        models = config.pacs_sr.models
        found_models = []
        for model in models:
            model_dir = models_root / model
            if model_dir.exists():
                found_models.append(model)

        if not found_models:
            return StageResult.fail(
                f"No expert model directories found in {models_root}. "
                f"Expected: {list(models)}"
            )

        if len(found_models) < len(models):
            missing = set(models) - set(found_models)
            self.log(context, f"Warning: Missing expert directories: {missing}", "warning")

        # Check manifest path or data availability
        manifest_path = Path(data_config.out)
        if not manifest_path.exists():
            self.log(context, "Manifest not found - will be generated", "info")

        # Check registration atlas if enabled
        pacs_sr_config = config.pacs_sr
        if pacs_sr_config.use_registration:
            if config.registration:
                atlas_dir = Path(config.registration.atlas)
                if not atlas_dir.exists():
                    return StageResult.fail(f"Atlas directory not found: {atlas_dir}")
            else:
                return StageResult.fail("Registration enabled but no registration config provided")

        self.log(context, "Configuration validated successfully")
        return StageResult.ok()

    def _log_config_summary(self, context: "PipelineContext") -> None:
        """Log configuration summary."""
        config = context.config

        self.log(context, "Configuration Summary:")
        self.log(context, f"  Experiment: {context.experiment_name}")
        self.log(context, f"  Folds: {list(context.folds)}")
        self.log(context, f"  Spacings: {list(context.spacings)}")
        self.log(context, f"  Pulses: {list(context.pulses)}")
        self.log(context, f"  Models: {list(config.pacs_sr.models)}")

        total_tasks = len(context.folds) * len(context.spacings) * len(context.pulses)
        self.log(context, f"  Total training tasks: {total_tasks}")

        self.log(context, f"  Patch size: {config.pacs_sr.patch_size}")
        self.log(context, f"  Stride: {config.pacs_sr.stride}")
        self.log(context, f"  Simplex: {config.pacs_sr.simplex}")
        self.log(context, f"  Registration: {config.pacs_sr.use_registration}")
