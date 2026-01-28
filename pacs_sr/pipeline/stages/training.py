"""
Training Stage
==============

Executes training across all folds, spacings, and pulses with checkpointing.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

from .base import PipelineStage, StageResult

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


class TrainingStage(PipelineStage):
    """
    Training stage: Execute training for all fold/spacing/pulse combinations.

    Tasks:
    - Iterate over all training tasks (fold × spacing × pulse)
    - Skip already completed tasks (checkpoint-aware)
    - Train model using PatchwiseConvexStacker
    - Evaluate on train/test splits
    - Save weights and metrics
    - Checkpoint progress after each task
    """

    @property
    def name(self) -> str:
        return "training"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute training stage."""
        self.log(context, "Starting training stage...")

        # Get training tasks
        tasks = list(context.get_training_tasks())
        total_tasks = len(tasks)

        # Get progress
        progress = checkpoint.get_training_progress()
        completed = progress["completed"]

        self.log(
            context,
            f"Training tasks: {completed}/{total_tasks} completed"
        )

        # Track failures
        failed_tasks = []

        for i, (fold, spacing, pulse) in enumerate(tasks):
            task_id = context.task_id(fold, spacing, pulse)

            # Skip if already completed
            if checkpoint.is_task_completed(task_id):
                self.log(context, f"[{i+1}/{total_tasks}] Skipping {task_id} (already completed)")
                continue

            self.log(context, f"[{i+1}/{total_tasks}] Starting {task_id}...")
            checkpoint.mark_task_started(task_id)

            try:
                # Run training for this task
                metrics = self._train_task(context, fold, spacing, pulse)

                # Mark completed with metrics
                checkpoint.mark_task_completed(task_id, metrics)
                self.log(
                    context,
                    f"[{i+1}/{total_tasks}] Completed {task_id}: "
                    f"PSNR={metrics.get('test', {}).get('psnr', 'N/A'):.4f}, "
                    f"SSIM={metrics.get('test', {}).get('ssim', 'N/A'):.4f}"
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                checkpoint.mark_task_failed(task_id, error_msg)
                failed_tasks.append((task_id, error_msg))
                self.log(context, f"[{i+1}/{total_tasks}] FAILED {task_id}: {error_msg}", "error")

                # Log full traceback for debugging
                self.log(context, traceback.format_exc(), "debug")

                # Continue with next task (fault tolerance)
                continue

        # Summary
        final_progress = checkpoint.get_training_progress()
        self.log(
            context,
            f"Training complete: {final_progress['completed']}/{total_tasks} succeeded, "
            f"{final_progress['failed']} failed"
        )

        if failed_tasks:
            self.log(context, f"Failed tasks: {[t[0] for t in failed_tasks]}", "warning")
            # Still return success if some tasks completed
            if final_progress["completed"] > 0:
                return StageResult.ok(
                    f"Training completed with {len(failed_tasks)} failures",
                    data={"failed_tasks": failed_tasks}
                )
            else:
                return StageResult.fail(f"All training tasks failed")

        return StageResult.ok(f"Training completed: {final_progress['completed']} tasks")

    def _train_task(
        self,
        context: "PipelineContext",
        fold: int,
        spacing: str,
        pulse: str,
    ) -> Dict[str, Any]:
        """
        Train and evaluate for a single fold/spacing/pulse combination.

        Returns:
            Dictionary with train and test metrics
        """
        from pacs_sr.model.model import PatchwiseConvexStacker
        from pacs_sr.config.config import PacsSRConfig

        config = context.config
        manifest = context.manifest

        # Get fold data from manifest
        fold_data = manifest["folds"][fold - 1]  # fold is 1-indexed

        # Create fold-specific manifest structure expected by model
        fold_manifest = {
            "train": {entry["patient_id"]: entry for entry in fold_data["train"]},
            "test": {entry["patient_id"]: entry for entry in fold_data["test"]},
        }

        # Configure model for this task
        pacs_config = self._create_task_config(context, fold, spacing, pulse)

        # Initialize model
        model = PatchwiseConvexStacker(pacs_config, fold_num=fold)

        # Train
        self.log(context, f"  Fitting model for fold {fold}, {spacing}, {pulse}...")
        model.fit_one(fold_manifest, spacing, pulse)

        # Evaluate
        self.log(context, f"  Evaluating on train/test splits...")
        metrics = model.evaluate_split(fold_manifest, spacing, pulse)

        return metrics

    def _create_task_config(
        self,
        context: "PipelineContext",
        fold: int,
        spacing: str,
        pulse: str,
    ) -> "PacsSRConfig":
        """Create PacsSRConfig for a specific training task."""
        from pacs_sr.config.config import PacsSRConfig

        config = context.config.pacs_sr

        # Create output directory structure
        # training/{spacing}/model_data/fold_{N}/{pulse}/
        task_out_root = context.training_dir

        return PacsSRConfig(
            # Core parameters from config
            models=config.models,
            patch_size=config.patch_size,
            stride=config.stride,
            simplex=config.simplex,
            lambda_ridge=config.lambda_ridge,
            lambda_edge=config.lambda_edge,
            edge_power=config.edge_power,
            lambda_grad=config.lambda_grad,
            grad_operator=config.grad_operator,
            laplacian_tau=config.laplacian_tau,
            mixing_window=config.mixing_window,
            normalize=config.normalize,

            # Parallel processing
            num_workers=config.num_workers,
            parallel_backend=config.parallel_backend,

            # Registration
            use_registration=config.use_registration,
            atlas_dir=context.config.registration.atlas if context.config.registration else None,

            # Output settings
            out_root=str(task_out_root),
            experiment_name=f"fold_{fold}",
            save_blends=config.save_blends,
            save_weight_volumes=config.save_weight_volumes,
            evaluate_train=config.evaluate_train,

            # Logging
            log_level=config.log_level,
            log_to_file=config.log_to_file,
            log_region_freq=config.log_region_freq,
            disable_tqdm=config.disable_tqdm,

            # SSIM configuration
            ssim_axis=config.ssim_axis,

            # Spacings and pulses (for reference)
            spacings=config.spacings,
            pulses=config.pulses,
        )
