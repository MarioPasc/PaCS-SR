"""
Manifest Stage
==============

Builds or loads the K-fold cross-validation manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

from .base import PipelineStage, StageResult

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


class ManifestStage(PipelineStage):
    """
    Manifest stage: Build or load K-fold manifest.

    Tasks:
    - Check if manifest exists at configured path
    - If not, generate manifest using folds_builder
    - Copy manifest to experiment directory
    - Load and validate manifest structure
    """

    @property
    def name(self) -> str:
        return "manifest"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute manifest stage."""
        self.log(context, "Starting manifest stage...")

        config = context.config
        source_manifest_path = Path(config.data.out)

        # Check if manifest exists at source location
        if source_manifest_path.exists():
            self.log(context, f"Loading existing manifest: {source_manifest_path}")
        else:
            # Generate manifest
            self.log(context, "Manifest not found, generating...")
            try:
                self._generate_manifest(context, source_manifest_path)
            except Exception as e:
                return StageResult.fail(f"Failed to generate manifest: {e}")

        # Copy manifest to experiment directory
        dest_manifest_path = context.manifest_path
        if not dest_manifest_path.exists():
            shutil.copy2(source_manifest_path, dest_manifest_path)
            self.log(context, f"Copied manifest to: {dest_manifest_path}")

        # Load and validate manifest
        try:
            manifest = self._load_manifest(dest_manifest_path)
            context.manifest = manifest
        except Exception as e:
            return StageResult.fail(f"Failed to load manifest: {e}")

        # Validate manifest structure
        validation_result = self._validate_manifest(context, manifest)
        if not validation_result.success:
            return validation_result

        self.log(context, "Manifest stage completed successfully")
        return StageResult.ok("Manifest loaded", data={"manifest": manifest})

    def _generate_manifest(self, context: "PipelineContext", output_path: Path) -> None:
        """Generate K-fold manifest using folds_builder."""
        from pacs_sr.data.folds_builder import build_kfold_manifest

        config = context.config

        self.log(context, "Building K-fold manifest...")
        manifest = build_kfold_manifest(
            models_root=Path(config.data.models_root),
            hr_root=Path(config.data.hr_root),
            lr_root=Path(config.data.lr_root) if config.data.lr_root else None,
            spacings=list(config.data.spacings),
            pulses=list(config.data.pulses),
            models=list(config.data.models),
            kfolds=config.data.kfolds,
            seed=config.data.seed,
        )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.log(context, f"Generated manifest with {len(manifest['folds'])} folds")

    def _load_manifest(self, path: Path) -> Dict[str, Any]:
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _validate_manifest(
        self,
        context: "PipelineContext",
        manifest: Dict[str, Any],
    ) -> StageResult:
        """Validate manifest structure and content."""
        # Check required keys
        if "folds" not in manifest:
            return StageResult.fail("Manifest missing 'folds' key")

        folds = manifest["folds"]
        if not folds:
            return StageResult.fail("Manifest has no folds")

        # Check requested folds exist
        num_folds = len(folds)
        for fold in context.folds:
            if fold < 1 or fold > num_folds:
                return StageResult.fail(
                    f"Requested fold {fold} not in manifest (has {num_folds} folds)"
                )

        # Check fold structure
        for i, fold_data in enumerate(folds):
            if "train" not in fold_data or "test" not in fold_data:
                return StageResult.fail(f"Fold {i+1} missing 'train' or 'test' key")

            # Log fold info
            n_train = len(fold_data["train"])
            n_test = len(fold_data["test"])
            self.log(context, f"  Fold {i+1}: {n_train} train, {n_test} test")

        # Validate that requested spacings and pulses have data
        sample_patient = folds[0]["train"][0] if folds[0]["train"] else folds[0]["test"][0]

        for spacing in context.spacings:
            for pulse in context.pulses:
                # Check at least one model has this spacing/pulse
                found = False
                for model in context.config.pacs_sr.models:
                    if model in sample_patient:
                        if spacing in sample_patient[model]:
                            if pulse in sample_patient[model][spacing]:
                                found = True
                                break

                if not found:
                    self.log(
                        context,
                        f"Warning: No data found for {spacing}/{pulse}",
                        "warning"
                    )

        self.log(context, "Manifest validation passed")
        return StageResult.ok()
