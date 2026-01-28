"""
Visualization Stage
===================

Generates publication-ready figures from analysis results.
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


class VisualizationStage(PipelineStage):
    """
    Visualization stage: Generate publication-ready figures.

    Tasks:
    - Generate metrics comparison tables
    - Create boxplots for PSNR/SSIM across conditions
    - Generate weight heatmaps
    - Create patient example visualizations
    """

    @property
    def name(self) -> str:
        return "visualization"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute visualization stage."""
        self.log(context, "Starting visualization stage...")

        # Check for matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return StageResult.fail("matplotlib not available for visualization")

        # Load aggregated metrics
        metrics_path = context.analysis_dir / "metrics_aggregated.json"
        if not metrics_path.exists():
            return StageResult.fail(f"Aggregated metrics not found: {metrics_path}")

        with open(metrics_path, "r") as f:
            aggregated = json.load(f)

        # Create figures directory structure
        figures_dir = context.figures_dir
        (figures_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (figures_dir / "weights").mkdir(parents=True, exist_ok=True)
        (figures_dir / "patient_examples").mkdir(parents=True, exist_ok=True)

        # Get figure settings from config
        fig_config = self._get_figure_config(context)

        # Generate figures
        generated = []

        # 1. Metrics table
        self.log(context, "Generating metrics table...")
        table_path = self._generate_metrics_table(context, aggregated, fig_config)
        if table_path:
            generated.append(table_path)

        # 2. Boxplots
        self.log(context, "Generating boxplots...")
        boxplot_paths = self._generate_boxplots(context, aggregated, fig_config)
        generated.extend(boxplot_paths)

        # 3. Weight heatmaps
        self.log(context, "Generating weight heatmaps...")
        heatmap_paths = self._generate_weight_heatmaps(context, fig_config)
        generated.extend(heatmap_paths)

        # 4. Per-fold comparison
        self.log(context, "Generating per-fold comparison...")
        fold_paths = self._generate_fold_comparison(context, aggregated, fig_config)
        generated.extend(fold_paths)

        self.log(context, f"Visualization complete: {len(generated)} figures generated")
        return StageResult.ok(
            f"Generated {len(generated)} figures",
            data={"figures": [str(p) for p in generated]}
        )

    def _get_figure_config(self, context: "PipelineContext") -> Dict[str, Any]:
        """Get figure configuration settings."""
        # Default settings
        config = {
            "dpi": 300,
            "formats": ["pdf", "png"],
            "figsize": (10, 6),
            "style": "seaborn-v0_8-whitegrid",
        }

        # Override from pipeline config if available
        if hasattr(context.config, "pipeline") and context.config.pipeline:
            if hasattr(context.config.pipeline, "figures"):
                fig_cfg = context.config.pipeline.figures
                if hasattr(fig_cfg, "dpi"):
                    config["dpi"] = fig_cfg.dpi
                if hasattr(fig_cfg, "format"):
                    config["formats"] = fig_cfg.format

        return config

    def _generate_metrics_table(
        self,
        context: "PipelineContext",
        aggregated: Dict[str, Any],
        fig_config: Dict[str, Any],
    ) -> Optional[Path]:
        """Generate metrics comparison table as figure."""
        import matplotlib.pyplot as plt

        overall = aggregated.get("overall", {})
        if not overall:
            return None

        # Create table data
        metrics = ["psnr", "ssim", "mae", "mse"]
        rows = []
        for m in metrics:
            if m in overall:
                stats = overall[m]
                rows.append([
                    m.upper(),
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                ])

        if not rows:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")

        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Mean", "Std", "Min", "Max"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", weight="bold")

        plt.title("PaCS-SR Overall Performance Metrics", fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()

        # Save
        base_path = context.figures_dir / "metrics" / "table_comparison"
        saved_path = self._save_figure(fig, base_path, fig_config)
        plt.close(fig)

        return saved_path

    def _generate_boxplots(
        self,
        context: "PipelineContext",
        aggregated: Dict[str, Any],
        fig_config: Dict[str, Any],
    ) -> List[Path]:
        """Generate boxplots for metrics across spacings."""
        import matplotlib.pyplot as plt

        saved_paths = []
        per_fold = aggregated.get("per_fold", {})

        if not per_fold:
            return saved_paths

        # Group data by spacing
        by_spacing = {}
        for task_id, data in per_fold.items():
            spacing = data["spacing"]
            if spacing not in by_spacing:
                by_spacing[spacing] = {"psnr": [], "ssim": []}
            if data.get("psnr"):
                by_spacing[spacing]["psnr"].append(data["psnr"])
            if data.get("ssim"):
                by_spacing[spacing]["ssim"].append(data["ssim"])

        spacings = sorted(by_spacing.keys())

        # PSNR boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        psnr_data = [by_spacing[s]["psnr"] for s in spacings if by_spacing[s]["psnr"]]
        if psnr_data:
            bp = ax.boxplot(psnr_data, labels=spacings, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#4472C4")
                patch.set_alpha(0.7)
            ax.set_xlabel("Spacing", fontsize=12)
            ax.set_ylabel("PSNR (dB)", fontsize=12)
            ax.set_title("PSNR Distribution by Spacing", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            base_path = context.figures_dir / "metrics" / "boxplot_psnr"
            saved_path = self._save_figure(fig, base_path, fig_config)
            if saved_path:
                saved_paths.append(saved_path)
        plt.close(fig)

        # SSIM boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        ssim_data = [by_spacing[s]["ssim"] for s in spacings if by_spacing[s]["ssim"]]
        if ssim_data:
            bp = ax.boxplot(ssim_data, labels=spacings, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#ED7D31")
                patch.set_alpha(0.7)
            ax.set_xlabel("Spacing", fontsize=12)
            ax.set_ylabel("SSIM", fontsize=12)
            ax.set_title("SSIM Distribution by Spacing", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            base_path = context.figures_dir / "metrics" / "boxplot_ssim"
            saved_path = self._save_figure(fig, base_path, fig_config)
            if saved_path:
                saved_paths.append(saved_path)
        plt.close(fig)

        return saved_paths

    def _generate_weight_heatmaps(
        self,
        context: "PipelineContext",
        fig_config: Dict[str, Any],
    ) -> List[Path]:
        """Generate weight heatmaps showing expert contributions."""
        import matplotlib.pyplot as plt

        saved_paths = []

        # Look for weight files in training directory
        weight_files = list(context.training_dir.glob("**/weights.json"))

        if not weight_files:
            self.log(context, "No weight files found for heatmap generation", "warning")
            return saved_paths

        # Load and aggregate weights across all folds
        all_weights = {}
        model_names = list(context.config.pacs_sr.models)

        for wf in weight_files:
            try:
                with open(wf, "r") as f:
                    weights = json.load(f)

                # Extract fold/spacing/pulse from path
                parts = wf.parts
                for i, p in enumerate(parts):
                    if p.startswith("fold_"):
                        fold = p
                        break
                else:
                    fold = "unknown"

                key = str(wf.parent)
                if key not in all_weights:
                    all_weights[key] = weights
            except Exception:
                continue

        if not all_weights:
            return saved_paths

        # Create average weight heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get sample weight file to determine structure
        sample_weights = next(iter(all_weights.values()))
        n_regions = len(sample_weights)
        n_models = len(model_names)

        # Compute average weights per region
        avg_matrix = np.zeros((min(n_regions, 50), n_models))  # Limit regions for visibility
        region_ids = sorted(sample_weights.keys(), key=int)[:50]

        for i, rid in enumerate(region_ids):
            w = sample_weights[rid]
            if isinstance(w, list):
                avg_matrix[i, :] = w

        # Plot heatmap
        im = ax.imshow(avg_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel("Region ID", fontsize=12)
        ax.set_xlabel("Expert Model", fontsize=12)
        ax.set_title("Expert Weights by Region (Sample)", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Weight", fontsize=12)

        plt.tight_layout()

        base_path = context.figures_dir / "weights" / "heatmap_expert_region"
        saved_path = self._save_figure(fig, base_path, fig_config)
        if saved_path:
            saved_paths.append(saved_path)
        plt.close(fig)

        # Weight distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        for j, model in enumerate(model_names):
            weights_for_model = avg_matrix[:, j].flatten()
            ax.hist(weights_for_model, bins=20, alpha=0.5, label=model)

        ax.set_xlabel("Weight", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Expert Weights", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        base_path = context.figures_dir / "weights" / "weight_distribution"
        saved_path = self._save_figure(fig, base_path, fig_config)
        if saved_path:
            saved_paths.append(saved_path)
        plt.close(fig)

        return saved_paths

    def _generate_fold_comparison(
        self,
        context: "PipelineContext",
        aggregated: Dict[str, Any],
        fig_config: Dict[str, Any],
    ) -> List[Path]:
        """Generate per-fold comparison plots."""
        import matplotlib.pyplot as plt

        saved_paths = []
        per_fold = aggregated.get("per_fold", {})

        if not per_fold:
            return saved_paths

        # Group by fold
        by_fold = {}
        for task_id, data in per_fold.items():
            fold = data["fold"]
            if fold not in by_fold:
                by_fold[fold] = {"psnr": [], "ssim": []}
            if data.get("psnr"):
                by_fold[fold]["psnr"].append(data["psnr"])
            if data.get("ssim"):
                by_fold[fold]["ssim"].append(data["ssim"])

        folds = sorted(by_fold.keys())

        # Bar plot comparing folds
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR
        psnr_means = [np.mean(by_fold[f]["psnr"]) if by_fold[f]["psnr"] else 0 for f in folds]
        psnr_stds = [np.std(by_fold[f]["psnr"]) if by_fold[f]["psnr"] else 0 for f in folds]

        axes[0].bar(range(len(folds)), psnr_means, yerr=psnr_stds, capsize=5, color="#4472C4", alpha=0.7)
        axes[0].set_xticks(range(len(folds)))
        axes[0].set_xticklabels([f"Fold {f}" for f in folds])
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title("PSNR by Fold")
        axes[0].grid(True, alpha=0.3, axis="y")

        # SSIM
        ssim_means = [np.mean(by_fold[f]["ssim"]) if by_fold[f]["ssim"] else 0 for f in folds]
        ssim_stds = [np.std(by_fold[f]["ssim"]) if by_fold[f]["ssim"] else 0 for f in folds]

        axes[1].bar(range(len(folds)), ssim_means, yerr=ssim_stds, capsize=5, color="#ED7D31", alpha=0.7)
        axes[1].set_xticks(range(len(folds)))
        axes[1].set_xticklabels([f"Fold {f}" for f in folds])
        axes[1].set_ylabel("SSIM")
        axes[1].set_title("SSIM by Fold")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.suptitle("Performance Comparison Across Folds", fontsize=14, fontweight="bold")
        plt.tight_layout()

        base_path = context.figures_dir / "metrics" / "fold_comparison"
        saved_path = self._save_figure(fig, base_path, fig_config)
        if saved_path:
            saved_paths.append(saved_path)
        plt.close(fig)

        return saved_paths

    def _save_figure(
        self,
        fig,
        base_path: Path,
        fig_config: Dict[str, Any],
    ) -> Optional[Path]:
        """Save figure in configured formats."""
        dpi = fig_config.get("dpi", 300)
        formats = fig_config.get("formats", ["pdf", "png"])

        saved = None
        for fmt in formats:
            path = base_path.with_suffix(f".{fmt}")
            try:
                fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
                if saved is None:
                    saved = path
            except Exception as e:
                pass  # Continue with other formats

        return saved
