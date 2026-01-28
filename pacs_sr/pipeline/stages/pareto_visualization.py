"""
Pareto Visualization Stage
==========================

Generates Pareto frontier plots comparing PaCS-SR vs expert models.
Reads from pre-computed metrics (no recomputation needed).

Key visualizations:
- LPIPS vs KID Pareto frontiers per spacing/pulse
- Method comparison bar charts
- Summary tables
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


class ParetoVisualizationStage(PipelineStage):
    """
    Pareto visualization stage: Generate Pareto frontier plots.

    Reads metrics from analysis/metrics_for_pareto.csv and generates:
    - Pareto frontier scatter plots (LPIPS vs KID)
    - Per-spacing and per-pulse comparison charts
    - Summary comparison tables
    """

    @property
    def name(self) -> str:
        return "pareto_visualization"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute Pareto visualization stage."""
        self.log(context, "Starting Pareto visualization stage...")

        # Check for matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError as e:
            return StageResult.fail(f"Required libraries not available: {e}")

        # Load pre-computed metrics
        pareto_path = context.analysis_dir / "metrics_for_pareto.csv"
        if not pareto_path.exists():
            return StageResult.fail(f"Pareto metrics not found: {pareto_path}")

        df = pd.read_csv(pareto_path)

        if df.empty:
            return StageResult.fail("No metrics data available")

        # Get figure configuration
        fig_config = self._get_figure_config(context)

        # Create output directories
        pareto_dir = context.figures_dir / "pareto"
        pareto_dir.mkdir(parents=True, exist_ok=True)

        generated = []

        # 1. Overall Pareto frontier (all conditions combined)
        self.log(context, "Generating overall Pareto frontier...")
        path = self._plot_overall_pareto(df, pareto_dir, fig_config)
        if path:
            generated.append(path)

        # 2. Per-spacing Pareto frontiers
        self.log(context, "Generating per-spacing Pareto frontiers...")
        for spacing in df["spacing"].unique():
            df_spacing = df[df["spacing"] == spacing]
            path = self._plot_pareto_frontier(
                df_spacing, f"Spacing: {spacing}",
                pareto_dir / f"pareto_{spacing}", fig_config
            )
            if path:
                generated.append(path)

        # 3. Per-pulse Pareto frontiers
        self.log(context, "Generating per-pulse Pareto frontiers...")
        for pulse in df["pulse"].unique():
            df_pulse = df[df["pulse"] == pulse]
            path = self._plot_pareto_frontier(
                df_pulse, f"Sequence: {pulse.upper()}",
                pareto_dir / f"pareto_{pulse}", fig_config
            )
            if path:
                generated.append(path)

        # 4. Method comparison bar charts
        self.log(context, "Generating method comparison charts...")
        comp_paths = self._plot_method_comparison(df, pareto_dir, fig_config)
        generated.extend(comp_paths)

        # 5. Faceted Pareto grid (all spacings x pulses)
        self.log(context, "Generating faceted Pareto grid...")
        path = self._plot_faceted_pareto(df, pareto_dir, fig_config)
        if path:
            generated.append(path)

        self.log(context, f"Pareto visualization complete: {len(generated)} figures")
        return StageResult.ok(
            f"Generated {len(generated)} Pareto figures",
            data={"figures": [str(p) for p in generated]}
        )

    def _get_figure_config(self, context: "PipelineContext") -> Dict[str, Any]:
        """Get figure configuration."""
        config = {
            "dpi": 300,
            "formats": ["pdf", "png"],
            "figsize": (10, 8),
        }

        if hasattr(context.config, "pipeline") and context.config.pipeline:
            if hasattr(context.config.pipeline, "figures"):
                fig_cfg = context.config.pipeline.figures
                if hasattr(fig_cfg, "dpi"):
                    config["dpi"] = fig_cfg.dpi
                if hasattr(fig_cfg, "format"):
                    config["formats"] = list(fig_cfg.format)

        return config

    def _plot_overall_pareto(
        self,
        df,
        output_dir: Path,
        fig_config: Dict[str, Any],
    ) -> Optional[Path]:
        """Plot overall Pareto frontier aggregated across conditions."""
        import matplotlib.pyplot as plt

        # Aggregate by method
        agg = df.groupby("method").agg({
            "lpips_mean": "mean",
            "kid_mean": "mean",
            "psnr_mean": "mean",
            "ssim_mean": "mean",
        }).reset_index()

        return self._plot_pareto_frontier(
            agg, "Overall: PaCS-SR vs Experts",
            output_dir / "pareto_overall", fig_config,
            aggregated=True
        )

    def _plot_pareto_frontier(
        self,
        df,
        title: str,
        base_path: Path,
        fig_config: Dict[str, Any],
        aggregated: bool = False,
    ) -> Optional[Path]:
        """
        Plot Pareto frontier for LPIPS vs KID.

        Lower is better for both metrics, so Pareto-optimal points
        are in the lower-left corner.
        """
        import matplotlib.pyplot as plt

        # Check if we have the required metrics
        if "lpips_mean" not in df.columns or "kid_mean" not in df.columns:
            return None

        # Filter out rows with missing data
        df_clean = df.dropna(subset=["lpips_mean", "kid_mean"])
        if df_clean.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colors for methods
        methods = df_clean["method"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        method_colors = {m: colors[i] for i, m in enumerate(methods)}

        # Plot each method
        for method in methods:
            df_method = df_clean[df_clean["method"] == method]

            if aggregated:
                x = df_method["lpips_mean"].values
                y = df_method["kid_mean"].values
            else:
                # Average across conditions for this method
                x = [df_method["lpips_mean"].mean()]
                y = [df_method["kid_mean"].mean()]

            is_pacs_sr = method == "PACS_SR"
            marker = "*" if is_pacs_sr else "o"
            size = 300 if is_pacs_sr else 150
            edgecolor = "black" if is_pacs_sr else "none"
            linewidth = 2 if is_pacs_sr else 0

            ax.scatter(x, y, c=[method_colors[method]], s=size,
                       marker=marker, label=method, edgecolors=edgecolor,
                       linewidths=linewidth, alpha=0.8, zorder=10 if is_pacs_sr else 5)

        # Compute and plot Pareto frontier
        points = df_clean[["lpips_mean", "kid_mean"]].values
        pareto_mask = self._compute_pareto_mask(points)
        pareto_points = points[pareto_mask]

        if len(pareto_points) > 1:
            # Sort by lpips for line plotting
            sorted_idx = np.argsort(pareto_points[:, 0])
            pareto_sorted = pareto_points[sorted_idx]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                    "k--", alpha=0.5, linewidth=1.5, label="Pareto Frontier")

        ax.set_xlabel("LPIPS (lower = better)", fontsize=12)
        ax.set_ylabel("KID (lower = better)", fontsize=12)
        ax.set_title(f"{title}\nPerceptual Quality Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add annotation for PaCS-SR
        pacs_data = df_clean[df_clean["method"] == "PACS_SR"]
        if not pacs_data.empty:
            px = pacs_data["lpips_mean"].mean()
            py = pacs_data["kid_mean"].mean()
            ax.annotate("PaCS-SR", (px, py), xytext=(10, 10),
                        textcoords="offset points", fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", alpha=0.7))

        plt.tight_layout()

        saved_path = self._save_figure(fig, base_path, fig_config)
        plt.close(fig)

        return saved_path

    def _compute_pareto_mask(self, points: np.ndarray) -> np.ndarray:
        """
        Compute Pareto-optimal points (for minimization on both axes).

        Returns boolean mask of Pareto-optimal points.
        """
        n = len(points)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # j dominates i if j is <= on both and < on at least one
                    if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                        (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                        is_pareto[i] = False
                        break

        return is_pareto

    def _plot_method_comparison(
        self,
        df,
        output_dir: Path,
        fig_config: Dict[str, Any],
    ) -> List[Path]:
        """Plot bar charts comparing methods on each metric."""
        import matplotlib.pyplot as plt

        saved_paths = []

        # Aggregate by method
        agg = df.groupby("method").agg({
            "lpips_mean": ["mean", "std"],
            "kid_mean": ["mean", "std"],
            "psnr_mean": ["mean", "std"],
            "ssim_mean": ["mean", "std"],
        }).reset_index()

        # Flatten column names
        agg.columns = ["method", "lpips_mean", "lpips_std",
                       "kid_mean", "kid_std", "psnr_mean", "psnr_std",
                       "ssim_mean", "ssim_std"]

        methods = agg["method"].tolist()
        x = np.arange(len(methods))

        # Define colors (highlight PaCS-SR)
        colors = ["#FF6B6B" if m == "PACS_SR" else "#4ECDC4" for m in methods]

        # Create 2x2 subplot for all metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_info = [
            ("lpips", "LPIPS", "lower = better", axes[0, 0]),
            ("kid", "KID", "lower = better", axes[0, 1]),
            ("psnr", "PSNR (dB)", "higher = better", axes[1, 0]),
            ("ssim", "SSIM", "higher = better", axes[1, 1]),
        ]

        for metric, ylabel, note, ax in metrics_info:
            means = agg[f"{metric}_mean"].fillna(0).values
            stds = agg[f"{metric}_std"].fillna(0).values

            bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                          edgecolor="black", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} ({note})")
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

        plt.suptitle("PaCS-SR vs Expert Models: Metric Comparison",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = self._save_figure(fig, output_dir / "method_comparison", fig_config)
        if path:
            saved_paths.append(path)
        plt.close(fig)

        return saved_paths

    def _plot_faceted_pareto(
        self,
        df,
        output_dir: Path,
        fig_config: Dict[str, Any],
    ) -> Optional[Path]:
        """Plot faceted Pareto frontiers (spacing x pulse grid)."""
        import matplotlib.pyplot as plt

        spacings = sorted(df["spacing"].unique())
        pulses = sorted(df["pulse"].unique())

        if len(spacings) == 0 or len(pulses) == 0:
            return None

        n_rows = len(spacings)
        n_cols = len(pulses)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        # Ensure axes is always 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        methods = df["method"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        method_colors = {m: colors[i] for i, m in enumerate(methods)}

        for i, spacing in enumerate(spacings):
            for j, pulse in enumerate(pulses):
                ax = axes[i, j]

                df_sub = df[(df["spacing"] == spacing) & (df["pulse"] == pulse)]
                df_sub = df_sub.dropna(subset=["lpips_mean", "kid_mean"])

                if df_sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_title(f"{spacing} / {pulse.upper()}")
                    continue

                for method in methods:
                    df_method = df_sub[df_sub["method"] == method]
                    if df_method.empty:
                        continue

                    x = df_method["lpips_mean"].values
                    y = df_method["kid_mean"].values

                    is_pacs = method == "PACS_SR"
                    marker = "*" if is_pacs else "o"
                    size = 150 if is_pacs else 80

                    ax.scatter(x, y, c=[method_colors[method]], s=size,
                               marker=marker, alpha=0.8,
                               edgecolors="black" if is_pacs else "none",
                               linewidths=1 if is_pacs else 0)

                ax.set_title(f"{spacing} / {pulse.upper()}", fontsize=10)
                ax.grid(True, alpha=0.3)

                if i == n_rows - 1:
                    ax.set_xlabel("LPIPS")
                if j == 0:
                    ax.set_ylabel("KID")

        # Add legend
        handles = [plt.Line2D([0], [0], marker="*" if m == "PACS_SR" else "o",
                              color="w", markerfacecolor=method_colors[m],
                              markersize=10, label=m)
                   for m in methods]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99))

        plt.suptitle("Pareto Frontiers by Spacing and Sequence",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        path = self._save_figure(fig, output_dir / "pareto_faceted", fig_config)
        plt.close(fig)

        return path

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
            except Exception:
                pass

        return saved
