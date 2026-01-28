#!/usr/bin/env python3
"""
Standalone Visualization Tool
=============================

Regenerates figures from pre-computed metrics without re-running the full pipeline.
Use this to refine visualizations after metrics have been computed.

Usage:
    pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_...
    pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_... --only pareto
    pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_... --only metrics weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate PaCS-SR figures from pre-computed metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate all figures
  pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_...

  # Only regenerate Pareto frontier plots
  pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_... --only pareto

  # Regenerate specific figure types
  pacs-sr-visualize --experiment-dir ./experiments/PaCS_SR_full_20260128_... --only metrics weights pareto
        """
    )

    parser.add_argument(
        "--experiment-dir", "-e",
        type=Path,
        required=True,
        help="Path to experiment directory (containing analysis/ and figures/)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config YAML (optional, uses experiment config if not provided)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["metrics", "weights", "pareto", "specialization", "folds"],
        help="Only generate specific figure types"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Override output directory for figures"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (default: pdf png)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def load_config(experiment_dir: Path, config_path: Optional[Path] = None):
    """Load configuration from experiment or specified path."""
    if config_path and config_path.exists():
        from pacs_sr.config.config import load_full_config
        return load_full_config(config_path)

    # Try experiment config
    exp_config = experiment_dir / "config.yaml"
    if exp_config.exists():
        from pacs_sr.config.config import load_full_config
        return load_full_config(exp_config)

    return None


def generate_pareto_figures(
    experiment_dir: Path,
    output_dir: Path,
    fig_config: dict,
    verbose: bool = False,
) -> List[Path]:
    """Generate Pareto frontier figures."""
    import pandas as pd

    pareto_path = experiment_dir / "analysis" / "metrics_for_pareto.csv"
    if not pareto_path.exists():
        print(f"Error: Pareto metrics not found: {pareto_path}")
        return []

    df = pd.read_csv(pareto_path)
    if df.empty:
        print("Warning: No metrics data in pareto file")
        return []

    # Import visualization stage
    from pacs_sr.pipeline.stages.pareto_visualization import ParetoVisualizationStage

    # Create a minimal context-like object for the visualization methods
    class MinimalContext:
        def __init__(self, exp_dir, out_dir):
            self.analysis_dir = exp_dir / "analysis"
            self.figures_dir = out_dir

    ctx = MinimalContext(experiment_dir, output_dir)
    stage = ParetoVisualizationStage()

    generated = []

    pareto_dir = output_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    # Overall Pareto
    if verbose:
        print("  Generating overall Pareto frontier...")
    path = stage._plot_overall_pareto(df, pareto_dir, fig_config)
    if path:
        generated.append(path)

    # Per-spacing
    for spacing in df["spacing"].unique():
        if verbose:
            print(f"  Generating Pareto for spacing={spacing}...")
        df_sub = df[df["spacing"] == spacing]
        path = stage._plot_pareto_frontier(
            df_sub, f"Spacing: {spacing}",
            pareto_dir / f"pareto_{spacing}", fig_config
        )
        if path:
            generated.append(path)

    # Per-pulse
    for pulse in df["pulse"].unique():
        if verbose:
            print(f"  Generating Pareto for pulse={pulse}...")
        df_sub = df[df["pulse"] == pulse]
        path = stage._plot_pareto_frontier(
            df_sub, f"Sequence: {pulse.upper()}",
            pareto_dir / f"pareto_{pulse}", fig_config
        )
        if path:
            generated.append(path)

    # Method comparison
    if verbose:
        print("  Generating method comparison...")
    paths = stage._plot_method_comparison(df, pareto_dir, fig_config)
    generated.extend(paths)

    # Faceted grid
    if verbose:
        print("  Generating faceted Pareto grid...")
    path = stage._plot_faceted_pareto(df, pareto_dir, fig_config)
    if path:
        generated.append(path)

    return generated


def generate_metrics_figures(
    experiment_dir: Path,
    output_dir: Path,
    fig_config: dict,
    verbose: bool = False,
) -> List[Path]:
    """Generate standard metrics figures (boxplots, tables)."""
    import json

    metrics_path = experiment_dir / "analysis" / "metrics_aggregated.json"
    if not metrics_path.exists():
        print(f"Warning: Aggregated metrics not found: {metrics_path}")
        return []

    with open(metrics_path, "r") as f:
        aggregated = json.load(f)

    from pacs_sr.pipeline.stages.visualization import VisualizationStage

    class MinimalContext:
        def __init__(self, exp_dir, out_dir):
            self.analysis_dir = exp_dir / "analysis"
            self.figures_dir = out_dir
            self.config = None

    ctx = MinimalContext(experiment_dir, output_dir)
    stage = VisualizationStage()

    generated = []

    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    if verbose:
        print("  Generating metrics table...")
    path = stage._generate_metrics_table(ctx, aggregated, fig_config)
    if path:
        generated.append(path)

    if verbose:
        print("  Generating boxplots...")
    paths = stage._generate_boxplots(ctx, aggregated, fig_config)
    generated.extend(paths)

    if verbose:
        print("  Generating fold comparison...")
    paths = stage._generate_fold_comparison(ctx, aggregated, fig_config)
    generated.extend(paths)

    return generated


def generate_weight_figures(
    experiment_dir: Path,
    output_dir: Path,
    fig_config: dict,
    config=None,
    verbose: bool = False,
) -> List[Path]:
    """Generate weight heatmap figures."""
    from pacs_sr.pipeline.stages.visualization import VisualizationStage

    class MinimalContext:
        def __init__(self, exp_dir, out_dir, cfg):
            self.training_dir = exp_dir / "training"
            self.figures_dir = out_dir
            self.config = cfg

        def log(self, msg, level="info"):
            pass

    ctx = MinimalContext(experiment_dir, output_dir, config)
    stage = VisualizationStage()

    (output_dir / "weights").mkdir(parents=True, exist_ok=True)

    if verbose:
        print("  Generating weight heatmaps...")

    return stage._generate_weight_heatmaps(ctx, fig_config)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.experiment_dir.exists():
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return 1

    print(f"Loading from: {args.experiment_dir}")

    # Load config
    config = load_config(args.experiment_dir, args.config)

    # Determine output directory
    output_dir = args.output_dir or (args.experiment_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Figure configuration
    fig_config = {
        "dpi": args.dpi,
        "formats": args.formats,
    }

    # Determine which figures to generate
    figure_types = args.only or ["metrics", "weights", "pareto", "specialization", "folds"]

    all_generated = []

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("Error: matplotlib not available")
        return 1

    # Generate requested figures
    if "pareto" in figure_types:
        print("\nGenerating Pareto frontier figures...")
        paths = generate_pareto_figures(args.experiment_dir, output_dir, fig_config, args.verbose)
        all_generated.extend(paths)
        print(f"  Generated {len(paths)} Pareto figures")

    if "metrics" in figure_types or "folds" in figure_types:
        print("\nGenerating metrics figures...")
        paths = generate_metrics_figures(args.experiment_dir, output_dir, fig_config, args.verbose)
        all_generated.extend(paths)
        print(f"  Generated {len(paths)} metrics figures")

    if "weights" in figure_types or "specialization" in figure_types:
        print("\nGenerating weight figures...")
        paths = generate_weight_figures(args.experiment_dir, output_dir, fig_config, config, args.verbose)
        all_generated.extend(paths)
        print(f"  Generated {len(paths)} weight figures")

    print(f"\nTotal figures generated: {len(all_generated)}")
    print(f"Output location: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
