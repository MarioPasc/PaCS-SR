#!/usr/bin/env python
"""
CLI entry point for metrics computation and statistical analysis.

This script computes volumetric metrics (PSNR, SSIM, MAE, etc.) for predictions,
performs statistical tests, and optionally generates visualization plots.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pacs_sr.analysis.analyze_metrics import run_analysis


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute metrics and perform statistical analysis on SR predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run metrics analysis only
  pacs-sr-analyze-metrics --config configs/analysis_config.yaml

  # Run analysis and generate plots
  pacs-sr-analyze-metrics --config configs/analysis_config.yaml --visualize

  # Run analysis and generate plots with custom output
  pacs-sr-analyze-metrics --config configs/analysis_config.yaml \\
      --visualize --plot-out results/figures

This will:
  1. Load GT and prediction volumes
  2. Compute metrics (PSNR, SSIM, MAE, RMSE, NCC) per case/sequence/method
  3. Perform paired statistical tests with multiple-comparison correction
  4. Write results as CSV/JSON tables
  5. Optionally generate visualization plots (if --visualize is specified)
"""
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file for analysis"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots after computing metrics"
    )
    parser.add_argument(
        "--stats-npz",
        type=Path,
        default=None,
        help="Path to stats NPZ file for visualization (if different from default)"
    )
    parser.add_argument(
        "--metrics-npz",
        type=Path,
        default=None,
        help="Path to metrics NPZ file for visualization (optional)"
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to <output_dir>/figures)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    """Main entry point for metrics analysis CLI."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("MetricsAnalysis")
    
    # Validate config file exists
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"Starting metrics analysis with config: {args.config}")
    
    # Run metrics computation and statistical analysis
    try:
        run_analysis(args.config)
        logger.info("✓ Metrics analysis completed successfully")
    except Exception as e:
        logger.error(f"✗ Metrics analysis failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Optionally run visualization
    if args.visualize:
        logger.info("Starting visualization generation...")
        
        # Determine output directory from config
        from pacs_sr.config.config import parse_analysis_config
        cfg = parse_analysis_config(args.config)
        output_dir = cfg.io.output_dir
        
        # Determine paths for stats visualization
        stats_npz = args.stats_npz
        if stats_npz is None:
            # Try to find stats file in output directory
            # This assumes the stats file follows a naming convention
            # Adjust this based on your actual output structure
            potential_stats = list(output_dir.glob("*stats*.npz"))
            if potential_stats:
                stats_npz = potential_stats[0]
                logger.info(f"Found stats file: {stats_npz}")
            else:
                logger.warning("No stats NPZ file found. Skipping visualization.")
                logger.warning("Use --stats-npz to specify the stats file explicitly.")
                return
        
        metrics_npz = args.metrics_npz
        plot_out = args.plot_out or (output_dir / "figures")
        plot_out.mkdir(parents=True, exist_ok=True)
        
        # Build command for stats visualization
        cmd = [
            sys.executable,
            "-m",
            "pacs_sr.utils.stats",
            "--stats_npz", str(stats_npz),
            "--out_dir", str(plot_out)
        ]
        
        if metrics_npz is not None:
            cmd.extend(["--metrics_npz", str(metrics_npz)])
        
        logger.info(f"Running visualization: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("✓ Visualization completed successfully")
            if result.stdout:
                logger.debug(f"Visualization output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Visualization failed: {e}")
            if e.stderr:
                logger.error(f"Error output:\n{e.stderr}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"✗ Visualization failed: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info("All tasks completed successfully!")


if __name__ == "__main__":
    main()
