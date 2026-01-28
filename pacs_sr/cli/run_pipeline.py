#!/usr/bin/env python3
"""
PaCS-SR Pipeline Runner
=======================

End-to-end pipeline execution with a single command.
Handles training, validation, testing, analysis, and visualization.

Usage:
    pacs-sr-run --config configs/pipeline_config.yaml
    pacs-sr-run --config configs/pipeline_config.yaml --resume
    pacs-sr-run --config configs/pipeline_config.yaml --folds 1 2 --spacings 3mm
    pacs-sr-run --config configs/pipeline_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaCS-SR End-to-End Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  pacs-sr-run --config configs/pipeline_config.yaml

  # Resume from checkpoint
  pacs-sr-run --config configs/pipeline_config.yaml --resume

  # Run specific folds and spacings
  pacs-sr-run --config configs/pipeline_config.yaml --folds 1 2 --spacings 3mm 5mm

  # Dry run (validate only)
  pacs-sr-run --config configs/pipeline_config.yaml --dry-run
        """
    )

    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )

    # Optional arguments
    parser.add_argument(
        "--output-root", "-o",
        type=Path,
        default=None,
        help="Override output root directory"
    )
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        default=None,
        help="Override experiment name"
    )

    # Scope selection
    parser.add_argument(
        "--folds", "-f",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to run (default: all)"
    )
    parser.add_argument(
        "--spacings", "-s",
        type=str,
        nargs="+",
        default=None,
        help="Specific spacings to run (default: all)"
    )
    parser.add_argument(
        "--pulses", "-p",
        type=str,
        nargs="+",
        default=None,
        help="Specific pulses to run (default: all)"
    )

    # Execution control
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Force fresh start, ignore existing checkpoints"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        default=False,
        help="Validate config and show execution plan without running"
    )

    # Directory naming
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=False,
        help="Don't append timestamp to experiment directory"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress non-essential output"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the pipeline CLI."""
    args = parse_args()

    # Validate config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1

    # Import here to avoid slow startup for --help
    from pacs_sr.pipeline import PipelineOrchestrator

    # Determine resume behavior
    resume = args.resume and not args.no_resume

    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            config_path=args.config,
            output_root=args.output_root,
            experiment_name=args.experiment_name,
            timestamp_suffix=not args.no_timestamp,
            resume=resume,
            folds=args.folds,
            spacings=args.spacings,
            pulses=args.pulses,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        # Run pipeline
        success = orchestrator.run()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        print("Run with --resume to continue from checkpoint")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
