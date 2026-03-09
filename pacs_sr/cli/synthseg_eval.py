"""CLI entry point for SynthSeg-based brain segmentation evaluation.

Usage:
    # Export HDF5 volumes to NIfTI
    pacs-sr-synthseg-eval --config configs/config.yaml export

    # Run SynthSeg on exported NIfTIs
    pacs-sr-synthseg-eval --config configs/config.yaml segment

    # Compute metrics and statistical tests
    pacs-sr-synthseg-eval --config configs/config.yaml analyze

    # Full pipeline (export → segment → analyze)
    pacs-sr-synthseg-eval --config configs/config.yaml all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pacs_sr.experiments.synthseg_evaluation import (
    load_synthseg_eval_config,
    run_analyze,
    run_export,
    run_full_pipeline,
    run_segment,
)


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger with rich-style formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="SynthSeg brain segmentation evaluation for PaCS-SR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages:\n"
            "  export   Export HDF5 volumes to NIfTI with real affines\n"
            "  segment  Run SynthSeg (subprocess-isolated TF env)\n"
            "  analyze  Compute Dice, volume errors, QC, statistical tests\n"
            "  all      Run complete pipeline (export → segment → analyze)\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "stage",
        choices=["export", "segment", "analyze", "all"],
        help="Pipeline stage to run",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s", args.config)

    config = load_synthseg_eval_config(args.config)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Methods: %s", config.methods)
    logger.info("Spacings: %s", config.spacings)
    logger.info("Pulse: %s", config.pulse)

    if args.stage == "export":
        exported = run_export(config)
        total = sum(len(v) for v in exported.values())
        logger.info(
            "Export complete: %d NIfTI files across %d groups", total, len(exported)
        )

    elif args.stage == "segment":
        status = run_segment(config)
        n_ok = sum(1 for v in status.values() if v)
        logger.info("Segment complete: %d/%d successful", n_ok, len(status))
        if n_ok < len(status):
            failed = [k for k, v in status.items() if not v]
            logger.warning("Failed runs: %s", failed)

    elif args.stage == "analyze":
        results = run_analyze(config)
        logger.info("Analysis complete. Outputs saved to %s", config.output_dir)

    elif args.stage == "all":
        results = run_full_pipeline(config)
        logger.info("Full pipeline complete. Outputs saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
