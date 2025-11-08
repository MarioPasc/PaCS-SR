#!/usr/bin/env python3
"""
compute_pacs_sr_gain.py

Compute how much better the PACS_SR model is compared to other models
for each sequence, each spacing, and aggregate means per sequence and per spacing.

Inputs:
- metrics_by_case.csv: per-case metrics with columns:
    model, spacing, sequence, case_id, psnr, ssim, pred_path, gt_path
- metrics_summary.csv: pre-aggregated metrics with columns:
    model, spacing, sequence, psnr_mean, psnr_std, ssim_mean, ssim_std, n

Outputs (CSV):
- pacs_sr_delta_by_seq_spacing.csv
- pacs_sr_delta_means.csv
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import pandas as pd


# ------------------------- Configuration -------------------------

@dataclass(frozen=True)
class Config:
    """Configuration for file paths and key identifiers."""
    by_case_csv: Path
    summary_csv: Path
    target_model: str = "PACS_SR"   # Name of the model to compare against others
    out_dir: Path = Path(".")       # Output directory


# ------------------------- Utilities -------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        format="%(levelname)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def assert_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    """Raise ValueError if the DataFrame lacks required columns.
    
    Args:
        df: Dataframe to validate.
        required: List of required column names.
        name: Logical name for error messages.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# --------------------- Core computation logic ---------------------

def compute_means_from_by_case(by_case: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(model, sequence, spacing) means and counts from per-case records.
    
    Args:
        by_case: DataFrame with columns: model, spacing, sequence, case_id, psnr, ssim
    
    Returns:
        DataFrame with columns: model, spacing, sequence, psnr_mean, ssim_mean, n
    """
    assert_required_columns(
        by_case,
        ["model", "spacing", "sequence", "psnr", "ssim"],
        "metrics_by_case.csv",
    )
    grp = (
        by_case
        .groupby(["model", "sequence", "spacing"], as_index=False)
        .agg(psnr_mean=("psnr", "mean"),
             ssim_mean=("ssim", "mean"),
             n=("case_id", "nunique"))
    )
    return grp


def pick_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize provided summary to the expected columns.
    
    Args:
        summary: DataFrame with columns including psnr_mean and ssim_mean.
        
    Returns:
        Summary subset with columns: model, sequence, spacing, psnr_mean, ssim_mean, n
    """
    assert_required_columns(
        summary,
        ["model", "spacing", "sequence", "psnr_mean", "ssim_mean", "n"],
        "metrics_summary.csv",
    )
    cols = ["model", "sequence", "spacing", "psnr_mean", "ssim_mean", "n"]
    return summary[cols].copy()


def _pair_with_target(target_rows: pd.DataFrame, other_rows: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Join target model rows with each other model's rows on (sequence, spacing).
    
    Args:
        target_rows: Subset for target model with mean metrics.
        other_rows: Subset for non-target models with mean metrics.
        metric_cols: Metric base names like ["psnr", "ssim"].
        
    Returns:
        DataFrame with per-(sequence, spacing, other_model) deltas.
    """
    merged = other_rows.merge(
        target_rows,
        on=["sequence", "spacing"],
        suffixes=("_other", "_target"),
        how="inner",
    )
    # Compute deltas: target - other for higher-is-better metrics.
    for m in metric_cols:
        merged[f"delta_{m}"] = merged[f"{m}_mean_target"] - merged[f"{m}_mean_other"]
    # Rename columns for clarity
    merged = merged.rename(columns={"model_other": "other_model"})
    # Reorder columns
    base_cols = ["sequence", "spacing", "other_model"]
    stats_cols = []
    for m in metric_cols:
        stats_cols += [
            f"{m}_mean_target", f"{m}_mean_other", f"delta_{m}"
        ]
    if "n_target" in merged.columns and "n_other" in merged.columns:
        count_cols = ["n_target", "n_other"]
    else:
        count_cols = []
    ordered = base_cols + stats_cols + count_cols
    keep_cols = [c for c in ordered if c in merged.columns]
    return merged[keep_cols]


def compare_against_target(means: pd.DataFrame, target_model: str = "PACS_SR") -> pd.DataFrame:
    """Compute per-(sequence, spacing) deltas of target vs every other model.
    
    Args:
        means: DataFrame with columns model, sequence, spacing, psnr_mean, ssim_mean, n
        target_model: Name of the reference model
        
    Returns:
        DataFrame with columns sequence, spacing, other_model,
        psnr_mean_target, psnr_mean_other, delta_psnr,
        ssim_mean_target, ssim_mean_other, delta_ssim, n_target, n_other
    """
    metric_cols = ["psnr", "ssim"]
    needed = ["model", "sequence", "spacing", "psnr_mean", "ssim_mean", "n"]
    assert_required_columns(means, needed, "means")
    target_rows = means.loc[means["model"] == target_model].copy()
    other_rows = means.loc[means["model"] != target_model].copy()
    # Keep counts for transparency
    target_rows = target_rows.rename(columns={"n": "n_target"})
    other_rows = other_rows.rename(columns={"n": "n_other", "model": "model_other"})
    return _pair_with_target(target_rows, other_rows, metric_cols)


def aggregate_means(deltas: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deltas across spacings (per-sequence), across sequences (per-spacing), and overall.
    
    Args:
        deltas: Output of compare_against_target
        
    Returns:
        DataFrame with rows for three aggregation levels:
        - level = 'per_sequence': one row per (sequence, other_model)
        - level = 'per_spacing' : one row per (spacing, other_model)
        - level = 'overall'     : one row per other_model
    """
    assert_required_columns(
        deltas,
        ["sequence", "spacing", "other_model", "delta_psnr", "delta_ssim"],
        "deltas",
    )
    records = []
    # Per-sequence over all spacings
    g = deltas.groupby(["sequence", "other_model"], as_index=False)[["delta_psnr", "delta_ssim"]].mean()
    g.insert(0, "level", "per_sequence")
    g["spacing"] = pd.NA
    records.append(g)

    # Per-spacing over all sequences
    g = deltas.groupby(["spacing", "other_model"], as_index=False)[["delta_psnr", "delta_ssim"]].mean()
    g.insert(0, "level", "per_spacing")
    g["sequence"] = pd.NA
    records.append(g)

    # Overall across all
    g = deltas.groupby(["other_model"], as_index=False)[["delta_psnr", "delta_ssim"]].mean()
    g.insert(0, "level", "overall")
    g["sequence"] = pd.NA
    g["spacing"] = pd.NA
    records.append(g)

    out = pd.concat(records, ignore_index=True)
    # Column order
    cols = ["level", "sequence", "spacing", "other_model", "delta_psnr", "delta_ssim"]
    return out[cols]


# --------------------------- Main CLI ---------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Compute PACS_SR gains vs other models.")
    p.add_argument("--by_case_csv", type=Path, required=True, help="Path to metrics_by_case.csv")
    p.add_argument("--summary_csv", type=Path, required=True, help="Path to metrics_summary.csv")
    p.add_argument("--out_dir", type=Path, default=Path("."), help="Output directory for CSVs")
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    p.add_argument("--target_model", type=str, default="PACS_SR", help="Model to use as reference")
    return p.parse_args()


def main() -> None:
    """Entrypoint for the script."""
    args = parse_args()
    setup_logging(args.log_level)
    cfg = Config(
        by_case_csv=args.by_case_csv,
        summary_csv=args.summary_csv,
        target_model=args.target_model,
        out_dir=args.out_dir,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading CSVs")
    by_case = pd.read_csv(cfg.by_case_csv)
    summary = pd.read_csv(cfg.summary_csv)

    # Harmonize or rebuild summary means
    logging.info("Preparing mean metrics")
    # Prefer the provided summary, but recompute from by_case as a cross-check
    summary_means = pick_summary(summary)
    by_case_means = compute_means_from_by_case(by_case)

    # Compare target vs others using provided summary
    logging.info("Computing deltas from provided summary")
    deltas = compare_against_target(summary_means, target_model=cfg.target_model)

    # Also compute deltas from by_case-derived means for validation
    logging.info("Computing deltas from by-case derived means (sanity check)")
    deltas_from_cases = compare_against_target(by_case_means, target_model=cfg.target_model)

    # Save detailed deltas
    out_deltas_path = cfg.out_dir / "pacs_sr_delta_by_seq_spacing.csv"
    out_deltas_cases_path = cfg.out_dir / "pacs_sr_delta_by_seq_spacing_from_cases.csv"
    deltas.to_csv(out_deltas_path, index=False)
    deltas_from_cases.to_csv(out_deltas_cases_path, index=False)

    # Aggregate means
    logging.info("Aggregating means per sequence, per spacing, and overall")
    agg = aggregate_means(deltas)
    agg_cases = aggregate_means(deltas_from_cases)

    out_means_path = cfg.out_dir / "pacs_sr_delta_means.csv"
    out_means_cases_path = cfg.out_dir / "pacs_sr_delta_means_from_cases.csv"
    agg.to_csv(out_means_path, index=False)
    agg_cases.to_csv(out_means_cases_path, index=False)

    logging.info("Done")
    logging.info(f"Wrote: {out_deltas_path}")
    logging.info(f"Wrote: {out_deltas_cases_path}")
    logging.info(f"Wrote: {out_means_path}")
    logging.info(f"Wrote: {out_means_cases_path}")


if __name__ == "__main__":
    main()
