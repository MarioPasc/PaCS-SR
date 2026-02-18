#!/usr/bin/env python3
"""SERAM Table 1: 3D-MS-SSIM aggregated across sequences.

Produces a table with mean +/- std of 3D-MS-SSIM per model x spacing,
aggregated across all sequences. Output as CSV and LaTeX.

Layout:
         BSPLINE       ECLARE        PaCS-SR
3mm     0.95+-0.02    0.97+-0.01    0.98+-0.01
5mm     0.91+-0.03    0.94+-0.02    0.96+-0.02
7mm     0.87+-0.04    0.90+-0.03    0.93+-0.03

Usage:
    python -m pacs_sr.seram.table_msssim --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

LOG = logging.getLogger(__name__)

MODEL_ORDER = ["BSPLINE", "ECLARE", "PaCS-SR"]


def generate_msssim_table(
    metrics_csv: Path,
    spacings: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Generate 3D-MS-SSIM table aggregated across sequences.

    Args:
        metrics_csv: Path to seram_metrics.csv.
        spacings: List of spacings for row ordering.
        out_dir: Output directory for CSV and LaTeX files.

    Returns:
        DataFrame with the formatted table.
    """
    df = pd.read_csv(metrics_csv)
    df["model"] = df["model"].replace({"PACS_SR": "PaCS-SR", "PaCS_SR": "PaCS-SR"})

    rows = []
    for spacing in spacings:
        row = {"Spacing": spacing}
        for model in MODEL_ORDER:
            sub = df[(df["spacing"] == spacing) & (df["model"] == model)]
            vals = sub["ms_ssim_3d"].dropna()
            if len(vals) > 0:
                mean = vals.mean()
                std = vals.std()
                row[model] = f"{mean:.4f} $\\pm$ {std:.4f}"
            else:
                row[model] = "---"
        rows.append(row)

    table_df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV (plain numbers)
    csv_rows = []
    for spacing in spacings:
        csv_row = {"spacing": spacing}
        for model in MODEL_ORDER:
            sub = df[(df["spacing"] == spacing) & (df["model"] == model)]
            vals = sub["ms_ssim_3d"].dropna()
            csv_row[f"{model}_mean"] = vals.mean() if len(vals) > 0 else float("nan")
            csv_row[f"{model}_std"] = vals.std() if len(vals) > 0 else float("nan")
        csv_rows.append(csv_row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "table1_msssim.csv"
    csv_df.to_csv(csv_path, index=False)
    LOG.info("Saved CSV: %s", csv_path)

    # Save LaTeX
    latex_path = out_dir / "table1_msssim.tex"
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{3D Multi-Scale SSIM (mean $\pm$ std) aggregated across T1C, T2W, and T2F sequences.}",
        r"\label{tab:msssim}",
        r"\begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}",
        r"\toprule",
        r"Spacing & " + " & ".join(MODEL_ORDER) + r" \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        vals = [str(row[m]) for m in MODEL_ORDER]
        latex_lines.append(f"{row['Spacing']} & " + " & ".join(vals) + r" \\")
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    latex_path.write_text("\n".join(latex_lines))
    LOG.info("Saved LaTeX: %s", latex_path)

    return table_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SERAM Table 1: 3D-MS-SSIM table")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--metrics-csv", type=Path, default=None, help="Override metrics CSV path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s | %(message)s"
    )

    from pacs_sr.config.config import load_full_config

    full = load_full_config(args.config)
    pacs_sr = full.pacs_sr

    csv_path = args.metrics_csv or (
        Path(pacs_sr.out_root) / pacs_sr.experiment_name / "seram_metrics.csv"
    )
    out_dir = Path(pacs_sr.out_root) / pacs_sr.experiment_name / "figures"

    table = generate_msssim_table(
        metrics_csv=csv_path,
        spacings=list(pacs_sr.spacings),
        out_dir=out_dir,
    )
    print("\n3D-MS-SSIM Table:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
