#!/usr/bin/env python3
"""SERAM Figure 2: Per-sequence MMD-MF performance plot.

Generates a 1x3 subplot (per-sequence) showing MMD-MF metric
(mean +/- std across patients) per spacing.

Layout:
     T1C              T2W              T2F
  [MMD-MF vs spacing] [MMD-MF vs spacing] [MMD-MF vs spacing]

Usage:
    python -m pacs_sr.seram.figure_performance_plot --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pacs_sr.utils.settings import apply_ieee_style, PLOT_SETTINGS, SERAM_MODEL_COLORS


LOG = logging.getLogger(__name__)

# Consistent model display order
MODEL_ORDER = ["BSPLINE", "ECLARE", "PaCS-SR"]


def generate_performance_plot(
    metrics_csv: Path,
    pulses: List[str],
    spacings: List[str],
    out_path: Path,
    show_xlabel: bool = True,
) -> None:
    """Generate the 1x3 MMD-MF performance subplot.

    Args:
        metrics_csv: Path to seram_metrics.csv with columns
            (patient, spacing, pulse, model, ms_ssim_3d, mmd_mf).
        pulses: List of pulse sequences to plot.
        spacings: List of spacings for x-axis.
        out_path: Output file path.
        show_xlabel: If False, suppress the "Through-plane spacing" x-axis label.
    """
    apply_ieee_style()

    df = pd.read_csv(metrics_csv)
    palette = SERAM_MODEL_COLORS

    # Map PaCS_SR or PACS_SR to PaCS-SR for display
    df["model"] = df["model"].replace({"PACS_SR": "PaCS-SR", "PaCS_SR": "PaCS-SR"})

    n_pulses = len(pulses)
    w, h = (
        PLOT_SETTINGS["figure_width_double"],
        PLOT_SETTINGS["figure_width_double"] * 0.4,
    )
    fig, axes = plt.subplots(1, n_pulses, figsize=(w, h), constrained_layout=True)
    if n_pulses == 1:
        axes = [axes]

    x_pos = np.arange(len(spacings))

    for ax_idx, pulse in enumerate(pulses):
        ax = axes[ax_idx]
        sub = df[df["pulse"] == pulse]

        for m_idx, model in enumerate(MODEL_ORDER):
            model_sub = sub[sub["model"] == model]
            if model_sub.empty:
                continue

            means = []
            stds = []
            for sp in spacings:
                sp_sub = model_sub[model_sub["spacing"] == sp]["mmd_mf"]
                means.append(sp_sub.mean() if len(sp_sub) > 0 else np.nan)
                stds.append(sp_sub.std() if len(sp_sub) > 1 else 0.0)

            offset = (m_idx - 1) * 0.15
            color = palette.get(model, "#555555")
            ax.errorbar(
                x_pos + offset,
                means,
                yerr=stds,
                fmt="o-",
                color=color,
                lw=PLOT_SETTINGS["line_width"],
                ms=PLOT_SETTINGS["marker_size"],
                capsize=PLOT_SETTINGS["errorbar_capsize"],
                capthick=PLOT_SETTINGS["errorbar_capthick"],
                label=model,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(spacings)
        if show_xlabel:
            ax.set_xlabel("Through-plane spacing")
        ax.set_title(pulse.upper(), fontsize=PLOT_SETTINGS["axes_titlesize"])

        if ax_idx == 0:
            ax.set_ylabel(r"MMD-MF ($\downarrow$ better)")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(MODEL_ORDER),
            frameon=PLOT_SETTINGS["legend_frameon"],
            fontsize=PLOT_SETTINGS["legend_fontsize"],
            bbox_to_anchor=(0.5, -0.08),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        save_path = out_path.with_suffix(f".{ext}")
        fig.savefig(save_path, dpi=PLOT_SETTINGS["dpi_print"])
        LOG.info("Saved %s", save_path)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SERAM Figure 2: MMD-MF performance plot"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--metrics-csv", type=Path, default=None, help="Override metrics CSV path"
    )
    parser.add_argument(
        "--no-xlabel",
        action="store_true",
        default=False,
        help="Suppress the 'Through-plane spacing' x-axis label",
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

    generate_performance_plot(
        metrics_csv=csv_path,
        pulses=list(pacs_sr.pulses),
        spacings=list(pacs_sr.spacings),
        out_path=out_dir / "figure2_mmd_mf.pdf",
        show_xlabel=not args.no_xlabel,
    )


if __name__ == "__main__":
    main()
