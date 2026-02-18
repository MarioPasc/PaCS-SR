"""Publication-ready plot settings with IEEE-compliant typography.

IEEE-compliant settings with Paul Tol colorblind-friendly palettes.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - IEEE publication guidelines
    - scienceplots: https://github.com/garrettj403/SciencePlots
"""

from __future__ import annotations

from typing import Dict, Tuple

# =============================================================================
# Paul Tol Color Palettes (SRON - colorblind safe)
# =============================================================================

PAUL_TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

# =============================================================================
# IEEE Column Width Specifications
# =============================================================================

IEEE_COLUMN_WIDTH_INCHES = 3.39  # Single column (86 mm)
IEEE_COLUMN_GAP_INCHES = 0.24  # Gap between columns (6 mm)
IEEE_TEXT_WIDTH_INCHES = 7.0  # Full print area width (178 mm)
IEEE_TEXT_HEIGHT_INCHES = 9.0  # Full print area height (229 mm)

# =============================================================================
# Main Plot Settings Dictionary
# =============================================================================

PLOT_SETTINGS = {
    # Figure dimensions (IEEE compliant)
    "figure_width_single": IEEE_COLUMN_WIDTH_INCHES,  # 3.39 inches
    "figure_width_double": IEEE_TEXT_WIDTH_INCHES,  # 7.0 inches
    "figure_height_max": IEEE_TEXT_HEIGHT_INCHES,  # 9.0 inches (max)
    "figure_height_ratio": 0.75,  # Height = width * ratio (for plots)
    # Fonts (IEEE requires Times or similar serif)
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext_fontset": "stix",  # STIX for math (matches Times)
    "text_usetex": False,  # Set True if LaTeX is installed
    # Font sizes (IEEE guidelines)
    "font_size": 10,
    "axes_labelsize": 11,
    "axes_titlesize": 12,
    "tick_labelsize": 9,
    "legend_fontsize": 9,
    "annotation_fontsize": 8,
    "panel_label_fontsize": 11,
    # Line properties
    "line_width": 1.2,
    "line_width_thick": 1.8,
    "marker_size": 5,
    "marker_edge_width": 0.5,
    # Error bars
    "errorbar_capsize": 2,
    "errorbar_capthick": 0.8,
    "errorbar_linewidth": 0.8,
    # Error bands (for confidence intervals)
    "error_band_alpha": 0.2,
    # Boxplot properties
    "boxplot_linewidth": 0.8,
    "boxplot_flier_size": 3,
    "boxplot_width": 0.6,
    # Bar plot properties
    "bar_width": 0.18,
    "bar_alpha": 0.85,
    # Grid
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    # Spines
    "spine_linewidth": 0.8,
    "spine_color": "0.2",
    # Ticks
    "tick_direction": "in",
    "tick_major_width": 0.8,
    "tick_minor_width": 0.5,
    "tick_major_length": 3.5,
    "tick_minor_length": 2.0,
    # Legend
    "legend_frameon": False,
    "legend_framealpha": 0.9,
    "legend_edgecolor": "0.8",
    "legend_borderpad": 0.4,
    "legend_columnspacing": 1.0,
    "legend_handletextpad": 0.5,
    # UMAP/t-SNE scatter
    "scatter_alpha": 0.6,
    "scatter_size": 15,
    "scatter_edgewidth": 0.3,
    # DPI for output
    "dpi_print": 300,
    "dpi_screen": 150,
    # Significance annotations
    "significance_bracket_linewidth": 0.8,
    "significance_text_fontsize": 9,
    "effect_size_fontsize": 8,
}

# =============================================================================
# Condition Colors (for consistent styling across figures)
# =============================================================================

CONDITION_COLORS: Dict[str, str] = {
    "real_data": PAUL_TOL_BRIGHT["blue"],
    "synthetic_data": PAUL_TOL_BRIGHT["red"],
    "control": PAUL_TOL_BRIGHT["green"],
    "epilepsy": PAUL_TOL_BRIGHT["purple"],
    "vMF": PAUL_TOL_BRIGHT["cyan"],
    "baseline": PAUL_TOL_BRIGHT["yellow"],
    "threshold": PAUL_TOL_BRIGHT["grey"],
}

# =============================================================================
# SERAM / PaCS-SR Model Colors
# =============================================================================

SERAM_MODEL_COLORS: Dict[str, str] = {
    "BSPLINE": PAUL_TOL_BRIGHT["grey"],
    "ECLARE": PAUL_TOL_BRIGHT["blue"],
    "PaCS-SR": PAUL_TOL_BRIGHT["red"],
    "PACS_SR": PAUL_TOL_BRIGHT["red"],
    "SMORE": PAUL_TOL_BRIGHT["green"],
    "UNIRES": PAUL_TOL_BRIGHT["yellow"],
}


def apply_ieee_style() -> None:
    """Apply IEEE publication style using scienceplots if available.

    Falls back to manual style settings if scienceplots is not installed.
    Overrides default color cycle with Paul Tol colorblind-safe palette.
    """
    import matplotlib.pyplot as plt

    # Try to use scienceplots if available
    try:
        plt.style.use(["science", "ieee"])
        _scienceplots_available = True
    except OSError:
        _scienceplots_available = False
        _apply_fallback_ieee_style()

    # Override with condition colors and custom settings
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=list(CONDITION_COLORS.values())),
            # Ensure math rendering
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            "font.family": PLOT_SETTINGS["font_family"],
            # Grid settings
            "axes.grid": True,
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Tick settings
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        }
    )


def _apply_fallback_ieee_style() -> None:
    """Apply IEEE-like style without scienceplots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Fonts
            "font.family": PLOT_SETTINGS["font_family"],
            "font.serif": PLOT_SETTINGS["font_serif"],
            "font.size": PLOT_SETTINGS["font_size"],
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            # Axes
            "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
            "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
            "axes.linewidth": PLOT_SETTINGS["spine_linewidth"],
            "axes.grid": True,
            # Ticks
            "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
            "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.size": PLOT_SETTINGS["tick_major_length"],
            "ytick.major.size": PLOT_SETTINGS["tick_major_length"],
            "xtick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            "ytick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            # Legend
            "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
            "legend.frameon": PLOT_SETTINGS["legend_frameon"],
            "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
            "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
            # Grid
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Figure
            "figure.figsize": (
                PLOT_SETTINGS["figure_width_double"],
                PLOT_SETTINGS["figure_width_double"]
                * PLOT_SETTINGS["figure_height_ratio"],
            ),
            "figure.dpi": PLOT_SETTINGS["dpi_screen"],
            "savefig.dpi": PLOT_SETTINGS["dpi_print"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def get_significance_stars(p_val: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p_val: P-value from statistical test.

    Returns:
        String with stars: "***" (p<0.001), "**" (p<0.01),
        "*" (p<0.05), or "n.s." (not significant).
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def get_effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def get_figure_size(
    width: str = "single",
    height_ratio: float = None,
) -> Tuple[float, float]:
    """Get figure size tuple for IEEE format.

    Args:
        width: "single" for column width, "double" for full width.
        height_ratio: Custom height/width ratio. If None, uses default.

    Returns:
        Tuple of (width, height) in inches.
    """
    if width == "single":
        w = PLOT_SETTINGS["figure_width_single"]
    elif width == "double":
        w = PLOT_SETTINGS["figure_width_double"]
    else:
        raise ValueError(f"Unknown width: {width}")

    ratio = height_ratio or PLOT_SETTINGS["figure_height_ratio"]
    return (w, w * ratio)
