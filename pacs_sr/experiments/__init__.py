"""
PaCS-SR Scientific Experiments Module
=====================================

This module contains experiments for validating PaCS-SR's scientific claims:

1. **Regional Specialization Analysis** (regional_specialization.py)
   - Analyzes which SR experts excel in which brain regions
   - Computes per-ROI weight statistics
   - Generates heatmaps showing expert-region correlations

2. **Clinical Validation** (clinical_validation.py)
   - Computes metrics within clinical ROIs (tumor core, edema, surround)
   - Enables ROI-specific performance comparison

3. **Cross-Resolution Generalization** (cross_resolution.py)
   - Tests if weights learned at one resolution transfer to others
   - Quantifies generalization ability across different anisotropy levels

Usage:
    # Regional specialization
    python -m pacs_sr.experiments.regional_specialization --npz path/to/weights.npz

    # Clinical validation
    python -m pacs_sr.experiments.clinical_validation --results-dir ./results

    # Cross-resolution generalization
    python -m pacs_sr.experiments.cross_resolution --config configs/config.yaml
"""

from .regional_specialization import (
    analyze_regional_weights,
    compute_specialization_index,
    analyze_all_weights,
)
from .clinical_validation import (
    compute_roi_metrics,
    run_clinical_validation_demo,
)
from .cross_resolution import (
    train_and_transfer,
    compute_generalization_gap,
)

__all__ = [
    # Regional specialization
    "analyze_regional_weights",
    "compute_specialization_index",
    "analyze_all_weights",
    # Clinical validation
    "compute_roi_metrics",
    "run_clinical_validation_demo",
    # Cross-resolution
    "train_and_transfer",
    "compute_generalization_gap",
]
