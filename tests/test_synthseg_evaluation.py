"""Tests for SynthSeg brain segmentation evaluation module.

Uses synthetic data to validate:
- K-fold patient mapping
- Dice computation
- Volume error computation
- QC/volumes CSV parsing
- Statistical analysis pipeline
- Results assembly and DataFrame conversion
"""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pacs_sr.experiments.synthseg_evaluation import (
    SYNTHSEG_LABEL_MAP,
    SynthSegEvalConfig,
    _bootstrap_ci,
    build_patient_fold_map,
    collect_all_test_patients,
    compute_dice_per_region,
    compute_volume_errors,
    parse_qc_csv,
    parse_volumes_csv,
    results_to_dataframe,
    run_statistical_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest() -> dict:
    """Minimal 3-fold manifest with 6 patients."""
    return {
        "folds": [
            {"train": ["P1", "P2", "P3", "P4"], "test": ["P5", "P6"]},
            {"train": ["P1", "P2", "P5", "P6"], "test": ["P3", "P4"]},
            {"train": ["P3", "P4", "P5", "P6"], "test": ["P1", "P2"]},
        ]
    }


@pytest.fixture
def tmp_labels(tmp_path: Path) -> tuple[Path, Path]:
    """Create two synthetic SynthSeg label maps for Dice testing."""
    shape = (32, 32, 32)
    affine = np.eye(4)

    # Label map A: left hemisphere has labels, right is background
    labels_a = np.zeros(shape, dtype=np.int32)
    labels_a[:16, :, :] = 3  # Left-Cerebral-Cortex
    labels_a[16:, :16, :] = 17  # Left-Hippocampus
    labels_a[16:, 16:, :16] = 4  # Left-Lateral-Ventricle

    # Label map B: similar but shifted (partial overlap)
    labels_b = np.zeros(shape, dtype=np.int32)
    labels_b[:14, :, :] = 3  # Slightly smaller cortex
    labels_b[14:, :16, :] = 17  # Shifted hippocampus
    labels_b[16:, 16:, :16] = 4  # Same ventricle

    path_a = tmp_path / "labels_a.nii.gz"
    path_b = tmp_path / "labels_b.nii.gz"
    nib.save(nib.Nifti1Image(labels_a, affine), str(path_a))
    nib.save(nib.Nifti1Image(labels_b, affine), str(path_b))

    return path_a, path_b


@pytest.fixture
def tmp_qc_csv(tmp_path: Path) -> Path:
    """Create a synthetic QC CSV."""
    csv_path = tmp_path / "qc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "qc"])
        writer.writeheader()
        writer.writerow({"subject": "P1-t1c.nii.gz", "qc": "0.85"})
        writer.writerow({"subject": "P2-t1c.nii.gz", "qc": "0.92"})
        writer.writerow({"subject": "P3-t1c.nii.gz", "qc": "0.78"})
    return csv_path


@pytest.fixture
def tmp_volumes_csv(tmp_path: Path) -> Path:
    """Create a synthetic volumes CSV."""
    csv_path = tmp_path / "volumes.csv"
    fieldnames = [
        "subject",
        "Left-Hippocampus",
        "Right-Hippocampus",
        "Left-Cerebral-Cortex",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "subject": "P1-t1c.nii.gz",
                "Left-Hippocampus": "3200.5",
                "Right-Hippocampus": "3100.2",
                "Left-Cerebral-Cortex": "45000.0",
            }
        )
        writer.writerow(
            {
                "subject": "P2-t1c.nii.gz",
                "Left-Hippocampus": "3150.0",
                "Right-Hippocampus": "3050.8",
                "Left-Cerebral-Cortex": "44500.0",
            }
        )
    return csv_path


# ---------------------------------------------------------------------------
# K-fold tests
# ---------------------------------------------------------------------------


class TestKFoldUtilities:
    """Tests for K-fold patient mapping functions."""

    def test_build_patient_fold_map(self, manifest: dict) -> None:
        fold_map = build_patient_fold_map(manifest)
        assert fold_map["P5"] == 1
        assert fold_map["P6"] == 1
        assert fold_map["P3"] == 2
        assert fold_map["P4"] == 2
        assert fold_map["P1"] == 3
        assert fold_map["P2"] == 3

    def test_collect_all_test_patients(self, manifest: dict) -> None:
        patients = collect_all_test_patients(manifest)
        assert patients == ["P1", "P2", "P3", "P4", "P5", "P6"]

    def test_no_duplicate_patients(self, manifest: dict) -> None:
        patients = collect_all_test_patients(manifest)
        assert len(patients) == len(set(patients))

    def test_fold_map_covers_all_patients(self, manifest: dict) -> None:
        fold_map = build_patient_fold_map(manifest)
        patients = collect_all_test_patients(manifest)
        assert set(fold_map.keys()) == set(patients)


# ---------------------------------------------------------------------------
# Dice tests
# ---------------------------------------------------------------------------


class TestDiceComputation:
    """Tests for Dice coefficient computation."""

    def test_identical_labels(self, tmp_labels: tuple[Path, Path]) -> None:
        path_a, _ = tmp_labels
        dices = compute_dice_per_region(path_a, path_a)
        # Self-comparison should give Dice = 1.0 for all present regions
        for region, lid in SYNTHSEG_LABEL_MAP.items():
            if lid in (3, 17, 4):  # Labels present in our test data
                assert dices[region] == 1.0, f"Expected Dice=1.0 for {region}"

    def test_different_labels(self, tmp_labels: tuple[Path, Path]) -> None:
        path_a, path_b = tmp_labels
        dices = compute_dice_per_region(path_a, path_b)
        # Cortex (label 3) should have partial overlap
        assert 0.0 < dices["Left-Cerebral-Cortex"] < 1.0
        # Mean should exist
        assert "mean" in dices
        assert 0.0 <= dices["mean"] <= 1.0

    def test_empty_regions_dice_one(self, tmp_labels: tuple[Path, Path]) -> None:
        path_a, _ = tmp_labels
        dices = compute_dice_per_region(path_a, path_a)
        # Regions not present in labels should get Dice = 1.0 (both empty)
        assert dices.get("Right-Hippocampus", 1.0) == 1.0

    def test_custom_label_map(self, tmp_labels: tuple[Path, Path]) -> None:
        path_a, path_b = tmp_labels
        subset = {"Left-Cerebral-Cortex": 3}
        dices = compute_dice_per_region(path_a, path_b, label_map=subset)
        assert "Left-Cerebral-Cortex" in dices
        assert "mean" in dices
        assert len(dices) == 2  # 1 region + mean


# ---------------------------------------------------------------------------
# Volume error tests
# ---------------------------------------------------------------------------


class TestVolumeErrors:
    """Tests for volume error computation."""

    def test_identical_volumes(self) -> None:
        vols = {"Left-Hippocampus": 3200.0, "Left-Cerebral-Cortex": 45000.0}
        errors = compute_volume_errors(vols, vols)
        assert errors["absolute"]["Left-Hippocampus"] == 0.0
        assert errors["relative"]["Left-Hippocampus"] == 0.0

    def test_different_volumes(self) -> None:
        method_vols = {"Left-Hippocampus": 3000.0}
        hr_vols = {"Left-Hippocampus": 3200.0}
        errors = compute_volume_errors(method_vols, hr_vols)
        assert errors["absolute"]["Left-Hippocampus"] == pytest.approx(200.0)
        assert errors["relative"]["Left-Hippocampus"] == pytest.approx(200.0 / 3200.0)

    def test_missing_region_skipped(self) -> None:
        method_vols = {"Left-Hippocampus": 3000.0}
        hr_vols = {"Left-Hippocampus": 3200.0, "Left-Thalamus": 1000.0}
        errors = compute_volume_errors(method_vols, hr_vols)
        assert "Left-Hippocampus" in errors["absolute"]
        assert "Left-Thalamus" not in errors["absolute"]

    def test_zero_hr_volume(self) -> None:
        method_vols = {"Left-Hippocampus": 100.0}
        hr_vols = {"Left-Hippocampus": 0.0}
        errors = compute_volume_errors(method_vols, hr_vols)
        assert errors["absolute"]["Left-Hippocampus"] == 100.0
        assert np.isnan(errors["relative"]["Left-Hippocampus"])


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------


class TestCSVParsing:
    """Tests for QC and volumes CSV parsing."""

    def test_parse_qc_csv(self, tmp_qc_csv: Path) -> None:
        scores = parse_qc_csv(tmp_qc_csv)
        assert scores["P1-t1c"] == pytest.approx(0.85)
        assert scores["P2-t1c"] == pytest.approx(0.92)
        assert scores["P3-t1c"] == pytest.approx(0.78)

    def test_parse_volumes_csv(self, tmp_volumes_csv: Path) -> None:
        volumes = parse_volumes_csv(tmp_volumes_csv)
        assert "P1-t1c" in volumes
        assert volumes["P1-t1c"]["Left-Hippocampus"] == pytest.approx(3200.5)
        assert volumes["P2-t1c"]["Right-Hippocampus"] == pytest.approx(3050.8)

    def test_parse_nonexistent_csv(self, tmp_path: Path) -> None:
        from pacs_sr.experiments.synthseg_evaluation import _safe_parse_qc

        result = _safe_parse_qc(tmp_path / "nonexistent.csv")
        assert result == {}


# ---------------------------------------------------------------------------
# Bootstrap CI test
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, size=50)
        lo, hi = _bootstrap_ci(x, n_resamples=2000, ci=0.95)
        mean = np.mean(x)
        assert lo < mean < hi

    def test_bootstrap_ci_narrow_for_low_variance(self) -> None:
        x = np.ones(100) * 5.0  # No variance
        lo, hi = _bootstrap_ci(x)
        assert hi - lo < 0.01


# ---------------------------------------------------------------------------
# Statistical analysis test
# ---------------------------------------------------------------------------


class TestStatisticalAnalysis:
    """Tests for the statistical analysis pipeline."""

    def _make_synthetic_df(self, n_patients: int = 20) -> pd.DataFrame:
        """Create synthetic results DataFrame."""
        rng = np.random.default_rng(42)
        rows = []
        patients = [f"P{i:03d}" for i in range(n_patients)]

        for spacing in ["3mm", "5mm", "7mm"]:
            for pid in patients:
                # HR has highest QC
                rows.append(
                    {
                        "patient_id": pid,
                        "spacing": spacing,
                        "method": "HR",
                        "qc_score": rng.normal(0.95, 0.02),
                        "dice_mean": 1.0,
                    }
                )
                # PaCS_SR next best
                rows.append(
                    {
                        "patient_id": pid,
                        "spacing": spacing,
                        "method": "PACS_SR",
                        "qc_score": rng.normal(0.88, 0.03),
                        "dice_mean": rng.normal(0.85, 0.05),
                    }
                )
                # BSPLINE lower
                rows.append(
                    {
                        "patient_id": pid,
                        "spacing": spacing,
                        "method": "BSPLINE",
                        "qc_score": rng.normal(0.80, 0.04),
                        "dice_mean": rng.normal(0.78, 0.06),
                    }
                )
                # ECLARE similar to PaCS_SR
                rows.append(
                    {
                        "patient_id": pid,
                        "spacing": spacing,
                        "method": "ECLARE",
                        "qc_score": rng.normal(0.85, 0.03),
                        "dice_mean": rng.normal(0.82, 0.05),
                    }
                )

        return pd.DataFrame(rows)

    def test_statistical_analysis_runs(self) -> None:
        df = self._make_synthetic_df()
        config = SynthSegEvalConfig(
            methods=("HR", "BSPLINE", "ECLARE", "PACS_SR"),
            spacings=("3mm", "5mm", "7mm"),
            reference_method="PACS_SR",
            alpha=0.05,
            correction="fdr_bh",
        )
        results = run_statistical_analysis(df, config)

        # Should have qc_score and dice_mean
        assert "qc_score" in results
        assert "dice_mean" in results

        # Should have results for each spacing
        for spacing in ["3mm", "5mm", "7mm"]:
            assert spacing in results["qc_score"]

    def test_pairwise_comparisons_present(self) -> None:
        df = self._make_synthetic_df()
        config = SynthSegEvalConfig(
            methods=("HR", "BSPLINE", "ECLARE", "PACS_SR"),
            spacings=("3mm",),
            reference_method="PACS_SR",
        )
        results = run_statistical_analysis(df, config)
        pairwise = results["qc_score"]["3mm"]["pairwise"]
        assert "PACS_SR_vs_BSPLINE" in pairwise
        assert "PACS_SR_vs_ECLARE" in pairwise

    def test_effect_sizes_computed(self) -> None:
        df = self._make_synthetic_df()
        config = SynthSegEvalConfig(
            methods=("HR", "BSPLINE", "ECLARE", "PACS_SR"),
            spacings=("3mm",),
            reference_method="PACS_SR",
        )
        results = run_statistical_analysis(df, config)
        comp = results["qc_score"]["3mm"]["pairwise"]["PACS_SR_vs_BSPLINE"]
        assert "cohens_dz" in comp
        assert "cliffs_delta" in comp
        assert "ci_95" in comp
        assert not np.isnan(comp["cohens_dz"])

    def test_fdr_correction_applied(self) -> None:
        df = self._make_synthetic_df()
        config = SynthSegEvalConfig(
            methods=("HR", "BSPLINE", "ECLARE", "PACS_SR"),
            spacings=("3mm",),
            reference_method="PACS_SR",
            correction="fdr_bh",
        )
        results = run_statistical_analysis(df, config)
        comp = results["qc_score"]["3mm"]["pairwise"]["PACS_SR_vs_BSPLINE"]
        # p_adjusted should be filled (not NaN)
        assert not np.isnan(comp["p_adjusted"])
        # Adjusted should be >= raw p-value
        assert comp["p_adjusted"] >= comp["p_value"] - 1e-10

    def test_friedman_test_present(self) -> None:
        df = self._make_synthetic_df()
        config = SynthSegEvalConfig(
            methods=("HR", "BSPLINE", "ECLARE", "PACS_SR"),
            spacings=("3mm",),
            reference_method="PACS_SR",
        )
        results = run_statistical_analysis(df, config)
        friedman = results["qc_score"]["3mm"].get("friedman", {})
        assert "statistic" in friedman
        assert "p_value" in friedman


# ---------------------------------------------------------------------------
# DataFrame conversion test
# ---------------------------------------------------------------------------


class TestDataFrameConversion:
    """Tests for results_to_dataframe conversion."""

    def test_hr_rows_have_dice_one(self) -> None:
        results = {
            "metadata": {
                "pulse": "t1c",
                "spacings": ["3mm"],
                "methods": ["BSPLINE"],
            },
            "per_patient": {
                "P1": {
                    "3mm": {
                        "BSPLINE": {
                            "qc_score": 0.8,
                            "dice": {"mean": 0.85},
                            "volume_error": {},
                            "rel_volume_error": {},
                        }
                    }
                },
            },
            "hr_reference": {
                "P1": {"qc_score": 0.95, "volumes": {}},
            },
        }
        df = results_to_dataframe(results)
        hr_rows = df[df["method"] == "HR"]
        assert len(hr_rows) == 1
        assert hr_rows.iloc[0]["dice_mean"] == 1.0

    def test_all_methods_present(self) -> None:
        results = {
            "metadata": {
                "pulse": "t1c",
                "spacings": ["3mm"],
                "methods": ["BSPLINE", "ECLARE", "PACS_SR"],
            },
            "per_patient": {
                "P1": {
                    "3mm": {
                        "BSPLINE": {
                            "qc_score": 0.8,
                            "dice": {"mean": 0.7},
                            "volume_error": {},
                            "rel_volume_error": {},
                        },
                        "ECLARE": {
                            "qc_score": 0.85,
                            "dice": {"mean": 0.75},
                            "volume_error": {},
                            "rel_volume_error": {},
                        },
                        "PACS_SR": {
                            "qc_score": 0.9,
                            "dice": {"mean": 0.82},
                            "volume_error": {},
                            "rel_volume_error": {},
                        },
                    }
                },
            },
            "hr_reference": {"P1": {"qc_score": 0.95, "volumes": {}}},
        }
        df = results_to_dataframe(results)
        methods = set(df["method"].unique())
        assert methods == {"HR", "BSPLINE", "ECLARE", "PACS_SR"}


# ---------------------------------------------------------------------------
# Config loading test
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_default_config_values(self) -> None:
        config = SynthSegEvalConfig()
        assert config.robust is True
        assert config.pulse == "t1c"
        assert config.alpha == 0.05
        assert config.correction == "fdr_bh"
        assert config.reference_method == "PACS_SR"

    def test_config_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
synthseg:
  command: ["/usr/bin/python", "/opt/SynthSeg/predict.py"]
  robust: false
  threads: 4
  methods: ["HR", "BSPLINE", "PACS_SR"]
  spacings: ["3mm", "5mm"]
  pulse: "t1c"
  alpha: 0.01
  output_dir: "/tmp/test_synthseg"

data:
  source-h5: "/data/source.h5"
  experts-dir: "/data/experts"
  out: "/data/manifest.json"

pacs_sr:
  experiment_name: "test_exp"
  out_root: "/results/test"
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        from pacs_sr.experiments.synthseg_evaluation import load_synthseg_eval_config

        config = load_synthseg_eval_config(yaml_path)
        assert config.command == ("/usr/bin/python", "/opt/SynthSeg/predict.py")
        assert config.robust is False
        assert config.threads == 4
        assert config.methods == ("HR", "BSPLINE", "PACS_SR")
        assert config.spacings == ("3mm", "5mm")
        assert config.alpha == 0.01
        assert config.source_h5 == Path("/data/source.h5")
