"""Unit tests for BSPLINE generation script."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.generate_bspline import _process_one


class TestProcessOne:
    """Tests for the single-patient BSPLINE processing function."""

    def _setup_test_data(
        self,
        tmp_path: Path,
        lr_shape: tuple = (182, 218, 60),
        hr_shape: tuple = (182, 218, 182),
    ) -> tuple:
        """Create minimal LR and HR NIfTI files for testing.

        Args:
            tmp_path: Temporary directory.
            lr_shape: Shape of LR volume.
            hr_shape: Shape of HR volume.

        Returns:
            (patient_id, spacing, pulse, lr_root, hr_root, out_root)
        """
        patient_id = "BraTS-TEST-001"
        spacing = "3mm"
        pulse = "t1c"

        lr_root = tmp_path / "low_resolution"
        hr_root = tmp_path / "high_resolution"
        out_root = tmp_path / "BSPLINE"

        # Create LR volume
        lr_dir = lr_root / spacing / patient_id
        lr_dir.mkdir(parents=True)
        lr_data = np.random.rand(*lr_shape).astype(np.float32) * 100
        affine = np.eye(4)
        nib.save(
            nib.Nifti1Image(lr_data, affine),
            str(lr_dir / f"{patient_id}-{pulse}.nii.gz"),
        )

        # Create HR volume (used for shape reference and affine)
        hr_dir = hr_root / patient_id
        hr_dir.mkdir(parents=True)
        hr_data = np.random.rand(*hr_shape).astype(np.float32) * 100
        nib.save(
            nib.Nifti1Image(hr_data, affine),
            str(hr_dir / f"{patient_id}-{pulse}.nii.gz"),
        )

        return patient_id, spacing, pulse, lr_root, hr_root, out_root

    def test_output_shape_matches_hr(self, tmp_path: Path) -> None:
        """BSPLINE output shape should match HR shape."""
        patient_id, spacing, pulse, lr_root, hr_root, out_root = self._setup_test_data(
            tmp_path
        )

        result = _process_one(patient_id, spacing, pulse, lr_root, hr_root, out_root)
        assert result.startswith("OK")

        out_path = (
            out_root / spacing / "output_volumes" / f"{patient_id}-{pulse}.nii.gz"
        )
        assert out_path.exists()

        out_vol = nib.load(str(out_path)).get_fdata()
        assert out_vol.shape == (182, 218, 182), f"Shape mismatch: {out_vol.shape}"

    def test_skip_existing(self, tmp_path: Path) -> None:
        """Should skip if output already exists."""
        patient_id, spacing, pulse, lr_root, hr_root, out_root = self._setup_test_data(
            tmp_path
        )

        # Create dummy output
        out_path = (
            out_root / spacing / "output_volumes" / f"{patient_id}-{pulse}.nii.gz"
        )
        out_path.parent.mkdir(parents=True)
        out_path.write_text("dummy")

        result = _process_one(patient_id, spacing, pulse, lr_root, hr_root, out_root)
        assert result.startswith("SKIP")

    def test_missing_lr_reports_miss(self, tmp_path: Path) -> None:
        """Should report MISS if LR file doesn't exist."""
        patient_id = "BraTS-MISSING"
        lr_root = tmp_path / "lr"
        hr_root = tmp_path / "hr"
        out_root = tmp_path / "out"

        # Create HR only
        hr_dir = hr_root / patient_id
        hr_dir.mkdir(parents=True)
        hr_data = np.random.rand(64, 64, 64).astype(np.float32)
        nib.save(
            nib.Nifti1Image(hr_data, np.eye(4)),
            str(hr_dir / f"{patient_id}-t1c.nii.gz"),
        )

        result = _process_one(patient_id, "3mm", "t1c", lr_root, hr_root, out_root)
        assert result.startswith("MISS")

    def test_output_dtype_float32(self, tmp_path: Path) -> None:
        """Output volume should be float32."""
        patient_id, spacing, pulse, lr_root, hr_root, out_root = self._setup_test_data(
            tmp_path, lr_shape=(32, 32, 10), hr_shape=(32, 32, 32)
        )

        _process_one(patient_id, spacing, pulse, lr_root, hr_root, out_root)
        out_path = (
            out_root / spacing / "output_volumes" / f"{patient_id}-{pulse}.nii.gz"
        )
        out_vol = nib.load(str(out_path)).get_fdata(dtype=np.float32)
        assert out_vol.dtype == np.float32

    def test_isotropic_input_identity(self, tmp_path: Path) -> None:
        """When LR and HR have same shape, output should closely match LR."""
        patient_id, spacing, pulse, lr_root, hr_root, out_root = self._setup_test_data(
            tmp_path, lr_shape=(32, 32, 32), hr_shape=(32, 32, 32)
        )

        _process_one(patient_id, spacing, pulse, lr_root, hr_root, out_root)
        out_path = (
            out_root / spacing / "output_volumes" / f"{patient_id}-{pulse}.nii.gz"
        )
        out_vol = nib.load(str(out_path)).get_fdata(dtype=np.float32)

        lr_path = lr_root / spacing / patient_id / f"{patient_id}-{pulse}.nii.gz"
        lr_vol = nib.load(str(lr_path)).get_fdata(dtype=np.float32)

        # When zoom factor is 1.0 in all dims, output should match input
        assert np.allclose(out_vol, lr_vol, atol=1e-4), (
            "Identity zoom should preserve values"
        )
