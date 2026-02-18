"""Unit tests for 3D Multi-Scale SSIM metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pacs_sr.model.metrics_3d import ms_ssim_3d, _TORCH_AVAILABLE

# Real BraTS volume for integration tests (182×218×182, well above MS-SSIM minimum)
_REAL_VOL_PATH = Path(
    "/media/mpascual/PortableSSD/BraTS_GLI/LowRes_HighRes/high_resolution"
    "/BraTS-GLI-00033-100/BraTS-GLI-00033-100-t1c.nii.gz"
)

_HAS_REAL_DATA = _REAL_VOL_PATH.exists()


def _load_real_volume() -> np.ndarray:
    """Load the real BraTS volume for testing."""
    import nibabel as nib

    return nib.load(str(_REAL_VOL_PATH)).get_fdata(dtype=np.float32)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="pytorch-msssim not installed")
class TestMsSSIM3D:
    """Tests for ms_ssim_3d function."""

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="Real BraTS data not available")
    def test_identical_real_volume_near_one(self) -> None:
        """Identical real MRI volumes should produce MS-SSIM very close to 1.0."""
        vol = _load_real_volume()
        result = ms_ssim_3d(vol, vol)
        assert result > 0.999, f"Expected ~1.0 for identical volumes, got {result}"

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="Real BraTS data not available")
    def test_noisy_real_volume_below_one(self) -> None:
        """Real volume vs noisy version should give MS-SSIM < 1."""
        vol = _load_real_volume()
        rng = np.random.RandomState(42)
        noisy = vol + rng.randn(*vol.shape).astype(np.float32) * 200
        result = ms_ssim_3d(vol, noisy)
        assert 0.0 < result < 0.99, f"Expected degraded MS-SSIM, got {result}"

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="Real BraTS data not available")
    def test_with_mask_real_volume(self) -> None:
        """MS-SSIM with brain mask on identical volumes should be ~1.0."""
        vol = _load_real_volume()
        mask = vol > 0
        result = ms_ssim_3d(vol, vol, mask=mask)
        assert result > 0.999, (
            f"Expected ~1.0 for identical masked volumes, got {result}"
        )

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="Real BraTS data not available")
    def test_explicit_data_range_real_volume(self) -> None:
        """MS-SSIM should accept explicit data_range parameter."""
        vol = _load_real_volume()
        data_range = float(vol.max() - vol.min())
        result = ms_ssim_3d(vol, vol, data_range=data_range)
        assert result > 0.999, f"Expected ~1.0, got {result}"

    def test_small_volume_graceful(self) -> None:
        """Small synthetic volumes should degrade gracefully (no crash)."""
        rng = np.random.RandomState(42)
        vol = rng.rand(24, 24, 24).astype(np.float32) * 100
        result = ms_ssim_3d(vol, vol)
        assert np.isfinite(result) or np.isnan(result)

    def test_constant_volume_returns_one(self) -> None:
        """Constant volume (zero data_range) should return 1.0."""
        vol = np.ones((64, 64, 64), dtype=np.float32) * 42.0
        result = ms_ssim_3d(vol, vol)
        assert result == 1.0, f"Expected 1.0 for constant volume, got {result}"

    def test_medium_synthetic_volume(self) -> None:
        """64³ synthetic volume should work with reduced win_size/scales."""
        rng = np.random.RandomState(42)
        vol = rng.rand(64, 64, 64).astype(np.float32) * 100
        result = ms_ssim_3d(vol, vol)
        assert np.isfinite(result), f"Expected finite result for 64³, got {result}"
        if np.isfinite(result):
            assert result > 0.95, (
                f"Identical 64³ volumes should be near 1.0, got {result}"
            )
