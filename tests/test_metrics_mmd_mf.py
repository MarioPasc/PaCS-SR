"""Unit tests for MMD-MF metric."""

from __future__ import annotations

import numpy as np

from pacs_sr.model.metrics_mmd_mf import (
    mmd_mf,
    _extract_features,
    _sobel_magnitude,
    _log_response,
    _local_variance,
    _polynomial_mmd,
)


class TestFeatureExtraction:
    """Tests for morphological feature extraction functions."""

    def test_sobel_magnitude_shape(self) -> None:
        """Sobel magnitude should preserve volume shape."""
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        result = _sobel_magnitude(vol)
        assert result.shape == vol.shape

    def test_sobel_magnitude_nonnegative(self) -> None:
        """Gradient magnitude should be non-negative."""
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        result = _sobel_magnitude(vol)
        assert np.all(result >= 0)

    def test_log_response_shape(self) -> None:
        """LoG response should preserve volume shape."""
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        for sigma in [1.0, 2.0, 4.0]:
            result = _log_response(vol, sigma)
            assert result.shape == vol.shape

    def test_local_variance_nonnegative(self) -> None:
        """Local variance should be non-negative."""
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        result = _local_variance(vol, radius=2)
        assert np.all(result >= 0)

    def test_extract_features_output_shape(self) -> None:
        """Feature extraction should produce (N, 5) array."""
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        feats = _extract_features(vol)
        assert feats.shape == (16 * 16 * 16, 5)

    def test_constant_volume_zero_gradient(self) -> None:
        """Constant volume should have near-zero Sobel magnitude."""
        vol = np.ones((16, 16, 16), dtype=np.float32) * 42.0
        result = _sobel_magnitude(vol)
        assert np.allclose(result, 0.0, atol=1e-6)


class TestPolynomialMMD:
    """Tests for the polynomial kernel MMD function."""

    def test_identical_distributions_near_zero(self) -> None:
        """MMD between identical feature sets should be near zero."""
        rng = np.random.RandomState(42)
        feats = rng.randn(100, 5).astype(np.float32)
        result = _polynomial_mmd(feats, feats)
        assert result < 1e-6, f"Expected ~0 for identical distributions, got {result}"

    def test_different_distributions_positive(self) -> None:
        """MMD between different distributions should be positive."""
        rng = np.random.RandomState(42)
        feats_a = rng.randn(100, 5).astype(np.float32)
        feats_b = rng.randn(100, 5).astype(np.float32) + 5.0
        result = _polynomial_mmd(feats_a, feats_b)
        assert result > 0.0, (
            f"Expected positive MMD for different distributions, got {result}"
        )

    def test_empty_input_returns_nan(self) -> None:
        """Empty input should return NaN."""
        result = _polynomial_mmd(np.empty((0, 5)), np.empty((0, 5)))
        assert np.isnan(result)

    def test_nonnegative(self) -> None:
        """MMD should always be non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            feats_a = rng.randn(50, 5).astype(np.float32)
            feats_b = rng.randn(50, 5).astype(np.float32)
            result = _polynomial_mmd(feats_a, feats_b)
            assert result >= 0.0


class TestMmdMf:
    """Integration tests for the full mmd_mf function."""

    def test_identical_volumes_near_zero(self) -> None:
        """MMD-MF between identical volumes should be near zero."""
        rng = np.random.RandomState(42)
        vol = rng.rand(32, 32, 32).astype(np.float32) * 100
        result = mmd_mf(vol, vol)
        assert result < 0.01, f"Expected ~0 for identical volumes, got {result}"

    def test_different_volumes_positive(self) -> None:
        """MMD-MF between different volumes should be positive."""
        rng = np.random.RandomState(42)
        vol_a = rng.rand(32, 32, 32).astype(np.float32) * 100
        vol_b = rng.rand(32, 32, 32).astype(np.float32) * 100
        result = mmd_mf(vol_a, vol_b)
        assert result > 0.0, f"Expected positive MMD-MF, got {result}"

    def test_with_mask(self) -> None:
        """MMD-MF with mask should work."""
        rng = np.random.RandomState(42)
        vol = rng.rand(32, 32, 32).astype(np.float32) * 100
        mask = np.zeros((32, 32, 32), dtype=bool)
        mask[5:25, 5:25, 5:25] = True
        result = mmd_mf(vol, vol, mask=mask)
        assert result < 0.01, f"Expected ~0 for identical masked volumes, got {result}"

    def test_empty_mask_returns_nan(self) -> None:
        """Empty mask should return NaN."""
        vol = np.random.rand(32, 32, 32).astype(np.float32)
        mask = np.zeros((32, 32, 32), dtype=bool)
        result = mmd_mf(vol, vol, mask=mask)
        assert np.isnan(result)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same result."""
        rng = np.random.RandomState(42)
        vol_a = rng.rand(32, 32, 32).astype(np.float32) * 100
        vol_b = rng.rand(32, 32, 32).astype(np.float32) * 100
        r1 = mmd_mf(vol_a, vol_b, seed=123)
        r2 = mmd_mf(vol_a, vol_b, seed=123)
        assert r1 == r2, f"Results differ with same seed: {r1} vs {r2}"
