"""Round-trip tests for pacs_sr.data.hdf5_io."""

from __future__ import annotations

import numpy as np
import pytest

from pacs_sr.data.hdf5_io import (
    blend_key,
    expert_h5_path,
    expert_key,
    has_key,
    hr_key,
    list_groups,
    lr_key,
    read_volume,
    results_h5_path,
    weight_key,
    write_volume,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def h5_file(tmp_path):
    """Return path to a temporary HDF5 file."""
    return tmp_path / "test.h5"


def _random_volume(shape=(16, 24, 32)):
    return np.random.default_rng(42).standard_normal(shape).astype(np.float32)


def _random_affine():
    aff = np.eye(4, dtype=np.float64)
    aff[:3, 3] = np.random.default_rng(7).uniform(-100, 100, size=3)
    return aff


# ---------------------------------------------------------------------------
# Core I/O tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Write then read back — data and affine must match."""

    def test_basic_round_trip(self, h5_file):
        data = _random_volume()
        affine = _random_affine()
        key = "test/volume"

        write_volume(h5_file, key, data, affine)
        data_out, affine_out = read_volume(h5_file, key)

        np.testing.assert_array_almost_equal(data_out, data, decimal=6)
        np.testing.assert_array_almost_equal(affine_out, affine, decimal=12)

    def test_4d_round_trip(self, h5_file):
        """Weight volumes are 4D — ensure they survive round-trip."""
        data = np.random.default_rng(0).random((8, 12, 16, 4)).astype(np.float32)
        affine = _random_affine()
        key = "weights/patient/pulse"

        write_volume(h5_file, key, data, affine)
        data_out, affine_out = read_volume(h5_file, key)

        np.testing.assert_array_almost_equal(data_out, data, decimal=6)
        assert data_out.shape == (8, 12, 16, 4)

    def test_overwrite(self, h5_file):
        """Writing to the same key should overwrite cleanly."""
        key = "overwrite/test"
        data1 = _random_volume((4, 4, 4))
        data2 = _random_volume((8, 8, 8))
        affine = _random_affine()

        write_volume(h5_file, key, data1, affine)
        write_volume(h5_file, key, data2, affine)

        data_out, _ = read_volume(h5_file, key)
        assert data_out.shape == (8, 8, 8)
        np.testing.assert_array_almost_equal(data_out, data2, decimal=6)

    def test_extra_attrs(self, h5_file):
        """Extra attributes should be stored on the dataset."""
        import h5py

        data = _random_volume((4, 4, 4))
        affine = _random_affine()
        key = "attrs/test"
        extras = {"model_names": ["BSPLINE", "ECLARE"], "patch_size": 32}

        write_volume(h5_file, key, data, affine, extra_attrs=extras)

        with h5py.File(h5_file, "r") as f:
            ds = f[key]
            assert ds.attrs["patch_size"] == 32
            assert list(ds.attrs["model_names"]) == ["BSPLINE", "ECLARE"]

    def test_missing_key_raises(self, h5_file):
        """Reading a nonexistent key should raise KeyError."""
        data = _random_volume((2, 2, 2))
        write_volume(h5_file, "exists", data, np.eye(4))

        with pytest.raises(KeyError):
            read_volume(h5_file, "does_not_exist")


# ---------------------------------------------------------------------------
# Group listing and key existence
# ---------------------------------------------------------------------------


class TestListGroupsAndHasKey:
    def test_list_groups(self, h5_file):
        affine = np.eye(4)
        write_volume(h5_file, "parent/child_a/data", _random_volume((2, 2, 2)), affine)
        write_volume(h5_file, "parent/child_b/data", _random_volume((2, 2, 2)), affine)

        groups = list_groups(h5_file, "parent")
        assert groups == ["child_a", "child_b"]

    def test_has_key_true(self, h5_file):
        write_volume(h5_file, "a/b/c", _random_volume((2, 2, 2)), np.eye(4))
        assert has_key(h5_file, "a/b/c") is True

    def test_has_key_false(self, h5_file):
        write_volume(h5_file, "a/b/c", _random_volume((2, 2, 2)), np.eye(4))
        assert has_key(h5_file, "a/b/d") is False


# ---------------------------------------------------------------------------
# Key construction helpers
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_hr_key(self):
        assert hr_key("PAT001", "t1c") == "high_resolution/PAT001/t1c"

    def test_lr_key(self):
        assert lr_key("3mm", "PAT001", "t1c") == "low_resolution/3mm/PAT001/t1c"

    def test_expert_key(self):
        assert expert_key("5mm", "PAT001", "t2w") == "5mm/PAT001/t2w"

    def test_expert_h5_path(self, tmp_path):
        p = expert_h5_path(tmp_path, "BSPLINE")
        assert p == tmp_path / "bspline.h5"

    def test_blend_key(self):
        assert blend_key("3mm", "PAT001", "t1c") == "3mm/blends/PAT001/t1c"

    def test_weight_key(self):
        assert weight_key("3mm", "t1c", "PAT001") == "3mm/weights/t1c/PAT001"

    def test_results_h5_path(self, tmp_path):
        p = results_h5_path(tmp_path, "SERAM_GLI", 2)
        assert p == tmp_path / "SERAM_GLI" / "results_fold_2.h5"
