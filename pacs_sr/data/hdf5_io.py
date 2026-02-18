"""Central HDF5 I/O module for PaCS-SR.

Provides read/write primitives for 3D MRI volumes stored in HDF5 files,
along with key-construction helpers that encode the project's data layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------


def read_volume(h5_path: Path, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a float32 3D volume and its (4,4) affine from HDF5.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key inside the file (e.g. "high_resolution/BraTS-GLI-00033-100/t1c").

    Returns:
        Tuple of (data, affine) where data is float32 and affine is float64 (4,4).

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        KeyError: If the key does not exist in the file.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        data = ds[()].astype(np.float32)
        affine = np.array(ds.attrs["affine"], dtype=np.float64).reshape(4, 4)
    return data, affine


def write_volume(
    h5_path: Path,
    key: str,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    extra_attrs: dict | None = None,
) -> None:
    """Write a float32 volume with affine attribute to HDF5. Uses gzip-4 compression.

    Args:
        h5_path: Path to the HDF5 file (created if it doesn't exist).
        key: Dataset key inside the file.
        data: Volume array (any shape, stored as float32).
        affine: 4x4 affine matrix (stored as float64 attribute).
        extra_attrs: Optional dict of additional attributes to store on the dataset.
    """
    with h5py.File(h5_path, "a") as f:
        if key in f:
            del f[key]
        ds = f.create_dataset(
            key,
            data=data.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )
        ds.attrs["affine"] = affine.astype(np.float64).ravel()
        if extra_attrs:
            for k, v in extra_attrs.items():
                ds.attrs[k] = v


def list_groups(h5_path: Path, prefix: str) -> List[str]:
    """List immediate child group names under *prefix*.

    Args:
        h5_path: Path to the HDF5 file.
        prefix: Group path to inspect (e.g. "high_resolution").

    Returns:
        Sorted list of child group names (empty list if file does not exist).
    """
    if not Path(h5_path).exists():
        return []
    with h5py.File(h5_path, "r") as f:
        grp = f[prefix]
        return sorted(name for name in grp if isinstance(grp[name], h5py.Group))


def has_key(h5_path: Path, key: str) -> bool:
    """Check whether *key* exists as a dataset in the HDF5 file.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key to check.

    Returns:
        True if key exists, False otherwise.
    """
    if not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return key in f
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Key construction helpers
# ---------------------------------------------------------------------------


def hr_key(patient_id: str, pulse: str) -> str:
    """HDF5 key for a high-resolution volume in source_data.h5."""
    return f"high_resolution/{patient_id}/{pulse}"


def lr_key(spacing: str, patient_id: str, pulse: str) -> str:
    """HDF5 key for a low-resolution volume in source_data.h5."""
    return f"low_resolution/{spacing}/{patient_id}/{pulse}"


def expert_key(spacing: str, patient_id: str, pulse: str) -> str:
    """HDF5 key for an expert output in {model}.h5."""
    return f"{spacing}/{patient_id}/{pulse}"


def expert_h5_path(experts_dir: Path, model: str) -> Path:
    """Path to the HDF5 file for a given expert model."""
    return Path(experts_dir) / f"{model.lower()}.h5"


def blend_key(spacing: str, patient_id: str, pulse: str) -> str:
    """HDF5 key for a PaCS-SR blend in results_fold_{N}.h5."""
    return f"{spacing}/blends/{patient_id}/{pulse}"


def weight_key(spacing: str, pulse: str, patient_id: str) -> str:
    """HDF5 key for PaCS-SR weight maps in results_fold_{N}.h5."""
    return f"{spacing}/weights/{pulse}/{patient_id}"


def results_h5_path(out_root: Path, experiment_name: str, fold_num: int) -> Path:
    """Path to the results HDF5 file for a given fold."""
    return Path(out_root) / experiment_name / f"results_fold_{fold_num}.h5"
