#!/usr/bin/env python3
"""
Largest connected component brain mask from a NIfTI atlas.

Dependencies:
  - nibabel
  - numpy
  - scipy (scipy.ndimage)

CLI:
  python lcc_brain_mask.py --in t1_brain.nii --out t1_brain_mask.nii.gz --thr 0.1 --conn 26
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Config:
    in_path: Path
    out_path: Path | None
    thr: float = 0.1
    conn: int = 26
    verbose: bool = False


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Load a NIfTI file.

    Returns
    -------
    data : np.ndarray
        Image array loaded into memory.
    affine : np.ndarray
        Affine transform from voxel to world.
    header : nib.Nifti1Header
        Header used to preserve metadata on save.
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)  # float data for thresholding
    return data, img.affine, img.header


def largest_cc_mask(vol: np.ndarray, thr: float = 0.1, connectivity: int = 26) -> np.ndarray:
    """
    Compute the largest connected component after binary thresholding.

    Parameters
    ----------
    vol : np.ndarray
        3D input volume (or 4D with singleton dimensions that will be squeezed).
    thr : float
        Absolute threshold applied as (vol >= thr).
    connectivity : int
        6, 18, or 26 for 3D connectivity.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of the largest component, same shape as input.

    Raises
    ------
    ValueError
        If thresholding yields an empty mask.
    """
    # Squeeze out singleton dimensions (e.g., (240, 240, 155, 1) -> (240, 240, 155))
    vol = np.squeeze(vol)
    
    if vol.ndim != 3:
        raise ValueError(f"Input volume must be 3D after squeezing. Got shape: {vol.shape}")

    bin_img = vol >= thr

    if not np.any(bin_img):
        raise ValueError("Threshold produced an empty mask. Check the threshold value.")

    if connectivity == 6:
        structure = ndi.generate_binary_structure(3, 1)
    elif connectivity == 18:
        structure = np.array(
            [[[0,1,0],[1,1,1],[0,1,0]],
             [[1,1,1],[1,1,1],[1,1,1]],
             [[0,1,0],[1,1,1],[0,1,0]]], dtype=bool
        )
    elif connectivity == 26:
        structure = ndi.generate_binary_structure(3, 2)
    else:
        raise ValueError("connectivity must be one of {6, 18, 26}.")

    labeled, nlab = ndi.label(bin_img, structure=structure)
    if nlab == 1:
        return bin_img.astype(bool)

    # bincount ignores 0 by slicing from index 1
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    max_label = int(np.argmax(counts))
    mask = labeled == max_label
    return mask.astype(bool)


def save_mask(mask: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, out_path: Path) -> None:
    """
    Save a boolean mask as uint8 NIfTI.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or 0/1 mask.
    affine : np.ndarray
        Affine from source image.
    header : nib.Nifti1Header
        Header copied from source to preserve meta.
    out_path : Path
        Output path (.nii or .nii.gz).
    """
    hdr = header.copy()
    hdr.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(mask.astype(np.uint8, copy=False), affine, header=hdr)
    nib.save(img, str(out_path))


def infer_out_path(in_path: Path, explicit_out: Path | None) -> Path:
    """
    Derive an output path if not provided by appending '_mask' and using .nii.gz.
    """
    if explicit_out is not None:
        return explicit_out
    stem = in_path.name
    if stem.endswith(".nii.gz"):
        base = stem[:-7]
    elif stem.endswith(".nii"):
        base = stem[:-4]
    else:
        base = stem
    return in_path.with_name(f"{base}_mask.nii.gz")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Compute largest connected component brain mask from a NIfTI atlas."
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path, help="Input NIfTI (e.g., t1_brain.nii)")
    parser.add_argument("--out", dest="out_path", required=False, type=Path, help="Output NIfTI path")
    parser.add_argument("--thr", dest="thr", type=float, default=0.1, help="Absolute threshold (default 0.1)")
    parser.add_argument("--conn", dest="conn", type=int, default=26, choices=[6, 18, 26], help="Connectivity (default 26)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    return Config(in_path=args.in_path, out_path=args.out_path, thr=args.thr, conn=args.conn, verbose=args.verbose)


def main(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(levelname)s:%(message)s"
    )
    logging.info("Loading: %s", cfg.in_path)
    vol, affine, header = load_nifti(cfg.in_path)

    logging.info("Thresholding at %.6f and labeling with %d-connectivity", cfg.thr, cfg.conn)
    mask = largest_cc_mask(vol, thr=cfg.thr, connectivity=cfg.conn)

    out_path = infer_out_path(cfg.in_path, cfg.out_path)
    logging.info("Saving mask to: %s", out_path)
    save_mask(mask, affine, header, out_path)
    logging.info("Done.")


if __name__ == "__main__":
    main(parse_args())
