"""
imio.py  –  tiny I/O helpers for qualitative SR figures
=======================================================

Functions
---------
load_lps(path, like=None, order=1, dtype=np.float32)
    Load *path*, re-orient to LPS, optionally resample rigidly onto
    `like` (another NIfTI image).  Returns a NumPy array.

as_lps(img)
    Re-orient an in-memory nibabel image to LPS.
"""

from __future__ import annotations
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nibabel.processing import resample_from_to


# ------------------------------------------------------------------ public
def load_lps(path, *,
             like: nib.Nifti1Image | None = None,
             order: int = 1,
             dtype=np.float32) -> np.ndarray:
    """
    Load `path` as LPS-oriented data.  If `like` is given the image is
    rigidly resampled (nearest/linear) onto that grid first.

    Parameters
    ----------
    path : str or PathLike
        Input NIfTI.
    like : nibabel image, optional
        Reference geometry to resample onto.
    order : {0,1}
        0 = nearest-neighbour (labels), 1 = trilinear (images).
    dtype : np.dtype
        Cast final array to this type.

    Returns
    -------
    np.ndarray
        Array of shape *(H, W, D)* in **LPS** orientation.
    """
    img = nib.load(str(path))
    img = as_lps(img)
    if like is not None:
        img = resample_from_to(img, like, order=order)
    return img.get_fdata(dtype=dtype)


# ------------------------------------------------------------------ helper
def as_lps(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return a **new** nibabel image re-oriented to LPS."""
    # current orientation → e.g. [[0,-1],[1,1],[2,1]]  (axis, direction)
    cur_ornt = io_orientation(img.affine)
    tgt_ornt = axcodes2ornt(("L", "P", "S"))          # desired LPS
    if np.array_equal(cur_ornt, tgt_ornt):
        return img                                    # already LPS
    xform = ornt_transform(cur_ornt, tgt_ornt)
    return img.as_reoriented(xform)
