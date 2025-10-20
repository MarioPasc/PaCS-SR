"""
ANTs-based registration utilities for aligning volumes to atlas templates
and applying brain masks.
"""
from __future__ import annotations
import numpy as np
import ants
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Map pulse sequences to their corresponding atlas templates
ATLAS_TEMPLATE_MAP = {
    "t1c": "T1_brain.nii",
    "t1n": "T1_brain.nii",
    "t2w": "T2_brain.nii",
    "t2f": "T2_brain.nii",  # T2FLAIR uses T2 as closest contrast
}


def get_atlas_path(atlas_dir: Path, pulse: str) -> Path:
    """
    Get the atlas template path for a given pulse sequence.

    Args:
        atlas_dir: Directory containing atlas templates
        pulse: Pulse sequence name (t1c, t1n, t2w, t2f)

    Returns:
        Path to the atlas template file

    Raises:
        ValueError: If pulse is not recognized
        FileNotFoundError: If atlas file does not exist
    """
    template_name = ATLAS_TEMPLATE_MAP.get(pulse)
    if template_name is None:
        raise ValueError(f"No atlas template mapping for pulse: {pulse}")

    atlas_path = atlas_dir / template_name
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas template not found: {atlas_path}")

    return atlas_path


def register_to_atlas_rigid(
    moving_volume: np.ndarray,
    affine: np.ndarray,
    pulse: str,
    atlas_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigidly register a volume to the appropriate atlas template.

    Args:
        moving_volume: 3D numpy array to register
        affine: 4x4 affine matrix for the moving volume
        pulse: Pulse sequence name (t1c, t1n, t2w, t2f)
        atlas_dir: Directory containing atlas templates

    Returns:
        Tuple of (registered_volume, registered_affine)
    """
    atlas_path = get_atlas_path(atlas_dir, pulse)

    # Load atlas as fixed image
    fixed = ants.image_read(str(atlas_path))

    # Convert numpy volume to ANTs image
    # Extract spacing from affine diagonal
    spacing = tuple(np.abs(np.diag(affine[:3, :3])))
    origin = tuple(affine[:3, 3])
    direction = affine[:3, :3] / spacing  # Normalize to get direction matrix

    moving = ants.from_numpy(
        moving_volume,
        origin=origin,
        spacing=spacing,
        direction=direction
    )

    # Perform rigid registration (6 DOF: 3 translations + 3 rotations)
    logger.debug(f"Registering volume to atlas: {atlas_path}")
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Rigid',
        verbose=False
    )

    # Get registered image
    registered = registration['warpedmovout']
    registered_volume = registered.numpy()

    # Get registered affine from ANTs image
    registered_affine = np.eye(4)
    registered_affine[:3, :3] = registered.direction * registered.spacing
    registered_affine[:3, 3] = registered.origin

    return registered_volume, registered_affine


def extract_brain_mask(atlas_path: Path, atlas_dir: Path) -> np.ndarray:
    """
    Extract brain mask from atlas template.

    First tries to load brain_mask.nii from atlas_dir.
    If not found, computes mask from atlas template (non-zero voxels).

    Args:
        atlas_path: Path to atlas template
        atlas_dir: Directory containing atlas files

    Returns:
        Binary mask as boolean numpy array
    """
    # Try to load pre-computed brain mask
    brain_mask_path = atlas_dir / "brain_mask.nii"
    if brain_mask_path.exists():
        logger.debug(f"Loading pre-computed brain mask: {brain_mask_path}")
        mask_img = ants.image_read(str(brain_mask_path))
        mask = mask_img.numpy() > 0
        return mask

    # Fallback: compute mask from atlas template
    logger.debug(f"Computing brain mask from atlas: {atlas_path}")
    atlas = ants.image_read(str(atlas_path))
    mask = atlas.numpy() > 0
    return mask


def apply_brain_mask(
    volume: np.ndarray,
    pulse: str,
    atlas_dir: Path,
    mask_cache: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Apply brain mask to volume by setting out-of-brain voxels to 0.

    Args:
        volume: 3D numpy array (must already be registered to atlas space)
        pulse: Pulse sequence name
        atlas_dir: Directory containing atlas templates
        mask_cache: Optional cache dict to avoid reloading masks

    Returns:
        Masked volume (copy with out-of-brain voxels set to 0)
    """
    atlas_path = get_atlas_path(atlas_dir, pulse)

    # Try to use cached mask
    cache_key = f"{atlas_dir}/brain_mask"
    if mask_cache is not None and cache_key in mask_cache:
        mask = mask_cache[cache_key]
    else:
        mask = extract_brain_mask(atlas_path, atlas_dir)
        if mask_cache is not None:
            mask_cache[cache_key] = mask

    # Ensure mask and volume have same shape
    if mask.shape != volume.shape:
        logger.warning(
            f"Mask shape {mask.shape} != volume shape {volume.shape}. "
            f"Resampling mask to match volume."
        )
        from scipy.ndimage import zoom
        zoom_factors = [v / m for v, m in zip(volume.shape, mask.shape)]
        mask = zoom(mask.astype(float), zoom_factors, order=0) > 0.5

    # Apply mask
    masked = volume.copy()
    masked[~mask] = 0

    return masked


def register_and_mask(
    volume: np.ndarray,
    affine: np.ndarray,
    pulse: str,
    atlas_dir: Path,
    mask_cache: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register volume to atlas and apply brain mask in one step.

    This is the main preprocessing function that should be applied to all
    volumes (HR and SR experts) to ensure they're in atlas space with
    brain-only voxels.

    Args:
        volume: 3D numpy array
        affine: 4x4 affine matrix
        pulse: Pulse sequence name
        atlas_dir: Directory containing atlas templates
        mask_cache: Optional cache dict to avoid reloading masks

    Returns:
        Tuple of (registered_and_masked_volume, registered_affine)
    """
    # Step 1: Register to atlas
    registered, registered_affine = register_to_atlas_rigid(
        volume, affine, pulse, atlas_dir
    )

    # Step 2: Apply brain mask
    masked = apply_brain_mask(registered, pulse, atlas_dir, mask_cache)

    return masked, registered_affine


def register_and_mask_from_files(
    moving_path: Path,
    atlas_path: Path,
    mask_path: Path,
    output_path: Path,
    registration_type: str = 'Rigid',
    verbose: bool = False
) -> None:
    """
    CLI-friendly wrapper: register moving image to atlas and apply brain mask.

    Args:
        moving_path: Path to moving volume (.nii or .nii.gz)
        atlas_path: Path to atlas template (.nii or .nii.gz)
        mask_path: Path to brain mask (.nii or .nii.gz)
        output_path: Path to save registered and masked volume
        registration_type: Type of ANTs registration ('Rigid', 'Affine', 'SyN', etc.)
        verbose: Enable verbose logging
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Loading moving image: {moving_path}")
    moving = ants.image_read(str(moving_path))

    logger.info(f"Loading atlas template: {atlas_path}")
    fixed = ants.image_read(str(atlas_path))

    logger.info(f"Loading brain mask: {mask_path}")
    mask_img = ants.image_read(str(mask_path))
    mask = mask_img.numpy() > 0

    # Perform registration
    logger.info(f"Performing {registration_type} registration...")
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=registration_type,
        verbose=verbose
    )

    # Get registered image
    registered = registration['warpedmovout']
    registered_volume = registered.numpy()

    # Ensure mask matches registered volume shape
    if mask.shape != registered_volume.shape:
        logger.warning(
            f"Mask shape {mask.shape} != registered volume shape {registered_volume.shape}. "
            f"Resampling mask to match."
        )
        from scipy.ndimage import zoom
        zoom_factors = [v / m for v, m in zip(registered_volume.shape, mask.shape)]
        mask = zoom(mask.astype(float), zoom_factors, order=0) > 0.5

    # Apply brain mask
    logger.info("Applying brain mask...")
    masked_volume = registered_volume.copy()
    masked_volume[~mask] = 0

    # Create output ANTs image with registered properties
    output_img = ants.from_numpy(
        masked_volume,
        origin=registered.origin,
        spacing=registered.spacing,
        direction=registered.direction
    )

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving registered and masked volume to: {output_path}")
    ants.image_write(output_img, str(output_path))
    logger.info("✓ Done!")


def main():
    """CLI entry point for registration and brain masking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Register a moving volume to an atlas template and apply brain mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rigid registration (default, 6 DOF)
  python -m pacs_sr.utils.registration \\
      --moving patient_t1.nii.gz \\
      --atlas atlas/T1_brain.nii \\
      --mask atlas/brain_mask.nii \\
      --output registered_patient_t1.nii.gz

  # Affine registration (12 DOF)
  python -m pacs_sr.utils.registration \\
      --moving patient_t1.nii.gz \\
      --atlas atlas/T1_brain.nii \\
      --mask atlas/brain_mask.nii \\
      --output registered_patient_t1.nii.gz \\
      --type Affine

  # Non-linear registration (deformable)
  python -m pacs_sr.utils.registration \\
      --moving patient_t1.nii.gz \\
      --atlas atlas/T1_brain.nii \\
      --mask atlas/brain_mask.nii \\
      --output registered_patient_t1.nii.gz \\
      --type SyN \\
      --verbose
"""
    )
    parser.add_argument(
        "--moving",
        type=Path,
        required=True,
        help="Path to moving volume (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        required=True,
        help="Path to atlas template (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to brain mask (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save registered and masked volume"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="Rigid",
        choices=["Rigid", "Affine", "SyN", "SyNRA", "SyNOnly", "Translation"],
        help="Registration type (default: Rigid)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate input files exist
    for path, name in [
        (args.moving, "Moving image"),
        (args.atlas, "Atlas template"),
        (args.mask, "Brain mask")
    ]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            return 1

    try:
        register_and_mask_from_files(
            moving_path=args.moving,
            atlas_path=args.atlas,
            mask_path=args.mask,
            output_path=args.output,
            registration_type=args.type,
            verbose=args.verbose
        )
        return 0
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
