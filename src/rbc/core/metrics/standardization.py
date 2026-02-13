"""Z-score standardization of derivative maps.

Converts voxelwise derivative maps (ALFF, fALFF, ReHo, centrality) to
z-scores using the in-mask mean and standard deviation so that maps are
comparable across subjects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def zscore(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score a 3D map within a brain mask.

    Computes ``(data - mean) / std`` where mean and std are calculated
    over in-mask voxels only.  Out-of-mask voxels are set to zero.

    Args:
        data: 3D array of voxel values.
        mask: 3D boolean-compatible brain mask in the same space.

    Returns:
        3D float64 array of z-scored values.

    Raises:
        ValueError: If *data* is not 3D.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got {data.ndim}D")

    mask_bool = mask > 0
    masked = data[mask_bool]
    mean = masked.mean()
    std = masked.std()

    if std == 0:
        return np.zeros_like(data, dtype=np.float64)

    result = (data - mean) / std
    result[~mask_bool] = 0.0
    return result.astype(np.float64)


def compute_zscore(
    in_file: str | Path,
    mask_file: str | Path,
    out_file: str | Path | None = None,
) -> Path:
    """Load NIfTI files, z-score within mask, and write the result to disk.

    Thin wrapper around ``zscore()`` that handles file I/O.

    Args:
        in_file: Path to 3D NIfTI derivative map.
        mask_file: Path to 3D binary brain mask in the same space.
        out_file: Output path.  Defaults to ``<in_file stem>_zscored.nii.gz``.

    Returns:
        Path to the output NIfTI file.
    """
    import nibabel as nib

    in_file = Path(in_file)
    img = nib.nifti1.load(in_file)
    mask = nib.nifti1.load(mask_file).get_fdata()

    zscored = zscore(img.get_fdata(), mask)

    if out_file is None:
        stem = in_file.name.split(".nii")[0]
        out_file = in_file.parent / f"{stem}_zscored.nii.gz"
    out_file = Path(out_file)

    nib.nifti1.Nifti1Image(zscored, img.affine, img.header).to_filename(
        str(out_file)
    )
    return out_file
