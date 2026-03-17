"""Convert FSL affine transforms to ITK format.

Pure-Python replacement for the c3d_affine_tool pipeline:

    c3d_affine_tool -ref <ref> -src <src> <mat> -fsl2ras -oitk <out>

The conversion follows three steps:

1. Undo FSL's internal voxel-spacing and radiological-swap conventions to
   recover a RAS world-coordinate affine.
2. Flip RAS to LPS (the coordinate system ITK uses).
3. Write the 3x3 rotation + 3 translation as an ITK ``MatrixOffsetTransformBase``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def mat_to_itk(mat: Path, reference: Path, source: Path, output: str) -> Path:
    """Convert an FSL .mat affine to ITK compatible .txt format.

    Args:
        mat: Path to the input FSL-style affine matrix (.mat).
        reference: Path to the reference (fixed) image volume.
        source: Path to the source (moving) image volume.
        output: Filename or path for the resulting ITK transformation file (.txt).

    Returns:
        ITK-compatible transformation file.
    """
    fsl_mat = np.loadtxt(mat)
    _validate_fsl_matrix(fsl_mat, mat)
    ref_img = nib.nifti1.load(reference)
    src_img = nib.nifti1.load(source)

    ras = _fsl_to_ras(fsl_mat, ref_img, src_img)
    lps = _LPS_TO_RAS @ ras @ _LPS_TO_RAS

    return _write_itk_transform(Path(output), lps)


def _write_itk_transform(path: Path, matrix: np.ndarray) -> Path:
    """Write a 4x4 LPS matrix as an ITK transform file."""
    params = np.concatenate([matrix[:3, :3].ravel(), matrix[:3, 3]])
    with path.open("w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: MatrixOffsetTransformBase_double_3_3\n")
        f.write(f"Parameters: {' '.join(f'{v:.16g}' for v in params)}\n")
        f.write("FixedParameters: 0 0 0\n")
    return path


def _validate_fsl_matrix(mat: np.ndarray, path: Path) -> None:
    """Raise ValueError if *mat* is not a valid 4x4 affine."""
    if mat.shape != (4, 4):
        msg = f"Expected 4x4 matrix, got {mat.shape} from {path}"
        raise ValueError(msg)
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(mat[3], expected_last_row):
        msg = f"Last row of affine must be [0, 0, 0, 1], got {mat[3]} from {path}"
        raise ValueError(msg)


def _fsl_swap_matrix(img: nib.Nifti1Image) -> np.ndarray:
    """FSL radiological swap matrix.

    When the image sform has a positive determinant (neurological convention),
    FSL flips the x-axis so it always works in radiological convention internally.
    """
    swap = np.eye(4)
    if np.linalg.det(img.affine) > 0:
        swap[0, 0] = -1.0
        swap[0, 3] = (img.shape[0] - 1) * img.header.get_zooms()[0]
    return swap


def _fsl_to_ras(
    fsl_mat: np.ndarray, ref: nib.Nifti1Image, src: nib.Nifti1Image
) -> np.ndarray:
    """Convert an FSL affine matrix to RAS world coordinates."""
    spc_ref = np.diag([*ref.header.get_zooms()[:3], 1.0])
    spc_src = np.diag([*src.header.get_zooms()[:3], 1.0])
    swp_ref = _fsl_swap_matrix(ref)
    swp_src = _fsl_swap_matrix(src)

    return (
        src.affine
        @ np.linalg.inv(spc_src)
        @ swp_src
        @ np.linalg.inv(fsl_mat)
        @ swp_ref
        @ spc_ref
        @ np.linalg.inv(ref.affine)
    )
