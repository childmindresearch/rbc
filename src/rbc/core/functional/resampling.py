"""Resampling utilities for BOLD timeseries.

Three public entry points:

- :func:`apply_motion_transforms`: per-volume mcflirt affines applied to
  an STC timeseries (desc-preproc_bold derivative; not consumed
  downstream).
- :func:`resample_bold_to_template`: single-step BOLD->template
  resampling that fuses motion + BBR + anat-to-template (+ optional
  distortion) into one interpolation pass per volume.
- :func:`resample_image`: 3D/4D resampling through a single composite
  displacement field; used by the longitudinal pipeline.

The static (time-invariant) parts of the transform chain are composed
into a single template-grid RAS coordinate map up front; the per-volume
loop then applies only the motion affine and samples with
``scipy.ndimage.map_coordinates``. This is conceptually similar to
``nitransforms.resampling.apply`` with ``serialize_nvols=1`` but lets
us cache the static coord map once across all volumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import nitransforms as nt
import numpy as np
from nitransforms.base import ImageGrid
from nitransforms.nonlinear import DenseFieldTransform
from scipy import ndimage as ndi

from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from pathlib import Path


# Interpolation defaults shared by both resampling paths. Order 3 is the
# scipy.ndimage equivalent of the cubic spline used by fMRIPrep.
_INTERP_ORDER = 3
_INTERP_MODE = "constant"
_INTERP_CVAL = 0.0


def _load_ants_warp(path: Path) -> DenseFieldTransform:
    """Load an ANTs/ITK NIfTI displacement field as a DenseFieldTransform.

    Delegates to nitransforms' ``ITKDisplacementsField``, which enforces
    the canonical 5D ``(X, Y, Z, 1, 3)`` ANTs shape and handles the LPS
    -> RAS sign flip.
    """
    return DenseFieldTransform.from_filename(str(path), fmt="itk")


def _ras_to_voxel_grid(
    coords_ras: np.ndarray, affine_inv: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Convert (N, 3) RAS coordinates to a (3, *shape) voxel-coord grid."""
    n = coords_ras.shape[0]
    homog = np.empty((4, n), dtype=np.float64)
    homog[:3] = coords_ras.T
    homog[3] = 1.0
    voxel = (affine_inv @ homog)[:3]
    return voxel.reshape((3, *shape))


def _resample_4d(
    src_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image,
    static_coords_ras: np.ndarray,
    out_path: Path,
    motion_xfms: list[nt.linear.Affine] | None = None,
    order: int = _INTERP_ORDER,
) -> Path:
    """Resample a 3D or 4D image volume-by-volume into reference space.

    For each output volume, applies the (optional) per-volume motion-affine
    inverse to the static RAS coordinate map and samples with
    ``ndi.map_coordinates``. Output dimensionality matches the input
    (3D in, 3D out; 4D in, 4D out). The source's TR is preserved for 4D
    outputs; the spatial header comes from *reference_img*.

    Args:
        src_img: 3D or 4D source image.
        reference_img: Image whose grid defines the output space.
        static_coords_ras: ``(N, 3)`` RAS coordinates obtained by pulling
            the reference grid through every static transform; ``N`` must
            equal ``prod(reference_img.shape[:3])``.
        out_path: Destination NIfTI path.
        motion_xfms: Optional per-volume forward (volume->reference)
            affines. If ``None``, no per-volume affine is composed in.
        order: Spline interpolation order (0-5); default cubic.

    Returns:
        Path to the written NIfTI.
    """
    ref_shape = reference_img.shape[:3]
    is_4d = src_img.ndim == 4
    n_vols = src_img.shape[3] if is_4d else 1
    src_affine_inv = np.linalg.inv(src_img.affine).astype(np.float64)

    if motion_xfms is not None and len(motion_xfms) != n_vols:
        raise ValueError(
            f"Count mismatch: ({len(motion_xfms)}) motion mats, ({n_vols}) volumes"
        )

    static_voxel_grid: np.ndarray | None = None
    if motion_xfms is None:
        static_voxel_grid = _ras_to_voxel_grid(
            static_coords_ras, src_affine_inv, ref_shape
        )

    out_data = np.zeros((*ref_shape, n_vols), dtype=np.float32)
    dataobj = src_img.dataobj

    for t in range(n_vols):
        if motion_xfms is not None:
            vol_coords_ras = motion_xfms[t].map(static_coords_ras, inverse=True)
            voxel_grid = _ras_to_voxel_grid(
                vol_coords_ras, src_affine_inv, ref_shape
            )
        else:
            assert static_voxel_grid is not None  # noqa: S101
            voxel_grid = static_voxel_grid

        vol_data = np.asanyarray(
            dataobj[..., t] if is_4d else dataobj, dtype=np.float32
        )
        ndi.map_coordinates(
            vol_data,
            voxel_grid,
            output=out_data[..., t],
            order=order,
            mode=_INTERP_MODE,
            cval=_INTERP_CVAL,
            prefilter=True,
        )

    final_data = out_data if is_4d else out_data[..., 0]
    out_img = nib.Nifti1Image(final_data, reference_img.affine)
    if is_4d:
        zooms = (
            reference_img.header.get_zooms()[:3] + src_img.header.get_zooms()[3:4]
        )
    else:
        zooms = reference_img.header.get_zooms()[:3]
    out_img.header.set_zooms(zooms)
    nib.save(out_img, out_path)
    return out_path


def resample_image(
    src: Path, reference: Path, warp: Path, order: int = _INTERP_ORDER
) -> Path:
    """Resample a 3D or 4D image into *reference* space via an ANTs warp.

    Loads *warp* as an ANTs composite displacement field (the pull mapping
    from reference to source) and applies it to every volume of *src*.

    Args:
        src: 3D or 4D source image to be resampled.
        reference: Image whose grid defines the output space.
        warp: ANTs/ITK composite displacement field NIfTI.
        order: Spline interpolation order (0-5); default cubic.

    Returns:
        Path to the resampled NIfTI in *reference* space.
    """
    src_img = nib.nifti1.load(src)
    ref_img = nib.nifti1.load(reference)
    warp_xfm = _load_ants_warp(warp)

    ref_coords_ras = ImageGrid(ref_img).ndcoords.astype(np.float64)
    src_coords_ras = warp_xfm.map(ref_coords_ras)

    out_path = generate_exec_folder("resample_image") / "resampled.nii.gz"
    return _resample_4d(
        src_img=src_img,
        reference_img=ref_img,
        static_coords_ras=src_coords_ras,
        out_path=out_path,
        order=order,
    )


def _load_motion_xfms(
    motion_mat_dir: Path, bold_ref_img: nib.Nifti1Image
) -> list[nt.linear.Affine]:
    """Load mcflirt per-volume motion affines from a directory of .mat files."""
    motion_mats = sorted(motion_mat_dir.glob("MAT_*"))
    if not motion_mats:
        raise FileNotFoundError(f"No motion .mat files found in {motion_mat_dir}")
    return [
        nt.linear.load(
            str(m), fmt="fsl", reference=bold_ref_img, moving=bold_ref_img
        )
        for m in motion_mats
    ]


def apply_motion_transforms(
    stc_img: Path,
    motion_mat_dir: Path,
    bold_ref: Path,
) -> Path:
    """Apply mcflirt motion affines to STC volumes (desc-preproc_bold).

    Produces a motion-corrected + slice-timing corrected BOLD in native
    space. Motion was estimated on pre-STC (despiked) data; here the
    resulting per-volume .mat affines are applied to the STC timeseries.

    The output is exported as desc-preproc_bold but is not consumed by
    any downstream workflow step. Template-space output uses
    :func:`resample_bold_to_template` instead (single interpolation pass).

    Args:
        stc_img: Slice-timing corrected 4D BOLD timeseries.
        motion_mat_dir: Directory of per-volume MAT_* matrices from mcflirt.
        bold_ref: BOLD reference volume (used both as FSL reference and as
            the output spatial grid).

    Returns:
        4D BOLD (MC + STC) in native space.

    Raises:
        FileNotFoundError: No motion .mat files are found in the specified directory.
        ValueError: Number of motion matrix files does not match the number
            of slice-timing corrected volumes.
    """
    src_img = nib.nifti1.load(stc_img)
    bold_ref_img = nib.nifti1.load(bold_ref)
    motion_xfms = _load_motion_xfms(motion_mat_dir, bold_ref_img)
    static_coords_ras = ImageGrid(bold_ref_img).ndcoords.astype(np.float64)

    out_path = generate_exec_folder("preproc_bold_merge") / "preproc_bold.nii.gz"
    return _resample_4d(
        src_img=src_img,
        reference_img=bold_ref_img,
        static_coords_ras=static_coords_ras,
        motion_xfms=motion_xfms,
        out_path=out_path,
    )


def resample_bold_to_template(
    stc_bold: Path,
    motion_mat_dir: Path,
    bold_to_anat: Path,
    anat_to_template: Path,
    bold_ref: Path,
    template: Path,
    t1w_brain: Path,
    distortion_warp: Path | None = None,
) -> Path:
    """Single-step resampling of STC BOLD to template space.

    Composes the static transforms (anat-to-template warp, BBR affine,
    optional distortion warp) into a single template-grid RAS coordinate
    map, then loops over volumes applying the per-volume motion affine
    and sampling with cubic spline interpolation.

    Args:
        stc_bold: Slice-timing corrected 4D BOLD timeseries.
        motion_mat_dir: Directory of per-volume MAT_* matrices from mcflirt.
        bold_to_anat: BOLD to T1w FSL affine matrix (BBR output).
        anat_to_template: T1w to template ANTs composite displacement field.
        bold_ref: BOLD reference volume (FSL ref/moving for affines).
        template: Brain template in target space.
        t1w_brain: Skull-stripped T1w brain (FSL reference for bold_to_anat).
        distortion_warp: Optional ANTs/ITK displacement field on the BOLD grid.

    Returns:
        Resampled 4D BOLD in template space.

    Raises:
        FileNotFoundError: No motion .mat files found in the directory.
        ValueError: Number of motion matrices does not match STC volumes.
    """
    src_img = nib.nifti1.load(stc_bold)
    template_img = nib.nifti1.load(template)
    bold_ref_img = nib.nifti1.load(bold_ref)
    t1w_img = nib.nifti1.load(t1w_brain)

    if src_img.ndim != 4:
        raise ValueError(f"Expected 4D STC BOLD, got shape {src_img.shape}")

    # Fail fast before loading the (slow) static warps.
    motion_xfms = _load_motion_xfms(motion_mat_dir, bold_ref_img)
    n_vols = src_img.shape[3]
    if len(motion_xfms) != n_vols:
        raise ValueError(
            f"Count mismatch: ({len(motion_xfms)}) mats, ({n_vols}) volumes"
        )

    bold_to_anat_xfm = nt.linear.load(
        str(bold_to_anat), fmt="fsl", reference=t1w_img, moving=bold_ref_img
    )
    anat_to_tpl_xfm = _load_ants_warp(anat_to_template)
    distortion_xfm = (
        _load_ants_warp(distortion_warp) if distortion_warp is not None else None
    )

    # Static pull chain: template_ras -> anat_ras -> bold_undistorted_ras
    # (-> bold_distorted_ras if distortion).
    tpl_coords_ras = ImageGrid(template_img).ndcoords.astype(np.float64)
    anat_coords_ras = anat_to_tpl_xfm.map(tpl_coords_ras)
    bold_coords_ras = bold_to_anat_xfm.map(anat_coords_ras, inverse=True)
    if distortion_xfm is not None:
        bold_coords_ras = distortion_xfm.map(bold_coords_ras)

    out_path = (
        generate_exec_folder("bold_to_template_merge")
        / "bold_to_template_resampled.nii.gz"
    )
    return _resample_4d(
        src_img=src_img,
        reference_img=template_img,
        static_coords_ras=bold_coords_ras,
        motion_xfms=motion_xfms,
        out_path=out_path,
    )
