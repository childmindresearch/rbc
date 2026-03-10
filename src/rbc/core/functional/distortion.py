"""Susceptibility distortion correction for functional MRI.

Provides two correction strategies based on the available fieldmap data:

- **Phase-difference B0 fieldmap** (FSL FUGUE): Uses a magnitude image and
  either a pre-computed phase-difference map or two individual phase images
  to estimate and correct the B0 inhomogeneity field.

- **Opposite phase-encoding / PEPOLAR** (FSL TOPUP): Uses a pair of
  spin-echo EPI images acquired with reversed phase-encoding directions
  (e.g. AP/PA) to estimate the susceptibility-induced distortion field.

Both methods produce an ITK/ANTs-compatible displacement field that can be
composed with other transforms for single-step resampling to template space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, TypeGuard, cast

import nibabel as nib
import numpy as np
from niwrap import fsl
from scipy.ndimage import binary_erosion

from rbc.core.functional.resampling import merge_3d_to_4d
from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from pathlib import Path

BidsPhaseEncoding = Literal["i", "i-", "j", "j-", "k", "k-"]
FslDirection = Literal["x", "x-", "y", "y-", "z", "z-"]


class PhasediffFieldmap(NamedTuple):
    """Inputs for phase-difference B0 fieldmap distortion correction.

    Attributes:
        magnitude: Magnitude image corresponding to the fieldmap.
        delta_te: Echo time difference in milliseconds.
        effective_echo_spacing: EPI dwell time in seconds.
        pe_direction: BIDS phase-encoding axis (e.g. ``"j"``, ``"j-"``).
        phasediff: Pre-computed phase-difference image.
        phase1: Phase image at TE1 (alternative to *phasediff*).
        phase2: Phase image at TE2 (alternative to *phasediff*).
    """

    magnitude: Path
    delta_te: float
    effective_echo_spacing: float
    pe_direction: BidsPhaseEncoding
    phasediff: Path | None = None
    phase1: Path | None = None
    phase2: Path | None = None


class PepolarFieldmap(NamedTuple):
    """Inputs for opposite phase-encoding (PEPOLAR) distortion correction.

    Attributes:
        epi_ap: EPI image in the primary phase-encoding direction.
        epi_pa: EPI image in the reversed phase-encoding direction.
        readout_time_ap: Total readout time for *epi_ap* (seconds).
        readout_time_pa: Total readout time for *epi_pa* (seconds).
        pe_direction: BIDS phase-encoding axis of the primary EPI
            (e.g. ``"j"`` for AP).
        topup_config: Optional TOPUP configuration file.
    """

    epi_ap: Path
    epi_pa: Path
    readout_time_ap: float
    readout_time_pa: float
    pe_direction: BidsPhaseEncoding
    topup_config: Path | None = None


class DistortionCorrectionOutputs(NamedTuple):
    """Outputs from susceptibility distortion correction.

    Attributes:
        corrected_ref: Distortion-corrected BOLD reference volume.
        warp_field: ANTs/ITK-compatible displacement field for single-step
            resampling.
    """

    corrected_ref: Path
    warp_field: Path


# ---------------------------------------------------------------------------
# Phase-encoding direction mappings
# ---------------------------------------------------------------------------

_PE_TO_FUGUE: dict[str, str] = {
    "i": "x",
    "i-": "x-",
    "j": "y",
    "j-": "y-",
    "k": "z",
    "k-": "z-",
}
"""Map BIDS phase-encoding axis labels to FUGUE unwarp direction codes."""

_PE_DIR_VECTORS: dict[str, tuple[int, int, int]] = {
    "i": (1, 0, 0),
    "i-": (-1, 0, 0),
    "j": (0, 1, 0),
    "j-": (0, -1, 0),
    "k": (0, 0, 1),
    "k-": (0, 0, -1),
}
"""Map BIDS phase-encoding axis to unit direction vectors for TOPUP acqparams."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_valid_pe_direction(pe_direction: str) -> TypeGuard[BidsPhaseEncoding]:
    """Check whether *pe_direction* is a valid BIDS phase-encoding axis."""
    return pe_direction in _PE_TO_FUGUE


def _validate_pe_direction(pe_direction: str) -> None:
    """Raise ``ValueError`` if *pe_direction* is not a valid BIDS PE axis."""
    if not is_valid_pe_direction(pe_direction):
        raise ValueError(
            f"Invalid pe_direction '{pe_direction}'. "
            f"Must be one of {sorted(_PE_TO_FUGUE)}"
        )


def _write_acqparams(
    pe_dirs: list[str],
    readout_times: list[float],
    epi_files: list[Path],
    output: Path,
) -> Path:
    """Write a TOPUP-compatible acquisition parameters file.

    Each EPI volume gets one line: ``<dx> <dy> <dz> <readout_time>``.
    The number of volumes per EPI is detected via nibabel.

    Args:
        pe_dirs: Phase-encoding direction per EPI file (BIDS axis labels).
        readout_times: Total readout time per EPI file (seconds).
        epi_files: Paths to the EPI NIfTI files.
        output: Path to write the acqparams text file.

    Returns:
        Path to the written acqparams file.
    """
    lines: list[str] = []
    for pe_dir, readout, epi_path in zip(
        pe_dirs, readout_times, epi_files, strict=True
    ):
        _validate_pe_direction(pe_dir)
        vec = _PE_DIR_VECTORS[pe_dir]
        img = nib.nifti1.load(epi_path)
        n_vols = img.shape[3] if img.ndim == 4 else 1
        line = f"{vec[0]} {vec[1]} {vec[2]} {readout}"
        lines.extend([line] * n_vols)

    output.write_text("\n".join(lines) + "\n")
    return output


def _pe_axis_index(pe_direction: str) -> int:
    """Return the voxel axis index (0=i, 1=j, 2=k) for a PE direction."""
    return {"i": 0, "i-": 0, "j": 1, "j-": 1, "k": 2, "k-": 2}[pe_direction]


def _pe_sign(pe_direction: str) -> int:
    """Return +1 or -1 for the PE direction polarity."""
    return -1 if pe_direction.endswith("-") else 1


def _shiftmap_to_itk_warp(
    shiftmap: Path,
    pe_direction: str,
    output: Path,
) -> Path:
    """Convert a FUGUE pixel-shift map to an ITK displacement field.

    FUGUE shift maps contain per-voxel shifts (in voxels) along the PE
    axis.  This converts to a 3-component mm displacement field in LPS
    (ANTs/ITK convention).

    Args:
        shiftmap: FUGUE shift map NIfTI (voxel units, single component).
        pe_direction: BIDS phase-encoding direction.
        output: Path to write the ITK warp.

    Returns:
        Path to the ITK-compatible displacement field.
    """
    shift_img = nib.nifti1.load(shiftmap)
    shift_data = np.asarray(shift_img.dataobj, dtype=np.float32)
    voxel_sizes = np.array(shift_img.header.get_zooms()[:3])

    axis = _pe_axis_index(pe_direction)

    # Build 3-component displacement field (mm, RAS)
    warp = np.zeros((*shift_data.shape[:3], 3), dtype=np.float32)
    warp[..., axis] = shift_data * voxel_sizes[axis]

    # RAS → LPS for ANTs
    warp[..., 0] *= -1
    warp[..., 1] *= -1

    header = shift_img.header.copy()
    header.set_intent(1007)  # NIFTI_INTENT_VECTOR
    header.set_data_dtype(np.float32)

    nib.save(nib.Nifti1Image(warp, shift_img.affine, header), output)
    return output


def _fieldmap_hz_to_itk_warp(
    fieldmap_hz: Path,
    pe_direction: str,
    readout_time: float,
    output: Path,
) -> Path:
    """Convert a TOPUP Hz fieldmap to an ITK displacement field.

    Displacement along the PE axis in mm equals:
    ``fieldmap_hz * readout_time * voxel_size[pe_axis]``.

    Args:
        fieldmap_hz: TOPUP field output in Hz.
        pe_direction: BIDS phase-encoding direction.
        readout_time: Total readout time (seconds).
        output: Path to write the ITK warp.

    Returns:
        Path to the ITK-compatible displacement field.
    """
    fmap_img = nib.nifti1.load(fieldmap_hz)
    fmap_data = np.asarray(fmap_img.dataobj, dtype=np.float32)
    # TOPUP --fout can be 4D (one field per input volume); take the first
    if fmap_data.ndim == 4:
        fmap_data = fmap_data[..., 0]
    voxel_sizes = np.array(fmap_img.header.get_zooms()[:3])

    axis = _pe_axis_index(pe_direction)
    sign = _pe_sign(pe_direction)

    warp = np.zeros((*fmap_data.shape[:3], 3), dtype=np.float32)
    warp[..., axis] = fmap_data * readout_time * voxel_sizes[axis] * sign

    # RAS -> LPS for ANTs
    warp[..., 0] *= -1
    warp[..., 1] *= -1

    header = fmap_img.header.copy()
    header.set_intent(1007)
    header.set_data_dtype(np.float32)

    nib.save(nib.Nifti1Image(warp, fmap_img.affine, header), output)
    return output


def _opposite_pe(pe_direction: str) -> str:
    """Return the opposite phase-encoding direction."""
    if pe_direction.endswith("-"):
        return pe_direction[:-1]
    return pe_direction + "-"


# ---------------------------------------------------------------------------
# Phase-difference fieldmap correction (FSL FUGUE)
# ---------------------------------------------------------------------------


def correct_distortion_phasediff(
    bold_ref: Path,
    magnitude: Path,
    delta_te: float,
    effective_echo_spacing: float,
    pe_direction: BidsPhaseEncoding,
    phasediff: Path | None = None,
    phase1: Path | None = None,
    phase2: Path | None = None,
) -> DistortionCorrectionOutputs:
    """Correct susceptibility distortion using a phase-difference B0 fieldmap.

    Uses FSL ``fsl_prepare_fieldmap`` to create a fieldmap in rad/s, then
    ``fugue`` to unwarp the BOLD reference and produce a shift map, which is
    converted to an ANTs/ITK-compatible displacement field.

    Either *phasediff* or both *phase1* and *phase2* must be provided.

    Args:
        bold_ref: BOLD reference volume to correct.
        magnitude: Magnitude image corresponding to the fieldmap.
        delta_te: Echo time difference in milliseconds.
        effective_echo_spacing: EPI dwell time in seconds.
        pe_direction: BIDS phase-encoding axis (e.g. ``"j"``, ``"j-"``).
        phasediff: Pre-computed phase-difference image.
        phase1: Phase image at TE1.
        phase2: Phase image at TE2.

    Returns:
        Corrected reference and ITK-compatible warp field.

    Raises:
        ValueError: If neither phasediff nor both phase images are provided,
            or if pe_direction is invalid.
    """
    _validate_pe_direction(pe_direction)

    if phasediff is None and (phase1 is None or phase2 is None):
        raise ValueError(
            "Must provide either 'phasediff' or both 'phase1' and 'phase2'."
        )

    out_dir = generate_exec_folder(suffix="distortion_phasediff")

    # 1. Skull-strip magnitude
    bet_result = fsl.bet(
        infile=magnitude,
        fractional_intensity=0.5,
        binary_mask=True,
        maskfile="magnitude_bet",
    )

    # 2. Erode magnitude mask (scipy - avoids an FSL container call)
    mask_img = nib.nifti1.load(bet_result.binary_mask)
    eroded_data = binary_erosion(np.asarray(mask_img.dataobj) > 0).astype(np.uint8)
    eroded_mask_path = out_dir / "magnitude_mask_ero.nii.gz"
    nib.save(
        nib.Nifti1Image(eroded_data, mask_img.affine, mask_img.header),
        eroded_mask_path,
    )

    # 3. If phase1+phase2 provided, subtract to get phasediff
    if phasediff is None:
        assert phase1 is not None  # noqa: S101
        assert phase2 is not None  # noqa: S101
        p1_img = nib.nifti1.load(phase1)
        p2_img = nib.nifti1.load(phase2)
        diff_data = np.asarray(p1_img.dataobj, dtype=np.float32) - np.asarray(
            p2_img.dataobj, dtype=np.float32
        )
        diff_img = nib.Nifti1Image(
            diff_data, affine=p1_img.affine, header=p1_img.header
        )
        phasediff = out_dir / "phasediff.nii.gz"
        nib.save(diff_img, phasediff)

    # 4. Prepare fieldmap (rad/s)
    fieldmap = fsl.fsl_prepare_fieldmap(
        scanner="SIEMENS",
        phase_image=phasediff,
        magnitude_image=bet_result.outfile,
        out_image="fieldmap_rads",
        delta_te=delta_te,
    )

    # 5. Create fieldmap mask: nonzero fieldmap voxels ∩ eroded magnitude mask
    fmap_img = nib.nifti1.load(fieldmap.output_fieldmap)
    fmap_data = np.asarray(fmap_img.dataobj, dtype=np.float32)
    fmap_mask_data = ((np.abs(fmap_data) > 0) & (eroded_data > 0)).astype(np.uint8)
    fieldmap_mask_path = out_dir / "fieldmap_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(fmap_mask_data, fmap_img.affine, fmap_img.header),
        fieldmap_mask_path,
    )

    # 6. Unwarp bold_ref + get shift map
    fugue_dir = cast("FslDirection", _PE_TO_FUGUE[pe_direction])
    fugue_result = fsl.fugue(
        in_file=bold_ref,
        fmap_in_file=fieldmap.output_fieldmap,
        mask_file=fieldmap_mask_path,
        unwarp_direction=fugue_dir,
        dwell_time=effective_echo_spacing,
        shift_out_file="shiftmap.nii.gz",
        unwarped_file="bold_ref_unwarped.nii.gz",
        despike_2dfilter=True,
    )

    assert fugue_result.unwarped_file_outfile is not None  # noqa: S101
    assert fugue_result.shift_out_file_outfile is not None  # noqa: S101

    # 7. Convert FUGUE shift map to ITK displacement field
    itk_warp = _shiftmap_to_itk_warp(
        shiftmap=fugue_result.shift_out_file_outfile,
        pe_direction=pe_direction,
        output=out_dir / "distortion_warp_itk.nii.gz",
    )

    return DistortionCorrectionOutputs(
        corrected_ref=fugue_result.unwarped_file_outfile,
        warp_field=itk_warp,
    )


# ---------------------------------------------------------------------------
# Opposite phase-encoding / PEPOLAR correction (FSL TOPUP)
# ---------------------------------------------------------------------------


def correct_distortion_pepolar(
    bold_ref: Path,
    epi_ap: Path,
    epi_pa: Path,
    readout_time_ap: float,
    readout_time_pa: float,
    pe_direction: BidsPhaseEncoding,
    topup_config: Path | None = None,
) -> DistortionCorrectionOutputs:
    """Correct susceptibility distortion using opposite phase-encoding EPIs.

    Uses FSL ``topup`` to estimate the distortion field from a pair of
    reversed phase-encoding EPI images (e.g. AP/PA), then ``applytopup``
    to correct the BOLD reference. The estimated field is converted to an
    ANTs/ITK-compatible displacement field.

    Args:
        bold_ref: BOLD reference volume to correct.
        epi_ap: EPI image in the primary phase-encoding direction.
        epi_pa: EPI image in the reversed phase-encoding direction.
        readout_time_ap: Total readout time for *epi_ap* (seconds).
        readout_time_pa: Total readout time for *epi_pa* (seconds).
        pe_direction: BIDS phase-encoding axis of the primary EPI
            (e.g. ``"j"`` for AP).
        topup_config: Optional TOPUP configuration file. If *None*,
            FSL uses its default ``b02b0.cnf``.

    Returns:
        Corrected reference and ITK-compatible warp field.

    Raises:
        ValueError: If *pe_direction* is invalid.
    """
    _validate_pe_direction(pe_direction)

    out_dir = generate_exec_folder(suffix="distortion_pepolar")
    pe_opposite = _opposite_pe(pe_direction)

    # 1. Write acquisition parameters file
    acqparams = _write_acqparams(
        pe_dirs=[pe_direction, pe_opposite],
        readout_times=[readout_time_ap, readout_time_pa],
        epi_files=[epi_ap, epi_pa],
        output=out_dir / "acqparams.txt",
    )

    # 2. Merge AP/PA into 4D
    merged_path = merge_3d_to_4d(
        volumes=[epi_ap, epi_pa],
        output=out_dir / "merged_epi.nii.gz",
    )

    # 3. Estimate field with TOPUP
    topup_result = fsl.topup(
        imain=merged_path,
        datain=acqparams,
        out="topup_out",
        fout="topup_field",
        iout="topup_corrected",
        config=topup_config,
    )

    assert topup_result.fieldcoef is not None  # noqa: S101
    assert topup_result.fout is not None  # noqa: S101

    # 4. Apply correction to bold_ref
    applytopup_result = fsl.applytopup(
        imain=[bold_ref],
        datain=acqparams,
        inindex=["1"],
        topup=topup_result.fieldcoef.parent / "topup_out",
        out="bold_ref_corrected",
        method="jac",
    )

    # 5. Convert TOPUP Hz fieldmap to ITK displacement field
    itk_warp = _fieldmap_hz_to_itk_warp(
        fieldmap_hz=topup_result.fout,
        pe_direction=pe_direction,
        readout_time=readout_time_ap,
        output=out_dir / "distortion_warp_itk.nii.gz",
    )

    return DistortionCorrectionOutputs(
        corrected_ref=applytopup_result.output_file,
        warp_field=itk_warp,
    )
