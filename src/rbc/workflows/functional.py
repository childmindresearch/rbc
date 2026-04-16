"""Functional preprocessing workflow.

Chains the full functional stream and returns all output paths as a
:class:`FunctionalOutputs` named tuple.  No BIDS naming or file copying
is performed here -- that responsibility belongs to the CLI layer via
:class:`~rbc.context.RunContext`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from niwrap import ants

from rbc.core.common import deoblique_and_reorient
from rbc.core.fsl2itk import mat_to_itk
from rbc.core.functional import (
    PEPolarFieldmap,
    PhaseDiffFieldmap,
    apply_motion_transforms,
    apply_regression,
    apply_regression_bandpass,
    bandpass_regressor_file,
    bold_masking,
    compute_regressors,
    coregister_bold_to_t1w,
    correct_distortion_pepolar,
    correct_distortion_phasediff,
    despike_bold,
    extract_motion_reference,
    fsl_motion_correction,
    resample_bold_to_template,
    slice_timing_correction,
    truncate_trs,
)
from rbc.core.niwrap import generate_exec_folder
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal

    from rbc.core.functional.nuisance import (
        ApplyRegressionOutputs,
        ComputeRegressorsOutputs,
    )
    from rbc.metadata import FunctionalMetadata

_logger = logging.getLogger("rbc")


class FunctionalOutputs(NamedTuple):
    """Outputs from the functional preprocessing pipeline.

    Attributes:
        reoriented_bold: Deobliqued and RPI-reoriented BOLD.
        truncated_bold: BOLD after discarding initial TRs.
        despiked_bold: Despiked BOLD timeseries.
        sbref: Motion reference (single-band reference) volume.
        distortion_corrected_ref: Distortion-corrected BOLD reference, or
            *None* if no fieldmap data was provided.
        distortion_warp: ANTs/ITK-compatible distortion warp field, or
            *None* if no fieldmap data was provided.
        stc_bold: Slice-timing corrected BOLD.
        preproc_bold: Motion-corrected + STC BOLD in native space.
        motion_params: Six-column motion parameter file.
        rms_rel: Frame-to-frame relative RMS displacement.
        rms_abs: Volume-to-reference absolute RMS displacement.
        mat_dir: Directory of per-volume motion matrices.
        bold_mask: Final binary BOLD brain mask.
        skull_stripped_bold: Skull-stripped BOLD reference.
        bold_to_anat_matrix: BOLD-to-T1w affine matrix.
        bold_to_anat_itk: BOLD-to-T1w affine in ITK format.
        template_bold: BOLD resampled to template space.
        regressed_bold: Nuisance-regressed & non-bandpassed BOLD.
        cleaned_bold: Nuisance-regressed & bandpass-filtered BOLD.
        regressor_file: Raw (unfiltered) nuisance regressor ``.1D`` file,
            as computed from native-space BOLD.  Carried forward so
            longitudinal regression can reuse it without recomputation.
        bpf_regressor_file: Bandpass-filtered nuisance regressor ``.1D``
            file, matching what ``3dTproject -bandpass`` actually applied.
            For BIDS export only.
        template_brain_mask: Brain mask warped to template space.
    """

    reoriented_bold: Path
    truncated_bold: Path
    despiked_bold: Path
    sbref: Path
    distortion_corrected_ref: Path | None
    distortion_warp: Path | None
    stc_bold: Path
    preproc_bold: Path
    motion_params: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path
    bold_mask: Path
    skull_stripped_bold: Path
    bold_to_anat_matrix: Path
    bold_to_anat_itk: Path
    template_bold: Path
    regressed_bold: dict[str, Path]
    cleaned_bold: dict[str, Path]
    regressor_file: dict[str, Path]
    bpf_regressor_file: dict[str, Path]
    template_brain_mask: Path


def _warp_mask_to_template(mask: Path, reference: Path, transform: Path) -> Path:
    """Warp a single tissue mask to template space."""
    out_name = f"{mask.stem.split('.')[0]}_template.nii.gz"
    result = ants.ants_apply_transforms(
        input_image=mask,
        reference_image=reference,
        transform=[ants.ants_apply_transforms_transform_file_name(transform)],
        interpolation=ants.ants_apply_transforms_nearest_neighbor(),
        dimensionality=3,
        output=ants.ants_apply_transforms_warped_output(out_name),
    )
    return result.output.output_image_outfile


def _warp_mask_to_bold_space(mask: Path, reference: Path, bold_to_anat: Path) -> Path:
    """Warp a T1w-space mask to BOLD space using the inverse of bold_to_anat."""
    out_name = f"{mask.stem.split('.')[0]}_bold.nii.gz"
    result = ants.ants_apply_transforms(
        input_image=mask,
        reference_image=reference,
        transform=[ants.ants_apply_transforms_use_inverse(bold_to_anat)],
        interpolation=ants.ants_apply_transforms_nearest_neighbor(),
        dimensionality=3,
        output=ants.ants_apply_transforms_warped_output(out_name),
    )
    return result.output.output_image_outfile


def single_session_preprocess(
    in_bold: Path,
    t1w_brain: Path,
    wm_bbr_mask: Path,
    brain_mask: Path,
    csf_mask: Path,
    wm_mask: Path,
    anat_to_template: Path,
    metadata: FunctionalMetadata,
    start_tr: int = 2,
    regressor_set: Sequence[Literal["36-parameter", "aCompCor"]] = ("36-parameter",),
    fieldmap: PhaseDiffFieldmap | PEPolarFieldmap | None = None,
    func_template: Path = REGISTRATION_TEMPLATES.brain_2mm,
    func_template_mask: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
    func_template_ref: Path = REGISTRATION_TEMPLATES.bold_ref,
) -> FunctionalOutputs:
    """Run the full functional preprocessing pipeline for one session.

    Pipeline steps:

    1.  Deoblique & reorient BOLD to RPI.
    2.  Truncate initial TRs (default: first 2 TRs).
    3.  Despike BOLD.
    4.  Extract motion reference from despiked BOLD.
    5.  Susceptibility distortion correction (optional, requires fieldmaps).
    6.  Motion correction on despiked BOLD (pre-STC, for .par estimates
        and per-volume .mat affines).
    7.  Slice timing correction on despiked BOLD.
    8.  Apply motion .mat affines to STC BOLD (native-space MC + STC,
        used for nuisance regressor computation).
    9.  BOLD brain masking on motion reference volume.
    10. BBR coregistration of skull-stripped BOLD reference to T1w.
    11. Warp tissue masks (brain, CSF, WM) to BOLD space using inverse
        of BOLD-to-T1w affine.
    12. Compute nuisance regressors from native-space BOLD.
    13. Single-step resampling of STC BOLD to template space: motion +
        BBR + T1w-to-template composite warp applied in one interpolation pass
        per volume to minimize resampling artifacts.
    14. Warp brain mask to template space.
    15. Nuisance regression without bandpass on template-space BOLD
        (full frequency range preserved for ALFF/fALFF computation).
    16. Nuisance regression with simultaneous bandpass filtering
        on template-space BOLD (Hallquist 2013).
    17. Export bandpass-filtered regressors.

    Args:
        in_bold: Raw BOLD timeseries to preprocess.
        t1w_brain: Skull-stripped T1w brain from anatomical pipeline.
        wm_bbr_mask: WM boundary mask for BBR coregistration.
        brain_mask: Binary brain mask from anatomical pipeline.
        csf_mask: CSF tissue mask from anatomical pipeline.
        wm_mask: WM tissue mask from anatomical pipeline.
        anat_to_template: T1w-to-template composite warp.
        metadata: Validated BOLD metadata (TR, slice timing).
        start_tr: Number of initial TRs to discard.
        regressor_set: Nuisance regressor strategy.
        fieldmap: Fieldmap inputs for susceptibility distortion correction.
            Pass a :class:`PhaseDiffFieldmap` for B0 fieldmap correction or a
            :class:`PEPolarFieldmap` for opposite phase-encoding correction.
            *None* skips distortion correction.
        func_template: Brain template for functional resampling (default: MNI152 2 mm).
        func_template_mask: Brain mask for functional masking (default: MNI152 2 mm).
        func_template_ref: BOLD reference image for functional masking.

    Returns:
        All output paths bundled in a :class:`FunctionalOutputs` tuple.
    """
    # 1. Deoblique & reorient
    _logger.info("Deoblique and reorient BOLD")
    reoriented = deoblique_and_reorient(in_file=in_bold)

    # 2. Truncate TRs
    _logger.info("Truncating first %d TRs", start_tr)
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=start_tr)

    # 3. Despike truncated BOLD
    _logger.info("Despiking BOLD")
    despiked = despike_bold(in_file=truncated)

    # 4. Extract motion reference from despiked BOLD
    motion_ref = extract_motion_reference(in_file=despiked)

    # 5. Distortion correction (optional)
    distortion = None
    if isinstance(fieldmap, PhaseDiffFieldmap):
        _logger.info("Susceptibility distortion correction (phase-diff)")
        distortion = correct_distortion_phasediff(
            bold_ref=motion_ref,
            magnitude=fieldmap.magnitude,
            delta_te=fieldmap.delta_te,
            effective_echo_spacing=fieldmap.effective_echo_spacing,
            pe_direction=fieldmap.pe_direction,
            phasediff=fieldmap.phasediff,
            phase1=fieldmap.phase1,
            phase2=fieldmap.phase2,
        )
    elif isinstance(fieldmap, PEPolarFieldmap):
        _logger.info("Susceptibility distortion correction (PE-polar)")
        distortion = correct_distortion_pepolar(
            bold_ref=motion_ref,
            epi_forward=fieldmap.epi_forward,
            epi_reverse=fieldmap.epi_reverse,
            readout_time_forward=fieldmap.readout_time_forward,
            readout_time_reverse=fieldmap.readout_time_reverse,
            pe_direction=fieldmap.pe_direction,
            topup_config=fieldmap.topup_config,
        )

    effective_ref = distortion.corrected_ref if distortion else motion_ref
    distortion_warp = distortion.warp_field if distortion else None

    # 6. MC on despiked BOLD (pre-STC)
    _logger.info("Motion correction (MCFLIRT)")
    # .par motion estimates used for nuisance regression + QC
    # .mat per-volume affines used in steps 8 and 13
    mc = fsl_motion_correction(in_file=despiked, ref_file=effective_ref)

    # 7. Slice timing correction
    if metadata.slice_timing is not None:
        _logger.info("Slice timing correction")
        st_corrected = slice_timing_correction(
            in_file=despiked,
            tr=metadata.tr,
            tpattern=metadata.slice_timing,
        )
    else:
        _logger.info(
            "Skipping slice timing correction (no SliceTiming in sidecar or header)"
        )
        st_corrected = despiked

    # 8. Apply pre-STC motion transforms to STC BOLD
    # native-space MC + STC BOLD used in step 12
    _logger.info("Applying motion transforms to STC BOLD")
    preproc_bold = apply_motion_transforms(
        stc_img=st_corrected,
        motion_mat_dir=mc.mat_dir,
        bold_ref=effective_ref,
    )

    # 9. BOLD brain masking
    _logger.info("BOLD brain masking")
    masking = bold_masking(
        bold_ref=effective_ref,
        template_mask=func_template_mask,
        template_ref=func_template_ref,
    )

    # 10. BBR coregistration
    _logger.info("BBR coregistration (BOLD to T1w)")
    bbr = coregister_bold_to_t1w(
        in_file=masking.skull_stripped_bold,
        reference=t1w_brain,
        wm_seg=wm_bbr_mask,
    )

    # 11. Warp tissue masks T1w-to-BOLD space (inverse of BOLD-to-T1w affine)
    bold2anat_fpath_str = str(generate_exec_folder("bold2anat") / "bold2anat.txt")
    bold_to_anat_itk = mat_to_itk(
        bbr.out_matrix_file, t1w_brain, masking.skull_stripped_bold, bold2anat_fpath_str
    )
    native_brain = _warp_mask_to_bold_space(brain_mask, effective_ref, bold_to_anat_itk)
    native_csf = _warp_mask_to_bold_space(csf_mask, effective_ref, bold_to_anat_itk)
    native_wm = _warp_mask_to_bold_space(wm_mask, effective_ref, bold_to_anat_itk)

    # 12. Compute regressors from motion-corrected BOLD in native space
    regressors: dict[str, ComputeRegressorsOutputs] = {}
    for regressor in regressor_set:
        _logger.info("Computing nuisance regressors (%s)", regressor)
        regressors[regressor] = compute_regressors(
            bold_file=preproc_bold,
            brain_mask_file=native_brain,
            csf_mask_file=native_csf,
            wm_mask_file=native_wm,
            motion_params=mc.motion_params,
            regressor_set=regressor,
        )

    # 13. Single-step resampling (STC BOLD to template)
    # All spatial transforms (motion + BBR + T1w-to-template) applied in one
    # interpolation pass per volume to minimize resampling artifacts.
    _logger.info("Resampling BOLD to template space (single-step)")
    template_bold = resample_bold_to_template(
        stc_bold=st_corrected,
        motion_mat_dir=mc.mat_dir,
        bold_to_anat=bbr.out_matrix_file,
        anat_to_template=anat_to_template,
        bold_ref=masking.skull_stripped_bold,
        template=func_template,
        t1w_brain=t1w_brain,
        distortion_warp=distortion_warp,
    )

    # 14. Warp brain mask to template space (needed for regression + bandpass)
    tmpl_brain = _warp_mask_to_template(brain_mask, func_template, anat_to_template)

    regression: dict[str, ApplyRegressionOutputs] = {}
    cleaned: dict[str, ApplyRegressionOutputs] = {}
    raw_regressors: dict[str, Path] = {}
    filtered_regressors: dict[str, Path] = {}
    for regressor in regressor_set:
        # 15. Nuisance regression without bandpass (pre-bandpass residuals
        #     for ALFF/fALFF computation, where full frequency range matters)
        _logger.info("%s nuisance regression (no bandpass)", regressor)
        regression[regressor] = apply_regression(
            bold_file=template_bold,
            brain_mask_file=tmpl_brain,
            regressor_file=regressors[regressor].regressor_file,
        )

        # 16. Simultaneous regression + bandpass filtering (Hallquist 2013).
        #     Regressors are filtered to the same passband before projection,
        #     preventing re-introduction of removed frequencies.
        _logger.info("%s nuisance regression + bandpass filtering", regressor)
        cleaned[regressor] = apply_regression_bandpass(
            bold_file=template_bold,
            brain_mask_file=tmpl_brain,
            regressor_file=regressors[regressor].regressor_file,
        )

        # 17a. Carry raw (unfiltered) regressors forward for longitudinal reuse
        raw_regressors[regressor] = regressors[regressor].regressor_file

        # 17b. Export bandpass-filtered regressors (matches what 3dTproject
        #      actually applied)
        filtered_regressors[regressor] = bandpass_regressor_file(
            regressors[regressor].regressor_file,
            tr=metadata.tr,
            f_low=0.01,
            f_high=0.1,
        )

    return FunctionalOutputs(
        reoriented_bold=reoriented.out_file,
        truncated_bold=truncated,
        despiked_bold=despiked,
        sbref=motion_ref,
        distortion_corrected_ref=distortion.corrected_ref if distortion else None,
        distortion_warp=distortion_warp,
        stc_bold=st_corrected,
        preproc_bold=preproc_bold,
        motion_params=mc.motion_params,
        rms_rel=mc.rms_rel,
        rms_abs=mc.rms_abs,
        mat_dir=mc.mat_dir,
        bold_mask=masking.final_mask,
        skull_stripped_bold=masking.skull_stripped_bold,
        bold_to_anat_matrix=bbr.out_matrix_file,
        bold_to_anat_itk=bold_to_anat_itk,
        template_bold=template_bold,
        regressed_bold={r: regression[r].regressed_bold for r in regressor_set},
        cleaned_bold={r: cleaned[r].regressed_bold for r in regressor_set},
        regressor_file=raw_regressors,
        bpf_regressor_file=filtered_regressors,
        template_brain_mask=tmpl_brain,
    )
