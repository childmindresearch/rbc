"""Functional preprocessing workflow.

Chains the full functional stream and returns all output paths as a
:class:`FunctionalOutputs` named tuple.  No BIDS naming or file copying
is performed here -- that responsibility belongs to the CLI layer via
:class:`~rbc.context.PipelineContext`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from bids2table import load_bids_metadata
from niwrap import ants

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    bold_masking,
    coregister_bold_to_t1w,
    despike_bold,
    extract_motion_reference,
    fsl_motion_correction,
    nuisance_regression,
    resample_bold_to_template,
    slice_timing_correction,
    truncate_trs,
)
from rbc.core.longitudinal.transform import compose_transform, func_transform
from rbc_resources import MNI_TEMPLATES

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal


class FunctionalOutputs(NamedTuple):
    """Outputs from the functional preprocessing pipeline.

    Attributes:
        reoriented_bold: Deobliqued and RPI-reoriented BOLD.
        truncated_bold: BOLD after discarding initial TRs.
        stc_bold: Slice-timing corrected BOLD.
        despiked_bold: Despiked BOLD timeseries.
        sbref: Motion reference (single-band reference) volume.
        motion_corrected_bold: Motion-corrected BOLD.
        motion_params: Six-column motion parameter file.
        rms_rel: Frame-to-frame relative RMS displacement.
        rms_abs: Volume-to-reference absolute RMS displacement.
        mat_dir: Directory of per-volume motion matrices.
        bold_mask: Final binary BOLD brain mask.
        skull_stripped_bold: Skull-stripped BOLD reference.
        bold_to_anat_matrix: BOLD-to-T1w affine matrix (BBR).
        template_bold: BOLD resampled to template space.
        cleaned_bold: Nuisance-regressed BOLD.
        regressor_file: Nuisance regressor ``.1D`` file.
        template_brain_mask: Brain mask warped to template space.
    """

    reoriented_bold: Path
    truncated_bold: Path
    stc_bold: Path
    despiked_bold: Path
    sbref: Path
    motion_corrected_bold: Path
    motion_params: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path
    bold_mask: Path
    skull_stripped_bold: Path
    bold_to_anat_matrix: Path
    template_bold: Path
    cleaned_bold: Path
    regressor_file: Path
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


def single_session_preprocess(
    in_bold: Path,
    t1w_brain: Path,
    wm_bbr_mask: Path,
    brain_mask: Path,
    csf_mask: Path,
    wm_mask: Path,
    anat_to_template: Path,
    start_tr: int = 2,
    regressor_set: Literal["36-parameter", "aCompCor"] = "36-parameter",
    bandpass: tuple[float, float] | None = (0.01, 0.1),
) -> FunctionalOutputs:
    """Run the full functional preprocessing pipeline for one session.

    Pipeline steps:

    1.  Deoblique & reorient BOLD to RPI.
    2.  Truncate initial TRs.
    3.  Slice timing correction.
    4.  Despike the STC BOLD.
    5.  Extract motion reference from despiked STC.
    6.  Motion correction (despiked STC -> motion ref).
    7.  BOLD brain masking.
    8.  BBR coregistration (BOLD -> T1w).
    9.  Single-step resampling (despiked STC -> template).
    10. Warp tissue masks to template space.
    11. Nuisance regression in template space.

    Args:
        in_bold: Raw BOLD timeseries to preprocess.
        t1w_brain: Skull-stripped T1w brain from anatomical pipeline.
        wm_bbr_mask: WM boundary mask for BBR coregistration.
        brain_mask: Binary brain mask from anatomical pipeline.
        csf_mask: CSF tissue mask from anatomical pipeline.
        wm_mask: WM tissue mask from anatomical pipeline.
        anat_to_template: T1w -> template composite warp.
        start_tr: Number of initial TRs to discard.
        regressor_set: Nuisance regressor strategy.
        bandpass: Frequency band to retain, or *None* to skip.

    Returns:
        All output paths bundled in a :class:`FunctionalOutputs` tuple.
    """
    metadata = load_bids_metadata(in_bold)

    # 1. Deoblique & reorient
    reoriented = deoblique_and_reorient(in_file=in_bold)

    # 2. Truncate TRs
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=start_tr)

    # 3. Slice timing correction
    st_corrected = slice_timing_correction(
        in_file=truncated,
        tr=metadata.get("RepetitionTime"),
        tpattern=metadata.get("SliceTiming"),
    )

    # 4. Despike STC
    despiked = despike_bold(in_file=st_corrected)

    # 5. Extract motion reference from despiked STC
    motion_ref = extract_motion_reference(in_file=despiked)

    # 6. Motion correction on despiked STC
    mc = fsl_motion_correction(in_file=despiked, ref_file=motion_ref)

    # 7. BOLD brain masking
    masking = bold_masking(
        bold_ref=motion_ref,
        template_mask=MNI_TEMPLATES.brain_mask_2mm,
        template_ref=MNI_TEMPLATES.bold_ref,
    )

    # 8. BBR coregistration
    bbr = coregister_bold_to_t1w(
        in_file=masking.skull_stripped_bold,
        reference=t1w_brain,
        wm_seg=wm_bbr_mask,
    )

    # 9. Single-step resampling (despiked STC -> template)
    template_bold = resample_bold_to_template(
        stc_img=despiked,
        motion_mat_dir=mc.mat_dir,
        bold_to_anat=bbr.out_matrix_file,
        anat_to_template=anat_to_template,
        bold_ref=masking.skull_stripped_bold,
        template=MNI_TEMPLATES.brain_2mm,
        t1w_brain=t1w_brain,
    )

    # 10. Warp tissue masks to template space (same grid as resampled BOLD)
    tmpl_brain = _warp_mask_to_template(
        brain_mask, MNI_TEMPLATES.brain_2mm, anat_to_template
    )
    tmpl_csf = _warp_mask_to_template(
        csf_mask, MNI_TEMPLATES.brain_2mm, anat_to_template
    )
    tmpl_wm = _warp_mask_to_template(wm_mask, MNI_TEMPLATES.brain_2mm, anat_to_template)

    # 11. Nuisance regression in template space
    nuisance = nuisance_regression(
        bold_file=template_bold,
        brain_mask_file=tmpl_brain,
        csf_mask_file=tmpl_csf,
        wm_mask_file=tmpl_wm,
        motion_par_file=mc.par,
        regressor_set=regressor_set,
        bandpass=bandpass,
    )

    return FunctionalOutputs(
        reoriented_bold=reoriented.out_file,
        truncated_bold=truncated,
        stc_bold=st_corrected,
        despiked_bold=despiked,
        sbref=motion_ref,
        motion_corrected_bold=mc.bold,
        motion_params=mc.par,
        rms_rel=mc.rms_rel,
        rms_abs=mc.rms_abs,
        mat_dir=mc.mat_dir,
        bold_mask=masking.final_mask,
        skull_stripped_bold=masking.skull_stripped_bold,
        bold_to_anat_matrix=bbr.out_matrix_file,
        template_bold=template_bold,
        cleaned_bold=nuisance.cleaned_bold,
        regressor_file=nuisance.regressor_file,
        template_brain_mask=tmpl_brain,
    )


class FunctionalLongOutputs(NamedTuple):
    """Outputs from the functional preprocessing pipeline.

    Attributes:
        sbref:
        bold:
        bold_mask:
        forward_xfm:
    """

    forward_xfm: Path
    sbref: Path
    bold: Path
    bold_mask: Path | None = None


def longitudinal_process(
    template: Path,
    anat_to_template_xfm: Path,
    *,
    bold_to_anat_xfm: Path,
    sbref: Path,
    bold: Path,
    bold_mask: Path | None,
) -> FunctionalLongOutputs:
    """Transform preprocessed functional outputs to longitudinal template space.

    Assumes a template generated with transforms to anatomical, anatomical data already
    preprocessed to longitudinal.

    Args:
        template: Longitudinal template image.
        anat_to_template_xfm: Transform from subject to template space.
        bold_to_anat_xfm: Transformation from bold to preprocessed anatomical space.
        sbref: Motion reference (single-band reference) volume.
        bold: Preprocessed bold image.
        bold_mask: Bold brain mask, if available.

    Returns:
        :class:`FunctionalLongOutputs` with all non-null inputs transformed to template
            space.
    """
    bold_to_tpl_xfm = compose_transform(
        ref=template,
        bold_to_anat_xfm=bold_to_anat_xfm,
        anat_to_tpl_xfm=anat_to_template_xfm,
    )

    def _xfm(val: Path | None) -> Path | None:
        if val is None:
            return None
        return func_transform(in_file=val, template=template, xfm=bold_to_tpl_xfm)

    return FunctionalLongOutputs(
        sbref=func_transform(in_file=sbref, template=template, xfm=bold_to_tpl_xfm),
        bold=func_transform(in_file=bold, template=template, xfm=bold_to_tpl_xfm),
        bold_mask=_xfm(val=bold_mask),
        forward_xfm=bold_to_tpl_xfm,
    )
