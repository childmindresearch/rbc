"""Anatomical processing workflow.

Chains the full anatomical stream -- reorientation, brain extraction,
tissue segmentation, and template registration -- and performs longitudinal processing
-- transforming data to the longitudinal template space -- and returns all output
paths as an :class:`AnatomicalOutputs` named tuple.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from rbc.core.anatomical import (
    ants_brain_extraction,
    ants_registration,
    fsl_segmentation,
    fsl_tissue_masks,
    fsl_wm_bbr_mask,
)
from rbc.core.common import deoblique_and_reorient
from rbc.core.longitudinal.transform import anat_transform
from rbc_resources import (
    BRAIN_EXTRACTION_TEMPLATES,
    REGISTRATION_TEMPLATES,
    BrainExtractionTemplates,
)

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger("rbc")


class AnatomicalOutputs(NamedTuple):
    """Outputs from the anatomical preprocessing pipeline.

    Attributes:
        brain: Skull-stripped T1w brain.
        brain_mask: Binary brain mask.
        brain_tpl: Skull-stripped T1w brain in template space.
        csf_mask: CSF tissue mask.
        gm_mask: GM tissue mask.
        wm_mask: WM tissue mask.
        wm_bbr_mask: WM boundary mask for BBR coregistration.
        forward_xfm: T1w-to-template composite warp.
        inverse_xfm: Template-to-T1w composite warp.
    """

    brain: Path
    brain_mask: Path
    brain_tpl: Path
    csf_mask: Path
    gm_mask: Path
    wm_mask: Path
    wm_bbr_mask: Path
    forward_xfm: Path
    inverse_xfm: Path


def single_session_preprocess(
    in_t1w: Path,
    brain_extraction_templates: BrainExtractionTemplates = BRAIN_EXTRACTION_TEMPLATES,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
) -> AnatomicalOutputs:
    """Run the full anatomical preprocessing pipeline for one session.

    Pipeline steps:

    1. Deoblique and reorient T1w to RPI.
    2. ANTs brain extraction (via template; default is OASIS):
        a. N4 bias field correction
        b. Registration to template
        c. Warp brain probability mask to subject space
        d. Threshold mask to produce binary brain mask
    3. FSL FAST tissue segmentation on skull-stripped brain (CSF / GM / WM
       partial volume maps, thresholded at 0.95 for binary masks).
    4. WM boundary mask for BBR coregistration.
    5. ANTs registration to standard-space template (forward + inverse
       composite warps).

    Args:
        in_t1w: Raw T1w image to preprocess.
        brain_extraction_templates: Brain extraction template bundle.
        registration_template: Brain template for ANTs registration.

    Returns:
        All output paths bundled in an :class:`AnatomicalOutputs` tuple.
    """
    _logger.info("Deoblique and reorient T1w")
    reoriented_t1w = deoblique_and_reorient(in_file=in_t1w)
    _logger.info("Brain extraction (ANTs)")
    extracted_t1w = ants_brain_extraction(
        in_file=reoriented_t1w.out_file,
        brain_extraction_templates=brain_extraction_templates,
    )
    _logger.info("Tissue segmentation (FSL FAST)")
    segmentation = fsl_segmentation(in_file=extracted_t1w.brain)
    tissue_masks = fsl_tissue_masks(fast_result=segmentation)
    wm_bbr = fsl_wm_bbr_mask(fast_result=segmentation)
    _logger.info("Registration of T1w to template (ANTs)")
    transforms = ants_registration(
        in_file=extracted_t1w.brain,
        registration_template=registration_template,
    )

    return AnatomicalOutputs(
        brain=extracted_t1w.brain,
        brain_mask=extracted_t1w.brain_mask,
        brain_tpl=transforms.brain,
        csf_mask=tissue_masks.csf,
        gm_mask=tissue_masks.gm,
        wm_mask=tissue_masks.wm,
        wm_bbr_mask=wm_bbr,
        forward_xfm=transforms.forward,
        inverse_xfm=transforms.inverse,
    )


class AnatomicalLongOutputs(NamedTuple):
    """Outputs from the longitudinal anatomical preprocessing pipeline.

    Attributes:
        brain: Skull-stripped T1w brain warped to longitudinal template space.
        brain_mask: Binary brain mask warped to longitudinal template space,
            or *None* if not provided.
        csf_mask: CSF tissue mask warped to longitudinal template space,
            or *None* if not provided.
        gm_mask: GM tissue mask warped to longitudinal template space,
            or *None* if not provided.
        wm_mask: WM tissue mask warped to longitudinal template space,
            or *None* if not provided.
        forward_xfm: Longitudinal template-to-MNI152 composite warp.
        inverse_xfm: MNI152-to-longitudinal template composite warp.
    """

    brain: Path
    brain_mask: Path | None
    csf_mask: Path | None
    gm_mask: Path | None
    wm_mask: Path | None
    forward_xfm: Path
    inverse_xfm: Path


def longitudinal_process(
    template: Path,
    subj_to_template_xfm: Path,
    *,
    brain: Path,
    brain_mask: Path | None = None,
    csf_mask: Path | None = None,
    gm_mask: Path | None = None,
    wm_mask: Path | None = None,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
) -> AnatomicalLongOutputs:
    """Transform preprocessed anatomical outputs to longitudinal template space.

    Assumes a longitudinal template has been generated and a subject-to-template
    composite warp is available.

    Args:
        template: Longitudinal template image.
        subj_to_template_xfm: Subject-to-longitudinal-template composite warp.
        brain: Preprocessed brain image.
        brain_mask: Brain mask, if available.
        csf_mask: CSF partial volume mask, if available.
        gm_mask: Grey matter partial volume mask, if available.
        wm_mask: White matter partial volume mask, if available.
        registration_template: Brain template for ANTs registration.

    Returns:
        :class:`AnatomicalLongOutputs` with all non-null inputs transformed to template
            space.
    """

    def _xfm(val: Path | None) -> Path | None:
        if val is None:
            return None
        return anat_transform(in_file=val, template=template, xfm=subj_to_template_xfm)

    _logger.info("Transforming anatomical outputs to longitudinal template space")
    _logger.info("Registration of longitudinal template to standard-space (ANTs)")
    transforms = ants_registration(
        in_file=template,
        registration_template=registration_template,
    )
    return AnatomicalLongOutputs(
        brain=anat_transform(
            in_file=brain, template=template, xfm=subj_to_template_xfm
        ),
        brain_mask=_xfm(brain_mask),
        csf_mask=_xfm(csf_mask),
        gm_mask=_xfm(gm_mask),
        wm_mask=_xfm(wm_mask),
        forward_xfm=transforms.forward,
        inverse_xfm=transforms.inverse,
    )
