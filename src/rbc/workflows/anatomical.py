"""Anatomical processing workflow.

Chains the full anatomical stream -- reorientation, brain extraction,
tissue segmentation, and template registration -- and returns all output
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
        anat_to_template_xfm: anat-to-template composite warp.
        template_to_anat_xfm: Template-to-anat composite warp.
    """

    brain: Path
    brain_mask: Path
    brain_tpl: Path
    csf_mask: Path
    gm_mask: Path
    wm_mask: Path
    wm_bbr_mask: Path
    anat_to_template_xfm: Path
    template_to_anat_xfm: Path


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
        anat_to_template_xfm=transforms.anat_to_template,
        template_to_anat_xfm=transforms.template_to_anat,
    )
