"""Anatomical processing workflow.

Chains the full anatomical stream -- reorientation, brain extraction,
tissue segmentation, and template registration -- and returns all output
paths as an :class:`AnatomicalOutputs` named tuple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from rbc.core.anatomical import (
    ants_brain_extraction,
    ants_registration,
    fsl_segmentation,
    fsl_tissue_masks,
    fsl_wm_bbr_mask,
)
from rbc.core.common import deoblique_and_reorient

if TYPE_CHECKING:
    from pathlib import Path


class AnatomicalOutputs(NamedTuple):
    """Outputs from the anatomical preprocessing pipeline.

    Attributes:
        brain: Skull-stripped T1w brain.
        brain_mask: Binary brain mask.
        csf_mask: CSF tissue mask.
        gm_mask: GM tissue mask.
        wm_mask: WM tissue mask.
        wm_bbr_mask: WM boundary mask for BBR coregistration.
        forward_xfm: T1w -> template composite warp.
        inverse_xfm: Template -> T1w composite warp.
    """

    brain: Path
    brain_mask: Path
    csf_mask: Path
    gm_mask: Path
    wm_mask: Path
    wm_bbr_mask: Path
    forward_xfm: Path
    inverse_xfm: Path


def single_session_preprocess(in_t1w: Path) -> AnatomicalOutputs:
    """Run the full anatomical preprocessing pipeline for one session.

    Pipeline steps:

    1. Deoblique and reorient T1w to RPI.
    2. ANTs brain extraction (N4 bias correction + skull-stripping).
    3. FSL FAST tissue segmentation (CSF / GM / WM masks).
    4. WM boundary mask for BBR coregistration.
    5. ANTs registration to MNI152 template (forward + inverse transforms).

    Args:
        in_t1w: Raw T1w image to preprocess.

    Returns:
        All output paths bundled in an :class:`AnatomicalOutputs` tuple.
    """
    reoriented_t1w = deoblique_and_reorient(in_file=in_t1w)
    extracted_t1w = ants_brain_extraction(in_file=reoriented_t1w.out_file)
    segmentation = fsl_segmentation(in_file=extracted_t1w.brain_extracted_image)
    tissue_masks = fsl_tissue_masks(fast_result=segmentation)
    wm_bbr = fsl_wm_bbr_mask(fast_result=segmentation)
    transforms = ants_registration(in_file=extracted_t1w.brain_extracted_image)

    return AnatomicalOutputs(
        brain=extracted_t1w.brain_extracted_image,
        brain_mask=extracted_t1w.brain_mask,
        csf_mask=tissue_masks.csf,
        gm_mask=tissue_masks.gm,
        wm_mask=tissue_masks.wm,
        wm_bbr_mask=wm_bbr,
        forward_xfm=transforms.forward,
        inverse_xfm=transforms.inverse,
    )
