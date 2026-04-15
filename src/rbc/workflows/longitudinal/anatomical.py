"""Longitudinal anatomical processing workflow.

Transforms preprocessed anatomical outputs to a pre-computed longitudinal
template space and returns all output paths as an
:class:`AnatomicalLongOutputs` named tuple.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from rbc.core.anatomical import ants_registration
from rbc.core.longitudinal.transform import anat_transform
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger("rbc")


class AnatomicalLongOutputs(NamedTuple):
    """Outputs from the longitudinal anatomical preprocessing pipeline.

    Attributes:
        brain: Skull-stripped T1w brain warped to longitudinal template space.
        brain_mask: Binary brain mask warped to longitudinal template space,
            or *None* if not provided.
        long_to_template_xfm: Longitudinal template-to-MNI152 composite warp.
        template_to_long_xfm: MNI152-to-longitudinal template composite warp.
    """

    brain: Path
    brain_mask: Path | None
    long_to_template_xfm: Path
    template_to_long_xfm: Path


def longitudinal_process(
    template: Path,
    subj_to_template_xfm: Path,
    *,
    brain: Path,
    brain_mask: Path | None = None,
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
        registration_template: Brain template for ANTs registration.

    Returns:
        :class:`AnatomicalLongOutputs` with all non-null inputs transformed to template
            space.
    """
    _logger.info("Transforming anatomical outputs to longitudinal template space")
    _logger.info("Registration of longitudinal template to standard-space (ANTs)")
    transforms = ants_registration(
        in_file=template,
        registration_template=registration_template,
    )
    brain_mask_out = (
        anat_transform(in_file=brain_mask, template=template, xfm=subj_to_template_xfm)
        if brain_mask is not None
        else None
    )
    return AnatomicalLongOutputs(
        brain=anat_transform(
            in_file=brain, template=template, xfm=subj_to_template_xfm
        ),
        brain_mask=brain_mask_out,
        long_to_template_xfm=transforms.anat_to_template,
        template_to_long_xfm=transforms.template_to_anat,
    )
