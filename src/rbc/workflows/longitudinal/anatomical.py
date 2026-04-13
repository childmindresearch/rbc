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
        csf_mask: CSF tissue mask warped to longitudinal template space,
            or *None* if not provided.
        gm_mask: GM tissue mask warped to longitudinal template space,
            or *None* if not provided.
        wm_mask: WM tissue mask warped to longitudinal template space,
            or *None* if not provided.
        long_to_template_xfm: Longitudinal template-to-MNI152 composite warp.
        template_to_long_xfm: MNI152-to-longitudinal template composite warp.
    """

    brain: Path
    brain_mask: Path | None
    csf_mask: Path | None
    gm_mask: Path | None
    wm_mask: Path | None
    long_to_template_xfm: Path
    template_to_long_xfm: Path


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
        long_to_template_xfm=transforms.anat_to_template,
        template_to_long_xfm=transforms.template_to_anat,
    )
