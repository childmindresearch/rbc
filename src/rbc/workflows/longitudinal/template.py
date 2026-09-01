"""Longitudinal template construction workflow.

Composes :func:`generate_robust_template` and :func:`fs_to_itk_xfm` from
``rbc.core.longitudinal.freesurfer`` to build a robust within-subject
template plus ANTs-compatible per-session transforms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from rbc.core.longitudinal.freesurfer import (
    fs_to_itk_xfm,
    generate_robust_template,
)
from rbc.core.longitudinal.resampling import resample_img_to_bold_res

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from rbc.bids.longitudinal.template import BoldKey

_logger = logging.getLogger("rbc")


class LongitudinalTemplateOutputs(NamedTuple):
    """Outputs from the longitudinal template workflow.

    Attributes:
        template: Robust within-subject template volume.
        bold_templates: Within-subject template volumes resampled to task-specific BOLD
            resolutions.
        sessions: Session labels in the same order as ``transforms``.
        transforms: Per-session ITK-format session-to-template transforms.
    """

    template: Path
    bold_templates: dict[BoldKey, Path]
    sessions: list[str]
    transforms: list[Path]


def generate_subject_template(
    sub: str,
    sessions: Sequence[str],
    in_files: Sequence[Path],
    bold_files: Mapping[BoldKey, Path],
) -> LongitudinalTemplateOutputs:
    """Build a robust template and ITK transforms for one subject.

    Args:
        sub: Subject label (without the ``sub-`` prefix).
        sessions: Session labels parallel to ``in_files``.
        in_files: Per-session preprocessed T1w volumes (e.g. brain-extracted).
        bold_files: Reference bold volumes to resample template for functional data.

    Returns:
        :class:`LongitudinalTemplateOutputs` ready for BIDS export.
    """
    _logger.info("Building robust template for sub-%s", sub)
    robust = generate_robust_template(
        sub=sub, sessions=list(sessions), in_files=list(in_files)
    )

    _logger.info("Converting FreeSurfer transforms to ITK format")
    itk_xfms = fs_to_itk_xfm(
        sub=sub,
        sessions=list(sessions),
        reference=robust.template,
        sources=list(in_files),
        in_xfms=robust.transforms,
    )

    _logger.info("Creating reference volumes for each functional task")
    bold_templates = {
        btask: resample_img_to_bold_res(bfile, robust.template)
        for btask, bfile in bold_files.items()
    }

    return LongitudinalTemplateOutputs(
        template=robust.template,
        bold_templates=bold_templates,
        sessions=list(sessions),
        transforms=itk_xfms,
    )
