"""Longitudinal functional processing workflow.

Transforms preprocessed functional outputs to a pre-computed longitudinal
template space and returns all output paths as a
:class:`FunctionalLongOutputs` named tuple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from rbc.core.longitudinal.transform import (
    compose_transform,
    func_transform,
    mask_transform,
)

if TYPE_CHECKING:
    from pathlib import Path


class FunctionalLongOutputs(NamedTuple):
    """Outputs from the longitudinal functional preprocessing pipeline.

    Attributes:
        bold_to_long_xfm: BOLD-to-longitudinal-template composite warp.
        sbref: Motion reference volume warped to longitudinal template space.
        bold: Preprocessed BOLD warped to longitudinal template space.
        bold_mask: Brain mask warped to longitudinal template space,
            or *None* if no mask was provided.
    """

    bold_to_long_xfm: Path
    sbref: Path
    bold: Path
    bold_mask: Path | None = None


def longitudinal_process(
    template: Path,
    anat_to_template_xfm: Path,
    *,
    bold_to_anat_itk: Path,
    sbref: Path,
    bold: Path,
    bold_mask: Path | None,
) -> FunctionalLongOutputs:
    """Transform preprocessed functional outputs to longitudinal template space.

    Assumes a longitudinal template has been generated, the subject-to-template
    composite warp is available, and anatomical data has already been processed
    to longitudinal template space.

    Args:
        template: Longitudinal template image.
        anat_to_template_xfm: T1w-to-longitudinal-template composite warp.
        bold_to_anat_itk: BOLD-to-T1w affine in ITK format.
        sbref: Motion reference (single-band reference) volume.
        bold: Preprocessed bold image.
        bold_mask: Bold brain mask, if available.

    Returns:
        :class:`FunctionalLongOutputs` with all non-null inputs transformed to template
            space.
    """
    bold_to_tpl_xfm = compose_transform(
        ref=template,
        bold_to_anat_itk=bold_to_anat_itk,
        anat_to_tpl_xfm=anat_to_template_xfm,
    )

    return FunctionalLongOutputs(
        sbref=func_transform(  # 3D volume
            in_file=sbref, template=template, xfm=bold_to_tpl_xfm, strategy="single"
        ),
        bold=func_transform(
            in_file=bold, template=template, xfm=bold_to_tpl_xfm, strategy="chunked"
        ),
        bold_mask=mask_transform(mask=bold_mask, template=template, xfm=bold_to_tpl_xfm)
        if bold_mask
        else None,
        bold_to_long_xfm=bold_to_tpl_xfm,
    )
