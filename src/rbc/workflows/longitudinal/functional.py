"""Longitudinal functional processing workflow.

Transforms preprocessed functional outputs to a pre-computed longitudinal
template space, then re-runs nuisance regression on the warped BOLD using
raw regressors from the cross-sectional run.  Returns all output paths as a
:class:`FunctionalLongOutputs` named tuple.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from rbc.core.functional import apply_regression, apply_regression_bandpass
from rbc.core.longitudinal.transform import (
    compose_transform,
    func_transform,
    mask_transform,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal

_logger = logging.getLogger("rbc")


class FunctionalLongOutputs(NamedTuple):
    """Outputs from the longitudinal functional preprocessing pipeline.

    Attributes:
        bold_to_long_xfm: BOLD-to-longitudinal-template composite warp.
        sbref: Motion reference volume warped to longitudinal template space.
        bold: Preprocessed BOLD warped to longitudinal template space.
        bold_mask: Brain mask warped to longitudinal template space.
        regressed_bold: Per-regressor nuisance-regressed BOLD (no bandpass)
            in longitudinal template space, keyed by strategy name.
        cleaned_bold: Per-regressor nuisance-regressed + bandpass-filtered
            BOLD in longitudinal template space, keyed by strategy name.
    """

    bold_to_long_xfm: Path
    sbref: Path
    bold: Path
    bold_mask: Path
    regressed_bold: dict[str, Path]
    cleaned_bold: dict[str, Path]


def longitudinal_process(
    template: Path,
    anat_to_template_xfm: Path,
    *,
    bold_to_anat_itk: Path,
    sbref: Path,
    bold: Path,
    bold_mask: Path,
    regressor_files: dict[str, Path],
    regressor_set: Sequence[Literal["36-parameter", "aCompCor"]] = ("36-parameter",),
) -> FunctionalLongOutputs:
    """Transform preprocessed functional outputs to longitudinal template space.

    After warping the BOLD timeseries, re-runs nuisance regression using the
    raw (unfiltered) regressors produced by the cross-sectional pipeline.
    No regressor recomputation is performed: the same regressor matrix is
    applied in the new target space.

    Args:
        template: Longitudinal template image.
        anat_to_template_xfm: T1w-to-longitudinal-template composite warp.
        bold_to_anat_itk: BOLD-to-T1w affine in ITK format.
        sbref: Motion reference (single-band reference) volume.
        bold: Preprocessed bold image.
        bold_mask: Bold brain mask.
        regressor_files: Raw (unfiltered) regressor ``.1D`` files from
            the cross-sectional run, keyed by strategy name.
        regressor_set: Which regressor strategies to apply.  Must be a
            subset of the keys in *regressor_files*.

    Returns:
        :class:`FunctionalLongOutputs` with all inputs transformed to
        longitudinal template space and per-regressor regression outputs.
    """
    bold_to_tpl_xfm = compose_transform(
        ref=template,
        bold_to_anat_itk=bold_to_anat_itk,
        anat_to_tpl_xfm=anat_to_template_xfm,
    )

    long_sbref = func_transform(
        in_file=sbref, template=template, xfm=bold_to_tpl_xfm, strategy="single"
    )
    long_bold = func_transform(
        in_file=bold, template=template, xfm=bold_to_tpl_xfm, strategy="chunked"
    )
    long_mask = mask_transform(mask=bold_mask, template=template, xfm=bold_to_tpl_xfm)

    regressed: dict[str, Path] = {}
    cleaned: dict[str, Path] = {}
    for reg in regressor_set:
        reg_file = regressor_files[reg]
        _logger.info("Longitudinal %s nuisance regression (no bandpass)", reg)
        regressed[reg] = apply_regression(
            bold_file=long_bold,
            brain_mask_file=long_mask,
            regressor_file=reg_file,
        ).regressed_bold

        _logger.info("Longitudinal %s nuisance regression + bandpass filtering", reg)
        cleaned[reg] = apply_regression_bandpass(
            bold_file=long_bold,
            brain_mask_file=long_mask,
            regressor_file=reg_file,
        ).regressed_bold

    return FunctionalLongOutputs(
        bold_to_long_xfm=bold_to_tpl_xfm,
        sbref=long_sbref,
        bold=long_bold,
        bold_mask=long_mask,
        regressed_bold=regressed,
        cleaned_bold=cleaned,
    )
