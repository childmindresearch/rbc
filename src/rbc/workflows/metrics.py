"""Derivative metrics workflow.

Orchestrates computation of ALFF, fALFF, ReHo, smoothing, z-scoring,
and atlas-based timeseries extraction from a cleaned BOLD timeseries.
Returns all output paths as a :class:`MetricsOutputs` named tuple.
No BIDS naming or file copying is performed here -- that responsibility
belongs to the CLI layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from rbc.core.metrics.alff import compute_alff
from rbc.core.metrics.reho import compute_reho
from rbc.core.metrics.smoothing import smooth
from rbc.core.metrics.standardization import compute_zscore
from rbc.core.metrics.timeseries import compute_timeseries
from rbc_resources import get_atlas

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc_resources import AtlasName

_logger = logging.getLogger("rbc")


class MetricsOutputs(NamedTuple):
    """Outputs from the derivative metrics workflow.

    Attributes:
        alff: Raw ALFF map.
        falff: Raw fALFF map.
        alff_smooth: Smoothed ALFF map.
        falff_smooth: Smoothed fALFF map.
        alff_zscored: Z-scored (smoothed) ALFF map.
        falff_zscored: Z-scored (smoothed) fALFF map.
        reho: Raw ReHo map.
        reho_smooth: Smoothed ReHo map.
        reho_zscored: Z-scored (smoothed) ReHo map.
        timeseries: Atlas-based mean timeseries TSV.
        correlation_matrix: Pairwise correlation matrix TSV.
    """

    alff: Path
    falff: Path
    alff_smooth: Path
    falff_smooth: Path
    alff_zscored: Path
    falff_zscored: Path
    reho: Path
    reho_smooth: Path
    reho_zscored: Path
    timeseries: dict[str, Path]
    correlation_matrix: dict[str, Path]


def single_session_metrics(
    regressed_bold: Path,
    cleaned_bold: Path,
    template_brain_mask: Path,
    tr: float | None = None,
    atlas: Sequence[AtlasName] = ("schaefer_200",),
    fwhm: float = 6.0,
) -> MetricsOutputs:
    """Compute all derivative metrics for a single functional run.

    Args:
        regressed_bold: Nuisance-regressed (non-bandpassed) BOLD in template space.
        cleaned_bold: Nuisance-regressed & bandpass-filtered BOLD in template space.
        template_brain_mask: Brain mask warped to template space.
        tr: Repetition time in seconds; if *None*, read from NIfTI header.
        atlas: Atlas short name for timeseries extraction.
        fwhm: Smoothing kernel FWHM in mm.

    Returns:
        All metric outputs bundled in a :class:`MetricsOutputs` tuple.
    """
    # 1. ALFF / fALFF on regressed BOLD (non-bandpassed)
    _logger.info("Computing ALFF/fALFF")
    alff_path, falff_path = compute_alff(
        regressed_bold, template_brain_mask, tr=tr, method="qm"
    )

    # 2. ReHo on bandpass-filtered cleaned BOLD
    _logger.info("Computing ReHo")
    reho_path = compute_reho(cleaned_bold, template_brain_mask)

    # 3. Smooth raw maps
    _logger.info("Smoothing maps (FWHM=%.1f mm)", fwhm)
    alff_smooth_path = smooth(alff_path, template_brain_mask, fwhm=fwhm)
    falff_smooth_path = smooth(falff_path, template_brain_mask, fwhm=fwhm)
    reho_smooth_path = smooth(reho_path, template_brain_mask, fwhm=fwhm)

    # 4. Z-score smoothed maps
    _logger.info("Z-scoring smoothed maps")
    alff_zscored_path = compute_zscore(alff_smooth_path, template_brain_mask)
    falff_zscored_path = compute_zscore(falff_smooth_path, template_brain_mask)
    reho_zscored_path = compute_zscore(reho_smooth_path, template_brain_mask)

    # 5. Atlas timeseries + correlation matrix from nuisance-regressed,
    # bandpass-filtered BOLD
    ts_outputs = {}
    for atl in atlas:
        _logger.info("Extracting atlas timeseries (%s)", atl)
        ts_outputs[atl] = compute_timeseries(cleaned_bold, get_atlas(atl))

    return MetricsOutputs(
        alff=alff_path,
        falff=falff_path,
        alff_smooth=alff_smooth_path,
        falff_smooth=falff_smooth_path,
        alff_zscored=alff_zscored_path,
        falff_zscored=falff_zscored_path,
        reho=reho_path,
        reho_smooth=reho_smooth_path,
        reho_zscored=reho_zscored_path,
        timeseries={atl: ts.timeseries for atl, ts in ts_outputs.items()},
        correlation_matrix={
            atl: ts.correlation_matrix for atl, ts in ts_outputs.items()
        },
    )
