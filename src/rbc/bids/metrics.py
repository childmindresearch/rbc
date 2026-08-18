"""BIDS export and resolve for the metrics workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from rbc.bids import Suffix, bids_safe_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.metrics import MetricsOutputs


def _smooth_label(fwhm: float, precision: int | None = None) -> str:
    """Format FWHM as a BIDS-safe label (e.g. 6.0 -> 'sm6', 0.1 -> 'sm0p1')."""
    s = f"{fwhm:.{precision}f}" if precision is not None else str(fwhm)
    return "sm" + s.rstrip("0").rstrip(".").replace(".", "p")


class MetricsInputs(TypedDict):
    """Resolved functional inputs for the metrics workflow."""

    template_brain_mask: Path
    regressed_bold: Path
    cleaned_bold: Path


def resolve_metrics(
    mni_q: Bids,
    deriv_df: pl.DataFrame,
    *,
    regressor: str,
) -> MetricsInputs:
    """Resolve functional derivatives needed by the metrics workflow.

    Args:
        mni_q: Bids builder configured for MNI-space func queries.
        deriv_df: DataFrame of derivative outputs.
        regressor: Single regressor name (e.g. ``"36-parameter"``).

    Returns:
        Dict with keys: ``template_brain_mask``, ``regressed_bold``,
        ``cleaned_bold``.
    """
    return {
        "template_brain_mask": mni_q.expect(deriv_df, suffix=Suffix.MASK, desc="bold"),
        "regressed_bold": mni_q.expect(
            deriv_df,
            suffix=Suffix.BOLD,
            desc="regressed",
            extra={"reg": bids_safe_label(regressor)},
        ),
        "cleaned_bold": mni_q.expect(
            deriv_df,
            suffix=Suffix.BOLD,
            desc="preproc",
            extra={"reg": bids_safe_label(regressor)},
        ),
    }


def export_metrics(
    mni: Bids,
    outputs: MetricsOutputs,
    *,
    regressor: str,
    atlases: Sequence[str],
    smooth: float | None,
) -> None:
    """Export metrics for a single regressor to BIDS-named derivatives.

    Raw and z-scored raw maps are always exported. Smoothed and
    z-scored smoothed variants are exported only when the corresponding
    fields are not None (i.e. when ``smooth`` is not ``None`` in
    ``single_session_metrics``).

    Args:
        mni: MNI-space Bids builder (typically from
            :func:`~rbc.bids.functional.export_functional`).
        outputs: Results from the metrics workflow.
        regressor: The regressor this run used.
        atlases: Atlas names used for timeseries extraction.
        smooth: Smoothing kernel FWHM in mm, or ``None`` if smoothing
            was not requested.
    """
    mex = mni.derive(extra={"reg": bids_safe_label(regressor)})

    # Raw maps
    mex.save(outputs.alff, suffix="alff")
    mex.save(outputs.falff, suffix="falff")
    mex.save(outputs.reho, suffix="reho")

    # Z-scored raw maps
    mex.save(outputs.alff_zscored, suffix="alff", desc="zstd")
    mex.save(outputs.falff_zscored, suffix="falff", desc="zstd")
    mex.save(outputs.reho_zscored, suffix="reho", desc="zstd")

    # Smoothed + z-scored smoothed
    if smooth is not None:
        sm_desc = _smooth_label(smooth)
        if outputs.alff_smooth is not None:
            mex.save(outputs.alff_smooth, suffix="alff", desc=sm_desc)
            assert outputs.alff_smooth_zscored is not None  # noqa: S101
            mex.save(outputs.alff_smooth_zscored, suffix="alff", desc=f"{sm_desc}Zstd")
        if outputs.falff_smooth is not None:
            mex.save(outputs.falff_smooth, suffix="falff", desc=sm_desc)
            assert outputs.falff_smooth_zscored is not None  # noqa: S101
            mex.save(
                outputs.falff_smooth_zscored, suffix="falff", desc=f"{sm_desc}Zstd"
            )
        if outputs.reho_smooth is not None:
            mex.save(outputs.reho_smooth, suffix="reho", desc=sm_desc)
            assert outputs.reho_smooth_zscored is not None  # noqa: S101
            mex.save(outputs.reho_smooth_zscored, suffix="reho", desc=f"{sm_desc}Zstd")

    for atl in atlases:
        mex.save(
            outputs.timeseries[atl],
            suffix="timeseries",
            desc="mean",
            extension=".tsv",
            atlas=bids_safe_label(atl),
        )
        mex.save(
            outputs.correlation_matrix[atl],
            suffix="correlations",
            desc="pearson",
            extension=".tsv",
            atlas=bids_safe_label(atl),
        )
