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
            extra={"reg": regressor},
        ),
        "cleaned_bold": mni_q.expect(
            deriv_df,
            suffix=Suffix.BOLD,
            desc="preproc",
            extra={"reg": regressor},
        ),
    }


def export_metrics(
    mni: Bids,
    outputs: MetricsOutputs,
    *,
    regressor: str,
    atlases: Sequence[str],
) -> None:
    """Export metrics for a single regressor to BIDS-named derivatives.

    Args:
        mni: MNI-space Bids builder (typically from
            :func:`~rbc.bids.functional.export_functional`).
        outputs: Results from the metrics workflow.
        regressor: The regressor this run used.
        atlases: Atlas names used for timeseries extraction.
    """
    mex = mni.derive(extra={"reg": bids_safe_label(regressor)})
    mex.save(outputs.alff, suffix="alff")
    mex.save(outputs.falff, suffix="falff")
    mex.save(outputs.alff_smooth, suffix="alff", desc="smooth")
    mex.save(outputs.falff_smooth, suffix="falff", desc="smooth")
    mex.save(outputs.alff_zscored, suffix="alff", desc="smoothZstd")
    mex.save(outputs.falff_zscored, suffix="falff", desc="smoothZstd")
    mex.save(outputs.reho, suffix="reho")
    mex.save(outputs.reho_smooth, suffix="reho", desc="smooth")
    mex.save(outputs.reho_zscored, suffix="reho", desc="smoothZstd")
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
