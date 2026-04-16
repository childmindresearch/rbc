"""BIDS resolve and export for the longitudinal metrics workflow.

Reuses the cross-sectional :func:`~rbc.bids.metrics.export_metrics` for
export since the output structure (ALFF, fALFF, ReHo, timeseries,
correlations) is identical -- only the space changes to ``longitudinal``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from rbc.bids import Suffix, bids_safe_label

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids


class MetricsLongInputs(TypedDict):
    """Resolved functional inputs for the longitudinal metrics workflow."""

    template_brain_mask: Path
    regressed_bold: Path
    cleaned_bold: Path


def resolve_longitudinal_metrics(
    func_long_q: Bids,
    func_long_df: pl.DataFrame,
    *,
    regressor: str,
) -> MetricsLongInputs:
    """Resolve longitudinal-space functional derivatives for metrics.

    Args:
        func_long_q: Bids builder configured for ``space=longitudinal``
            func queries.
        func_long_df: DataFrame of longitudinal-space derivative outputs.
        regressor: Single regressor name (e.g. ``"36-parameter"``).

    Returns:
        Dict with keys: ``template_brain_mask``, ``regressed_bold``,
        ``cleaned_bold``.
    """
    return {
        "template_brain_mask": func_long_q.expect(
            func_long_df, suffix=Suffix.MASK, desc="brain"
        ),
        "regressed_bold": func_long_q.expect(
            func_long_df,
            suffix=Suffix.BOLD,
            desc="regressed",
            extra={"reg": bids_safe_label(regressor)},
        ),
        "cleaned_bold": func_long_q.expect(
            func_long_df,
            suffix=Suffix.BOLD,
            desc="preproc",
            extra={"reg": bids_safe_label(regressor)},
        ),
    }
