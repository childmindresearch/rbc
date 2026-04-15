"""BIDS resolve and export for the longitudinal anatomical workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Suffix, TemplateSpace, bids_safe_label

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.longitudinal.anatomical import AnatomicalLongOutputs


def _require_file(path: Path | None, field: str) -> Path:
    """Raise if an expected output is None."""
    if path is None:
        raise ValueError(f"Expected output {field!r} is missing.")
    return path


def resolve_longitudinal_anat(
    anat_q: Bids,
    tpl_q: Bids,
    anat_df: pl.DataFrame,
    tpl_df: pl.DataFrame,
    *,
    ses: str,
) -> dict[str, Path | None]:
    """Resolve inputs for longitudinal anatomical processing.

    Args:
        anat_q: Bids builder for anatomical datatype queries.
        tpl_q: Bids builder for longitudinal template queries.
        anat_df: DataFrame of anatomical derivatives.
        tpl_df: DataFrame of longitudinal template files.
        ses: Session label (used for template xfm lookup).

    Returns:
        Dict with keys matching ``longitudinal_process`` parameters.
    """
    return {
        "template": tpl_q.expect(tpl_df, suffix=Suffix.T1W),
        "subj_to_template_xfm": tpl_q.expect(
            tpl_df,
            suffix="xfm",
            extension=".txt",
            extra={"from": bids_safe_label(ses), "to": "longitudinal"},
        ),
        "brain": anat_q.expect(anat_df, suffix=Suffix.T1W, desc="brain"),
        "brain_mask": anat_q.find(anat_df, suffix=Suffix.MASK, desc="T1w"),
    }


def export_longitudinal_anat(aex: Bids, outputs: AnatomicalLongOutputs) -> None:
    """Export longitudinal anatomical outputs.

    Args:
        aex: Bids builder with ``space="longitudinal"`` and identity entities.
        outputs: Results from the longitudinal anatomical workflow.
    """
    aex.save(outputs.brain, suffix=Suffix.T1W, desc="brain")
    aex.save(
        _require_file(outputs.brain_mask, "brain_mask"),
        suffix=Suffix.MASK,
        desc="T1w",
    )
    aex.save(
        outputs.long_to_template_xfm,
        suffix="xfm",
        extra={
            "from": "longitudinal",
            "to": TemplateSpace.MNI152NLIN6ASYM,
            "mode": "image",
        },
    )
    aex.save(
        outputs.template_to_long_xfm,
        suffix="xfm",
        extra={
            "from": TemplateSpace.MNI152NLIN6ASYM,
            "to": "longitudinal",
            "mode": "image",
        },
    )
