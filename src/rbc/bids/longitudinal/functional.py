"""BIDS resolve and export for the longitudinal functional workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Suffix

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.longitudinal.functional import FunctionalLongOutputs


def resolve_longitudinal_func(
    func_q: Bids,
    tpl_q: Bids,
    func_df: pl.DataFrame,
    tpl_df: pl.DataFrame,
    *,
    ses: str,
) -> dict[str, Path | None]:
    """Resolve inputs for longitudinal functional processing.

    Args:
        func_q: Bids builder for functional datatype queries.
        tpl_q: Bids builder for longitudinal template queries.
        func_df: DataFrame of functional derivatives.
        tpl_df: DataFrame of longitudinal template files.
        ses: Session label (used for template xfm lookup).

    Returns:
        Dict with keys matching ``longitudinal_process`` parameters.
    """
    return {
        "template": tpl_q.expect(tpl_df, suffix="T1w"),
        "anat_to_template_xfm": tpl_q.expect(
            tpl_df,
            suffix="xfm",
            extension=".txt",
            extra={"from": ses},
        ),
        "bold_to_anat_itk": func_q.expect(
            func_df,
            suffix="xfm",
            desc="linearITK",
            extension=".txt",
            extra={"from": "bold", "to": "T1w", "mode": "image"},
        ),
        "sbref": func_q.expect(func_df, suffix=Suffix.SBREF, without=["space"]),
        "bold": func_q.expect(
            func_df, suffix=Suffix.BOLD, desc="preproc", without=["space"]
        ),
        "bold_mask": func_q.find(
            func_df, suffix=Suffix.MASK, desc="brain", without=["space"]
        ),
    }


def export_longitudinal_func(fex: Bids, outputs: FunctionalLongOutputs) -> None:
    """Export longitudinal functional outputs.

    Args:
        fex: Bids builder with ``space="longitudinal"`` and identity entities.
        outputs: Results from the longitudinal functional workflow.
    """
    fex.save(outputs.sbref, suffix=Suffix.SBREF)
    fex.save(outputs.bold, suffix=Suffix.BOLD, desc="preproc")
    fex.save(
        outputs.bold_to_long_xfm,
        suffix="xfm",
        desc="composite",
        extra={"from": "bold", "to": "longitudinal", "mode": "image"},
    )
    if outputs.bold_mask:
        fex.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
