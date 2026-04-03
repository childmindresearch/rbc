"""BIDS resolve and export for the longitudinal workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Extension, Suffix

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.anatomical import AnatomicalLongOutputs
    from rbc.workflows.functional import FunctionalLongOutputs


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
            extra={"from": ses},
        ),
        "brain": anat_q.expect(anat_df, suffix=Suffix.T1W, desc="brain"),
        "brain_mask": anat_q.find(anat_df, suffix=Suffix.MASK, desc="T1w"),
        "csf_mask": anat_q.find(anat_df, suffix=Suffix.MASK, desc="csf"),
        "gm_mask": anat_q.find(anat_df, suffix=Suffix.MASK, desc="gm"),
        "wm_mask": anat_q.find(anat_df, suffix=Suffix.MASK, desc="wm"),
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
        _require_file(outputs.csf_mask, "csf_mask"),
        suffix=Suffix.MASK,
        desc="csf",
    )
    aex.save(
        _require_file(outputs.gm_mask, "gm_mask"),
        suffix=Suffix.MASK,
        desc="gm",
    )
    aex.save(
        _require_file(outputs.wm_mask, "wm_mask"),
        suffix=Suffix.MASK,
        desc="wm",
    )
    aex.save(
        outputs.forward_xfm,
        suffix="xfm",
        extra={"from": "T1w", "to": "longitudinal", "mode": "image"},
    )
    aex.save(
        outputs.inverse_xfm,
        suffix="xfm",
        extra={"from": "longitudinal", "to": "T1w", "mode": "image"},
    )


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
        outputs.forward_xfm,
        suffix="xfm",
        desc="composite",
        extension=Extension.NII_GZ,
        extra={"from": "bold", "to": "longitudinal", "mode": "image"},
    )
    if outputs.bold_mask:
        fex.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
