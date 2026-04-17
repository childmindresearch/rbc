"""BIDS resolve and export for the longitudinal functional workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Suffix, bids_safe_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.longitudinal.functional import FunctionalLongOutputs

def _smooth_label(fwhm: float, precision: int | None = None) -> str:
    """Format FWHM as a BIDS-safe label (e.g. 6.0 -> 'sm6', 0.1 -> 'sm0p1').

    Trailing zeros are stripped and '.' is replaced with 'p' for BIDS compliance.

    Args:
        fwhm: Smoothing kernel FWHM in mm.
        precision: Optional number of decimal places to format to before stripping.

    Returns:
        BIDS-safe label string (e.g. 'sm6', 'sm0p1').
    """
    s = f"{fwhm:.{precision}f}" if precision is not None else str(fwhm)
    return "sm" + s.rstrip("0").rstrip(".").replace(".", "p")


def resolve_longitudinal_func(
    func_q: Bids,
    tpl_q: Bids,
    func_df: pl.DataFrame,
    tpl_df: pl.DataFrame,
    *,
    ses: str,
    regressors: Sequence[str] = ("36-parameter",),
) -> dict[str, Path | dict[str, Path]]:
    """Resolve inputs for longitudinal functional processing.

    Args:
        func_q: Bids builder for functional datatype queries.
        tpl_q: Bids builder for longitudinal template queries.
        func_df: DataFrame of functional derivatives.
        tpl_df: DataFrame of longitudinal template files.
        ses: Session label (used for template xfm lookup).
        regressors: Regressor strategy names to resolve raw regressor
            files for.

    Returns:
        Dict with keys matching ``longitudinal_process`` parameters,
        including ``regressor_files`` keyed by strategy name.
    """
    regressor_files: dict[str, Path] = {}
    for reg in regressors:
        regressor_files[reg] = func_q.expect(
            func_df,
            suffix="regressors",
            desc=bids_safe_label(reg),
            extension=".1D",
            without=["space"],
        )

    return {
        "template": tpl_q.expect(tpl_df, suffix="T1w"),
        "anat_to_template_xfm": tpl_q.expect(
            tpl_df,
            suffix="xfm",
            extension=".txt",
            extra={"from": bids_safe_label(ses), "to": "longitudinal"},
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
        "bold_mask": func_q.expect(
            func_df, suffix=Suffix.MASK, desc="brain", without=["space"]
        ),
        "regressor_files": regressor_files,
    }


def export_longitudinal_func(
    fex: Bids,
    outputs: FunctionalLongOutputs,
    *,
    regressors: Sequence[str],
    smooth: float | None = None,
) -> None:
    """Export longitudinal functional outputs.

    Args:
        fex: Bids builder with ``space="longitudinal"`` and identity entities.
        outputs: Results from the longitudinal functional workflow.
        regressors: Regressor strategy names that were applied.
        smooth: Smoothing kernel FWHM in mm, or ``None`` if not requested.
    """
    fex.save(outputs.sbref, suffix=Suffix.SBREF)
    fex.save(outputs.bold, suffix=Suffix.BOLD, desc="preproc")
    fex.save(
        outputs.bold_to_long_xfm,
        suffix="xfm",
        desc="composite",
        extra={"from": "bold", "to": "longitudinal", "mode": "image"},
    )
    fex.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")

    for reg in regressors:
        fex.save(
            outputs.regressed_bold[reg],
            suffix=Suffix.BOLD,
            desc="regressed",
            extra={"reg": bids_safe_label(reg)},
        )
        fex.save(
            outputs.cleaned_bold[reg],
            suffix=Suffix.BOLD,
            desc="preproc",
            extra={"reg": bids_safe_label(reg)},
        )
        if outputs.cleaned_bold_smooth is not None and smooth is not None:
            fex.save(
                outputs.cleaned_bold_smooth[reg],
                suffix=Suffix.BOLD,
                desc=f"{_smooth_label(smooth)}preproc",
                extra={"reg": bids_safe_label(reg)},
            )
