"""BIDS discovery, resolve, and export for the functional workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import polars as pl

from rbc.bids import Suffix, TemplateSpace, bids_safe_label, extract_entities
from rbc.bids.session import FUNC_GROUP_ENTITIES, SessionTables, iter_session_files

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rbc.bids import Bids, EntityKwargs
    from rbc.workflows.functional import FunctionalOutputs


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


class FunctionalRun(NamedTuple):
    """A single functional run discovered from a BIDS session.

    Attributes:
        path: Path to the BOLD NIfTI file.
        entities: BIDS entities for this run (task, run, acq, rec, dir, echo).
        anat_df: Matched anatomical DataFrame for this run.
    """

    path: Path
    entities: EntityKwargs
    anat_df: pl.DataFrame


def discover_functional(session: SessionTables) -> Iterator[FunctionalRun]:
    """Discover BOLD runs in a session, paired with matched anatomical data.

    Iterates via :func:`~rbc.bids.session.iter_session_files`, filters for
    raw (unprocessed) BOLD files, and extracts functional entities.

    Args:
        session: Session tables from :func:`~rbc.bids.session.load_session`.

    Yields:
        A :class:`FunctionalRun` for each BOLD group.
    """
    for func_df, anat_df in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
        func_df = func_df.filter(pl.col("desc").is_null())
        row = func_df.filter(suffix="bold").row(0, named=True)
        yield FunctionalRun(
            path=Path(row["root"]) / row["path"],
            entities=extract_entities(
                row, ["task", "run", "acq", "rec", "dir", "echo"]
            ),
            anat_df=anat_df,
        )


class FunctionalInputs(TypedDict):
    """Resolved anatomical inputs for the functional workflow."""

    t1w_brain: Path
    wm_bbr_mask: Path
    brain_mask: Path
    csf_mask: Path
    wm_mask: Path
    anat_to_template: Path


def resolve_functional(
    anat_q: Bids,
    anat_df: pl.DataFrame,
) -> FunctionalInputs:
    """Resolve anatomical prerequisites needed by functional preprocessing.

    Args:
        anat_q: Bids builder for anatomical datatype queries.
        anat_df: DataFrame of anatomical derivatives.

    Returns:
        Dict with keys matching ``single_session_preprocess`` parameters:
        ``t1w_brain``, ``wm_bbr_mask``, ``brain_mask``, ``csf_mask``,
        ``wm_mask``, ``anat_to_template``.
    """
    return {
        "t1w_brain": anat_q.expect(anat_df, suffix=Suffix.T1W, desc="brain"),
        "wm_bbr_mask": anat_q.expect(anat_df, suffix=Suffix.MASK, desc="wmBBR"),
        "brain_mask": anat_q.expect(anat_df, suffix=Suffix.MASK, desc="T1w"),
        "csf_mask": anat_q.expect(anat_df, suffix=Suffix.MASK, desc="csf"),
        "wm_mask": anat_q.expect(anat_df, suffix=Suffix.MASK, desc="wm"),
        "anat_to_template": anat_q.expect(
            anat_df,
            suffix="xfm",
            extra={
                "to": TemplateSpace.MNI152NLIN6ASYM,
                "from": "T1w",
                "mode": "image",
            },
        ),
    }


def export_functional(
    func: Bids,
    outputs: FunctionalOutputs,
    *,
    regressors: Sequence[str],
    smooth: float | None = None,
) -> Bids:
    """Export functional workflow outputs to BIDS-named derivatives.

    Args:
        func: Bids builder with ``datatype=FUNC`` and identity entities.
        outputs: Results from the functional preprocessing workflow.
        regressors: Regressor names (e.g. ``["36-parameter"]``).
        smooth: Smoothing kernel FWHM in mm, or ``None`` if smoothing
            was not requested.

    Returns:
        The MNI-space Bids builder, for use by downstream exports
        (metrics, QC).
    """
    func.save(outputs.sbref, suffix=Suffix.SBREF)
    func.save(outputs.preproc_bold, suffix=Suffix.BOLD, desc="preproc")
    func.save(
        outputs.motion_params,
        suffix=Suffix.MOTION,
        desc="motionParams",
        extension=".1D",
    )
    func.save(
        outputs.rms_rel,
        suffix=Suffix.MOTION,
        desc="relsDisplacement",
        extension=".rms",
    )
    func.save(
        outputs.rms_abs,
        suffix=Suffix.MOTION,
        desc="maxDisplacement",
        extension=".rms",
    )
    func.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
    func.save(
        outputs.bold_to_anat_matrix,
        suffix="xfm",
        desc="linear",
        extension=".txt",
        extra={"from": "bold", "to": "T1w", "mode": "image"},
    )
    func.save(
        outputs.bold_to_anat_itk,
        suffix="xfm",
        desc="linearITK",
        extension=".txt",
        extra={"from": "bold", "to": "T1w", "mode": "image"},
    )
    for reg in regressors:
        func.save(
            outputs.regressor_file[reg],
            suffix="regressors",
            desc=bids_safe_label(reg),
            extension=".1D",
        )
        func.save(
            outputs.bpf_regressor_file[reg],
            suffix="regressors",
            desc=f"{bids_safe_label(reg)}Filtered",
            extension=".1D",
        )

    mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)
    for reg in regressors:
        mni.save(
            outputs.regressed_bold[reg],
            suffix=Suffix.BOLD,
            desc="regressed",
            extra={"reg": bids_safe_label(reg)},
        )
        mni.save(
            outputs.cleaned_bold[reg],
            suffix=Suffix.BOLD,
            desc="preproc",
            extra={"reg": bids_safe_label(reg)},
        )
        if outputs.cleaned_bold_smooth is not None and smooth is not None:
            mni.save(
                outputs.cleaned_bold_smooth[reg],
                suffix=Suffix.BOLD,
                desc=f"{_smooth_label(smooth)}preproc",
                extra={"reg": bids_safe_label(reg)},
            )
    mni.save(outputs.template_bold, suffix=Suffix.BOLD, desc="preproc")
    mni.save(outputs.template_brain_mask, suffix=Suffix.MASK, desc="bold")

    return mni
