"""BIDS export and resolve for the QC workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from rbc.bids import Datatype, Suffix, bids_safe_label

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.context import RunContext
    from rbc.workflows.qc import QCOutputs


class QCInputs(TypedDict):
    """Resolved inputs for the QC workflow."""

    template_bold: Path
    cleaned_bold: Mapping[str, Path]
    motion_params: Path
    rms_rel: Path
    bold_mask: Path
    brain_mask: Path
    bold_to_anat_matrix: Path
    template_brain_mask: Path


def resolve_qc(
    func: Bids,
    func_mni: Bids,
    pipe_ctx: RunContext,
    deriv_df: pl.DataFrame,
    *,
    regressors: Sequence[str],
) -> QCInputs:
    """Resolve derivatives needed by the QC workflow.

    Args:
        func: Bids builder for native-space func queries.
        func_mni: Bids builder for MNI-space func queries.
        pipe_ctx: RunContext for anatomical queries.
        deriv_df: DataFrame of derivative outputs.
        regressors: Regressor names.

    Returns:
        Dict with keys matching ``single_session_qc`` parameters.
    """
    return {
        "template_bold": func_mni.expect(
            deriv_df, suffix=Suffix.BOLD, desc="preproc", extra={"reg": False}
        ),
        "cleaned_bold": {
            reg: func_mni.expect(
                deriv_df,
                suffix=Suffix.BOLD,
                desc="preproc",
                extra={"reg": bids_safe_label(reg)},
            )
            for reg in regressors
        },
        "motion_params": func.expect(
            deriv_df,
            suffix=Suffix.MOTION,
            desc="motionParams",
            extension=".1D",
        ),
        "rms_rel": func.expect(
            deriv_df,
            suffix=Suffix.MOTION,
            desc="relsDisplacement",
            extension=".rms",
        ),
        "bold_mask": func.expect(
            deriv_df,
            suffix=Suffix.MASK,
            desc="brain",
        ),
        "brain_mask": pipe_ctx.bids(datatype=Datatype.ANAT).expect(
            deriv_df, suffix=Suffix.MASK, desc="T1w"
        ),
        "bold_to_anat_matrix": func.expect(
            deriv_df,
            suffix="xfm",
            desc="linear",
            extension=".txt",
            extra={"from": "bold", "to": "T1w", "mode": "image"},
        ),
        "template_brain_mask": func_mni.expect(
            deriv_df, suffix=Suffix.MASK, desc="bold"
        ),
    }


def export_qc(
    mni: Bids,
    outputs: QCOutputs,
    *,
    regressors: Sequence[str],
) -> None:
    """Export QC outputs for all regressors.

    Args:
        mni: MNI-space Bids builder.
        outputs: Results from the QC workflow.
        regressors: Regressor names.
    """
    for reg in regressors:
        mni.save(
            outputs.qc_file[reg],
            suffix="quality",
            desc="xcp",
            extension=".tsv",
            extra={"reg": bids_safe_label(reg)},
        )
