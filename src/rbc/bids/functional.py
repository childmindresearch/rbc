"""BIDS export and resolve for the functional workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Suffix, TemplateSpace, bids_safe_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids
    from rbc.workflows.functional import FunctionalOutputs


def resolve_functional(
    anat_q: Bids,
    anat_df: pl.DataFrame,
) -> dict[str, Path]:
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
                "from": TemplateSpace.MNI152NLIN6ASYM,
                "to": "T1w",
                "mode": "image",
            },
        ),
    }


def export_functional(
    func: Bids,
    outputs: FunctionalOutputs,
    *,
    regressors: Sequence[str],
) -> Bids:
    """Export functional workflow outputs to BIDS-named derivatives.

    Args:
        func: Bids builder with ``datatype=FUNC`` and identity entities.
        outputs: Results from the functional preprocessing workflow.
        regressors: Regressor names (e.g. ``["36-parameter"]``).

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
    mni.save(outputs.template_bold, suffix=Suffix.BOLD, desc="preproc")
    mni.save(outputs.template_brain_mask, suffix=Suffix.MASK, desc="bold")

    return mni
