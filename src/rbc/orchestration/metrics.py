"""Orchestration for the cross-sectional metrics workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, Datatype, TemplateSpace, load_table
from rbc.bids.metrics import export_metrics, resolve_metrics
from rbc.bids.session import discover_derivative_runs
from rbc.context import RunContext
from rbc.workflows.metrics import single_session_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids import Bids
    from rbc.orchestration import Filters
    from rbc.workflows.functional import FunctionalOutputs
    from rbc.workflows.metrics import MetricsOutputs
    from rbc_resources import AtlasName

_logger = logging.getLogger(__name__)


def _read_header_tr(nifti_path: Path) -> float:
    """Read TR from a NIfTI header, raising on missing/zero values."""
    hdr = nib.nifti1.load(nifti_path).header
    tr = float(hdr["pixdim"][4])  # type: ignore[index]
    if tr <= 0:
        msg = (
            f"NIfTI header TR is {tr} for {nifti_path}. Pass --tr to specify manually."
        )
        raise ValueError(msg)
    _logger.info("TR: %.4f s (from NIfTI header)", tr)
    return tr


def process_run(
    mni: Bids,
    *,
    func_outputs: FunctionalOutputs,
    tr: float,
    regressor: str,
    atlases: Sequence[AtlasName],
    fwhm: float,
) -> MetricsOutputs:
    """Run metrics for a single regressor on a single BOLD run.

    Used by ``orchestration.all`` to process in-memory functional outputs.

    Args:
        mni: MNI-space Bids builder for this run.
        func_outputs: Functional outputs (in-memory from ``all`` pipeline).
        tr: Repetition time in seconds.
        regressor: Regressor name.
        atlases: Atlas names for timeseries extraction.
        fwhm: Smoothing kernel FWHM in mm.

    Returns:
        Metrics outputs for this run/regressor.
    """
    outputs = single_session_metrics(
        regressed_bold=func_outputs.regressed_bold[regressor],
        cleaned_bold=func_outputs.cleaned_bold[regressor],
        template_brain_mask=func_outputs.template_brain_mask,
        tr=tr,
        atlas=atlases,
        fwhm=fwhm,
    )
    export_metrics(mni, outputs, regressor=regressor, atlases=atlases)
    return outputs


def run(
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    atlases: Sequence[AtlasName],
    fwhm: float,
    tr: float | None = None,
    verbose: bool = False,
) -> None:
    """Run the metrics pipeline for all matching subjects/sessions.

    Args:
        output_dir: Directory containing functional derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        atlases: Atlas names for timeseries extraction.
        fwhm: Smoothing kernel FWHM in mm.
        tr: TR override in seconds, or ``None`` to read from headers.
        verbose: Show progress bar.
    """
    df = load_table(
        dataset_dir=output_dir, index_fpath=None, max_workers=0, verbose=verbose
    )

    filter_exprs = [
        pl.col("datatype") == "func",
        pl.col("suffix") == "bold",
        pl.col("desc") == "preproc",
        pl.col("space") == TemplateSpace.MNI152NLIN6ASYM,
    ]
    if len(filters.participant_label) > 0:
        filter_exprs.append(pl.col("sub").is_in(filters.participant_label))
    if len(filters.session_label) > 0:
        filter_exprs.append(pl.col("ses").is_in(filters.session_label))
    if filters.task is not None:
        filter_exprs.append(pl.col("task") == filters.task)
    df = df.filter(pl.all_horizontal(filter_exprs))

    for _, group in tqdm(df.group_by(SUB_SES_QUERY), disable=not verbose):
        sub: str = group["sub"][0]
        ses: str | None = group["ses"][0] or None
        pipe_ctx = RunContext(sub=sub, ses=ses, output_dir=output_dir)

        deriv_df = load_table(
            dataset_dir=output_dir,
            index_fpath=None,
            max_workers=0,
            verbose=False,
        )

        for deriv_run in discover_derivative_runs(group):
            mni_q = pipe_ctx.bids(
                datatype=Datatype.FUNC,
                entities=deriv_run.entities,
                space=TemplateSpace.MNI152NLIN6ASYM,
            )

            for regressor in regressors:
                resolved = resolve_metrics(mni_q, deriv_df, regressor=regressor)
                run_tr = (
                    tr
                    if tr is not None
                    else _read_header_tr(resolved["regressed_bold"])
                )
                outputs = single_session_metrics(
                    regressed_bold=resolved["regressed_bold"],
                    cleaned_bold=resolved["cleaned_bold"],
                    template_brain_mask=resolved["template_brain_mask"],
                    tr=run_tr,
                    atlas=atlases,
                    fwhm=fwhm,
                )
                export_metrics(
                    mni_q,
                    outputs,
                    regressor=regressor,
                    atlases=atlases,
                )

        pipe_ctx.ensure_dataset_description()
