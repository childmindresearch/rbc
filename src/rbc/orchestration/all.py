"""Orchestration for the combined cross-sectional pipeline.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory between stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, load_table
from rbc.bids.metrics import export_metrics
from rbc.bids.qc import export_qc
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.anatomical import process_session as process_anat
from rbc.orchestration.functional import process_session as process_func
from rbc.workflows.metrics import single_session_metrics
from rbc.workflows.qc import single_session_qc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc_resources import AtlasName

_logger = logging.getLogger(__name__)


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    atlases: Sequence[AtlasName],
    fwhm: float,
    start_tr: int,
    tr: float | None = None,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the full pipeline (anat + func + metrics + QC) per session.

    Unlike running each workflow separately, this passes outputs in-memory
    between stages without disk round-trips.

    Args:
        input_dir: BIDS dataset directory.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        atlases: Atlas names for timeseries extraction.
        fwhm: Smoothing kernel FWHM in mm.
        start_tr: Number of initial TRs discarded during preprocessing.
        tr: TR override in seconds, or ``None`` to read from headers.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC full pipeline")
    df = load_table(
        dataset_dir=input_dir, index_fpath=None, max_workers=0, verbose=verbose
    )

    df = filters.apply(
        df,
        pl.col("ses") != "longitudinal",
        pl.col("space").is_null(),
        pl.col("desc").is_null(),
    )

    for _, sub_ses_group in tqdm(
        df.group_by(SUB_SES_QUERY, maintain_order=True), disable=not verbose
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        # --- Anatomical ---
        anat_outputs = process_anat(session, pipe_ctx)

        # --- Functional + Metrics + QC (per BOLD run) ---
        func_results = process_func(session, pipe_ctx, regressors=regressors, tr=tr)

        for func_outputs, mni, func_metadata in func_results:
            # Metrics (per regressor)
            for regressor in regressors:
                _logger.info(
                    "Metrics: sub-%s regressor-%s",
                    pipe_ctx.sub,
                    regressor,
                )
                metrics_outputs = single_session_metrics(
                    regressed_bold=func_outputs.regressed_bold[regressor],
                    cleaned_bold=func_outputs.cleaned_bold[regressor],
                    template_brain_mask=func_outputs.template_brain_mask,
                    tr=func_metadata.tr,
                    atlas=atlases,
                    fwhm=fwhm,
                )
                export_metrics(
                    mni,
                    metrics_outputs,
                    regressor=regressor,
                    atlases=atlases,
                )

            # QC
            _logger.info("QC: sub-%s", pipe_ctx.sub)
            qc_outputs = single_session_qc(
                template_bold=func_outputs.template_bold,
                cleaned_bold=func_outputs.cleaned_bold,
                motion_params=func_outputs.motion_params,
                rms_rel=func_outputs.rms_rel,
                bold_mask=func_outputs.bold_mask,
                brain_mask=anat_outputs.brain_mask,
                bold_to_anat_matrix=func_outputs.bold_to_anat_matrix,
                template_brain_mask=func_outputs.template_brain_mask,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses or "",
                task="",
                run=0,
                start_tr=start_tr,
                regressor_set=regressors,
            )
            export_qc(mni, qc_outputs, regressors=regressors)

            status = "PASSED" if qc_outputs.passed else "FAILED"
            _logger.info("QC %s for sub-%s", status, pipe_ctx.sub)

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC full pipeline complete")
