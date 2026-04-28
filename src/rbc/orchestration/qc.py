"""Orchestration for the cross-sectional QC workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, Datatype, Suffix, TemplateSpace, load_table
from rbc.bids.qc import export_qc, resolve_qc
from rbc.bids.session import discover_derivative_runs
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.qc import single_session_qc
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_logger = logging.getLogger(__name__)


def run(
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    start_tr: int,
    mni_brain_mask_2mm: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the QC pipeline for all matching subjects/sessions.

    Args:
        output_dir: Directory containing functional derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        start_tr: Number of initial TRs discarded during preprocessing.
        mni_brain_mask_2mm: Brain mask for normalization QC (default: MNI152 2 mm).
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC QC workflow")
    full_df = load_table(
        dataset_dirs=output_dir, index_fpath=None, max_workers=0, verbose=verbose
    )

    df = filters.apply(
        full_df,
        pl.col("datatype") == Datatype.FUNC,
        pl.col("suffix") == Suffix.BOLD,
        pl.col("desc") == "preproc",
        pl.col("space") == TemplateSpace.MNI152NLIN6ASYM,
    )

    for _, group in tqdm(df.group_by(SUB_SES_QUERY), disable=not verbose):
        sub: str = group["sub"][0]
        ses: str | None = group["ses"][0] or None
        pipe_ctx = RunContext(sub=sub, ses=ses, output_dir=output_dir)

        for deriv_run in discover_derivative_runs(group):
            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=deriv_run.entities)
            func_mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)

            resolved = resolve_qc(
                func,
                func_mni,
                pipe_ctx,
                full_df,
                regressors=regressors,
            )

            qc_outputs = single_session_qc(
                **resolved,
                sub=sub,
                ses=ses or "",
                task=deriv_run.entities.get("task", ""),
                run=deriv_run.entities.get("run", 0),
                start_tr=start_tr,
                regressor_set=regressors,
                mni_brain_mask_2mm=mni_brain_mask_2mm,
            )

            export_qc(func_mni, qc_outputs, regressors=regressors)

            status = "PASSED" if qc_outputs.passed else "FAILED"
            _logger.info(
                "QC %s for sub-%s ses-%s task-%s run-%s",
                status,
                sub,
                ses,
                deriv_run.entities.get("task", ""),
                deriv_run.entities.get("run", 0),
            )

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC QC workflow complete")
