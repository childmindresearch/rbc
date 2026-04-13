"""Orchestration for the longitudinal workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import (
    FUNC_GROUP_ENTITIES,
    SUB_SES_QUERY,
    load_table,
)
from rbc.bids.session import iter_session_files, load_session
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.longitudinal.anatomical import process_anat
from rbc.orchestration.longitudinal.functional import process_func
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["process_anat", "process_func", "run"]

_logger = logging.getLogger(__name__)


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    anatomical: bool = True,
    functional: bool = True,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the longitudinal pipeline for all matching subjects/sessions.

    Args:
        input_dirs: BIDS dataset directories.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        anatomical: Run anatomical longitudinal processing.
        functional: Run functional longitudinal processing.
        registration_template: Brain template for ANTs registration.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.warning(
        "This workflow is experimental and may be sensitive to input file "
        "naming conventions."
    )
    _logger.info("Preparing to run RBC longitudinal workflow")
    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )

    group_df = filters.apply(df, pl.col("ses") != "longitudinal")

    for _, sub_ses_group in tqdm(
        group_df.group_by(SUB_SES_QUERY, maintain_order=True),
        disable=not verbose,
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0],
            output_dir=output_dir,
        )
        if pipe_ctx.ses is None:
            raise ValueError(
                "No session data, unable to perform longitudinal processing"
            )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)
        tpl_df = df.filter(
            pl.all_horizontal(
                pl.col("sub") == pipe_ctx.sub,
                pl.col("ses") == "longitudinal",
            )
        )
        if tpl_df.is_empty():
            raise ValueError("No longitudinal template found")

        if anatomical:
            for _, anat_df in session.anat.filter(pl.col("suffix") == "T1w").group_by(
                ("run", "acq"), maintain_order=True
            ):
                process_anat(
                    pipe_ctx=pipe_ctx,
                    anat_df=anat_df,
                    tpl_df=tpl_df,
                    registration_template=registration_template,
                )

        if functional:
            for func_df, _ in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
                process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC longitudinal workflow complete")
