"""Orchestration for the cross-sectional anatomical workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, Datatype, load_table
from rbc.bids.anatomical import discover_anatomical, export_anatomical
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.anatomical import AnatomicalOutputs, single_session_preprocess

if TYPE_CHECKING:
    from pathlib import Path

    from rbc.bids.session import SessionTables

_logger = logging.getLogger(__name__)


def process_session(
    session: SessionTables,
    pipe_ctx: RunContext,
) -> AnatomicalOutputs:
    """Run anatomical preprocessing for one session.

    Args:
        session: Session tables for a single subject/session.
        pipe_ctx: RunContext bound to this subject/session.

    Returns:
        The last :class:`AnatomicalOutputs` (for use by downstream workflows).
    """
    outputs: AnatomicalOutputs | None = None
    for anat_run in discover_anatomical(session):
        _logger.info("Anatomical: %s", anat_run.path)
        outputs = single_session_preprocess(in_t1w=anat_run.path)
        anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=anat_run.entities)
        export_anatomical(anat, outputs)

    if outputs is None:
        msg = f"No T1w files found for sub-{pipe_ctx.sub} ses-{pipe_ctx.ses}"
        raise FileNotFoundError(msg)
    return outputs


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    filters: Filters,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the anatomical pipeline for all matching subjects/sessions.

    Args:
        input_dir: BIDS dataset directory.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC anatomical workflow")
    df = load_table(
        dataset_dir=input_dir, index_fpath=None, max_workers=0, verbose=verbose
    )

    filter_exprs = [
        pl.col("ses") != "longitudinal",
        pl.col("space").is_null(),
        pl.col("desc").is_null(),
    ]
    if len(filters.participant_label) > 0:
        filter_exprs.append(pl.col("sub").is_in(filters.participant_label))
    if len(filters.session_label) > 0:
        filter_exprs.append(pl.col("ses").is_in(filters.session_label))
    df = df.filter(pl.all_horizontal(filter_exprs))

    for _, sub_ses_group in tqdm(
        df.group_by(SUB_SES_QUERY, maintain_order=True), disable=not verbose
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)
        process_session(session, pipe_ctx)
        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC anatomical workflow complete")
