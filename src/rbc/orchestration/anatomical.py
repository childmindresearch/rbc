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
from rbc_resources import (
    BRAIN_EXTRACTION_TEMPLATES,
    REGISTRATION_TEMPLATES,
    BrainExtractionTemplates,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids.session import SessionTables

_logger = logging.getLogger(__name__)


def process_session(
    session: SessionTables,
    pipe_ctx: RunContext,
    brain_extraction_templates: BrainExtractionTemplates = BRAIN_EXTRACTION_TEMPLATES,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
) -> AnatomicalOutputs:
    """Run anatomical preprocessing for one session.

    Args:
        session: Session tables for a single subject/session.
        pipe_ctx: RunContext bound to this subject/session.
        brain_extraction_templates: Brain extraction template bundle.
        registration_template: Brain template for ANTs registration.

    Returns:
        The last :class:`AnatomicalOutputs` (for use by downstream workflows).
    """
    outputs: AnatomicalOutputs | None = None
    for anat_run in discover_anatomical(session):
        _logger.info("Anatomical: %s", anat_run.path)
        outputs = single_session_preprocess(
            in_t1w=anat_run.path,
            brain_extraction_templates=brain_extraction_templates,
            registration_template=registration_template,
        )
        anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=anat_run.entities)
        export_anatomical(anat, outputs)

    if outputs is None:
        msg = f"No T1w files found for sub-{pipe_ctx.sub} ses-{pipe_ctx.ses}"
        raise FileNotFoundError(msg)
    return outputs


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    brain_extraction_templates: BrainExtractionTemplates = BRAIN_EXTRACTION_TEMPLATES,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the anatomical pipeline for all matching subjects/sessions.

    Args:
        input_dirs: BIDS dataset directories.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        brain_extraction_templates: Brain extraction template bundle.
        registration_template: Brain template for ANTs registration.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC anatomical workflow")
    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
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
        process_session(
            session,
            pipe_ctx,
            brain_extraction_templates=brain_extraction_templates,
            registration_template=registration_template,
        )
        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC anatomical workflow complete")
