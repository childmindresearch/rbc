"""Orchestration for the longitudinal anatomical workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from rbc.bids import Datatype, extract_entities
from rbc.bids.longitudinal.anatomical import (
    export_longitudinal_anat,
    resolve_longitudinal_anat,
)
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.longitudinal._iter import iter_sessions_with_template
from rbc.workflows.longitudinal.anatomical import (
    longitudinal_process as anatomical_longitudinal,
)
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.context import RunContext

__all__ = ["process_anat", "run"]

_logger = logging.getLogger(__name__)


def process_anat(
    pipe_ctx: RunContext,
    anat_df: pl.DataFrame,
    tpl_df: pl.DataFrame,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
) -> None:
    """Handle anatomical longitudinal processing for one anat group.

    Args:
        pipe_ctx: RunContext bound to this subject/session.
        anat_df: Anatomical derivative DataFrame for this group.
        tpl_df: Longitudinal template DataFrame.
        registration_template: Brain template for ANTs registration.
    """
    anat_df = anat_df.filter(pl.col("space").is_null())
    ents = extract_entities(anat_df.row(0, named=True), ["run"])

    anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
    tpl_q = anat_q.derive(ses="longitudinal")

    resolved = resolve_longitudinal_anat(
        anat_q,
        tpl_q,
        anat_df,
        tpl_df,
        ses=pipe_ctx.ses,  # type: ignore[arg-type]
    )
    outputs = anatomical_longitudinal(
        **resolved,  # type: ignore[arg-type]
        registration_template=registration_template,
    )
    aex = anat_q.derive(entities=ents, space="longitudinal")
    export_longitudinal_anat(aex, outputs)


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run longitudinal anatomical processing for all matching subjects/sessions.

    Args:
        input_dirs: BIDS dataset directories (must include preprocessed
            cross-sectional anatomical derivatives and longitudinal templates).
        output_dir: Output directory for derivatives.
        filters: Participant/session filters applied before grouping.
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
    _logger.info("Preparing to run RBC longitudinal anatomical workflow")

    for pipe_ctx, session, tpl_df in iter_sessions_with_template(
        input_dirs, output_dir, filters=filters, verbose=verbose
    ):
        for _, anat_df in session.anat.filter(pl.col("suffix") == "T1w").group_by(
            ("run", "acq"), maintain_order=True
        ):
            process_anat(
                pipe_ctx=pipe_ctx,
                anat_df=anat_df,
                tpl_df=tpl_df,
                registration_template=registration_template,
            )
        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC longitudinal anatomical workflow complete")
