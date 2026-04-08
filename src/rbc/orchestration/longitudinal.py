"""Orchestration for the longitudinal workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import (
    FUNC_GROUP_ENTITIES,
    SUB_SES_QUERY,
    Datatype,
    Suffix,
    extract_entities,
    load_table,
)
from rbc.bids.longitudinal import (
    export_longitudinal_anat,
    export_longitudinal_func,
    resolve_longitudinal_anat,
    resolve_longitudinal_func,
)
from rbc.bids.session import iter_session_files, load_session
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.anatomical import longitudinal_process as anatomical_longitudinal
from rbc.workflows.functional import longitudinal_process as functional_longitudinal
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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


def process_func(
    pipe_ctx: RunContext,
    func_df: pl.DataFrame,
    tpl_df: pl.DataFrame,
) -> None:
    """Handle functional longitudinal processing for one BOLD run.

    Args:
        pipe_ctx: RunContext bound to this subject/session.
        func_df: Functional derivative DataFrame for this run.
        tpl_df: Longitudinal template DataFrame.
    """
    row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
    ents = extract_entities(row, ["task", "run"])

    func_q = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
    tpl_q = pipe_ctx.bids(datatype=Datatype.ANAT).derive(ses="longitudinal")

    resolved = resolve_longitudinal_func(
        func_q,
        tpl_q,
        func_df,
        tpl_df,
        ses=pipe_ctx.ses,  # type: ignore[arg-type]
    )
    func_outputs = functional_longitudinal(**resolved)  # type: ignore[arg-type]
    fex = func_q.derive(space="longitudinal")
    export_longitudinal_func(fex, func_outputs)


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
