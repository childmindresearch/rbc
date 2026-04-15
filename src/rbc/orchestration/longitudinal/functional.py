"""Orchestration for the longitudinal functional workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rbc.bids import FUNC_GROUP_ENTITIES, Datatype, Suffix, extract_entities
from rbc.bids.longitudinal.functional import (
    export_longitudinal_func,
    resolve_longitudinal_func,
)
from rbc.bids.session import iter_session_files
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.longitudinal.functional import (
    longitudinal_process as functional_longitudinal,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from rbc.context import RunContext

__all__ = ["process_func", "run"]

_logger = logging.getLogger(__name__)


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
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run longitudinal functional processing for all matching subjects/sessions.

    Args:
        input_dirs: BIDS dataset directories (must include preprocessed
            cross-sectional functional derivatives, anatomical derivatives,
            and longitudinal templates).
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters applied before grouping.
        runner_config: Execution backend configuration.
    """
    # Local import to break the import cycle with the package ``__init__``.
    from rbc.orchestration.longitudinal import iter_sessions_with_template

    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.warning(
        "This workflow is experimental and may be sensitive to input file "
        "naming conventions."
    )
    _logger.info("Preparing to run RBC longitudinal functional workflow")

    for pipe_ctx, session, tpl_df in iter_sessions_with_template(
        input_dirs, output_dir, filters=filters, verbose=verbose
    ):
        for func_df, _ in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
            process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC longitudinal functional workflow complete")
