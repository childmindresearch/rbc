"""Orchestration for the longitudinal workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, load_table
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.orchestration.longitudinal.anatomical import process_anat
from rbc.orchestration.longitudinal.functional import process_func

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from rbc.bids.session import SessionTables
    from rbc.orchestration import Filters

__all__ = ["iter_sessions_with_template", "process_anat", "process_func"]

_logger = logging.getLogger(__name__)


def iter_sessions_with_template(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    verbose: bool = False,
) -> Iterator[tuple[RunContext, SessionTables, pl.DataFrame]]:
    """Yield ``(pipe_ctx, session, tpl_df)`` for each matching subject/session.

    Shared iteration primitive for the longitudinal anatomical and functional
    stages. Loads the BIDS table, filters out ``ses-longitudinal`` rows, then
    groups by subject/session. For each non-template session a longitudinal
    template must exist for that subject, otherwise ``ValueError`` is raised.

    Args:
        input_dirs: BIDS dataset directories.
        output_dir: Output directory for derivatives.
        filters: Participant/session filters applied before grouping.
        verbose: Whether to show a progress bar.

    Yields:
        Tuples of ``(pipe_ctx, session, tpl_df)`` for each subject/session.

    Raises:
        ValueError: If a matching session has no longitudinal template, or
            has no session label.
    """
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
        yield pipe_ctx, session, tpl_df
