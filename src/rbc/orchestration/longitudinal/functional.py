"""Per-run functional longitudinal processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Datatype, Suffix, extract_entities
from rbc.bids.longitudinal.functional import (
    export_longitudinal_func,
    resolve_longitudinal_func,
)
from rbc.workflows.longitudinal.functional import (
    longitudinal_process as functional_longitudinal,
)

if TYPE_CHECKING:
    import polars as pl

    from rbc.context import RunContext


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
