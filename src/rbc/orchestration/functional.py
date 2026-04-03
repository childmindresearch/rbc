"""Orchestration for the cross-sectional functional workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, Datatype, load_table
from rbc.bids.functional import (
    discover_functional,
    export_functional,
    resolve_functional,
)
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.metadata import FunctionalMetadata
from rbc.workflows.functional import single_session_preprocess

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids import Bids
    from rbc.bids.session import SessionTables
    from rbc.orchestration import Filters
    from rbc.workflows.functional import FunctionalOutputs

_logger = logging.getLogger(__name__)


def process_session(
    session: SessionTables,
    pipe_ctx: RunContext,
    *,
    regressors: Sequence[str],
    tr: float | None = None,
) -> list[tuple[FunctionalOutputs, Bids, FunctionalMetadata]]:
    """Run functional preprocessing for one session.

    Args:
        session: Session tables for a single subject/session.
        pipe_ctx: RunContext bound to this subject/session.
        regressors: Regressor names.
        tr: TR override in seconds, or ``None`` to read from headers.

    Returns:
        List of (outputs, mni_builder, metadata) per BOLD run,
        for use by downstream workflows (metrics, QC).
    """
    results = []
    for func_run in discover_functional(session):
        _logger.info("Functional: %s", func_run.path)

        anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
        resolved = resolve_functional(anat_q, func_run.anat_df)

        func_metadata = FunctionalMetadata.load(func_run.path, tr_override=tr)

        outputs = single_session_preprocess(
            in_bold=func_run.path,
            t1w_brain=resolved["t1w_brain"],
            wm_bbr_mask=resolved["wm_bbr_mask"],
            brain_mask=resolved["brain_mask"],
            csf_mask=resolved["csf_mask"],
            wm_mask=resolved["wm_mask"],
            anat_to_template=resolved["anat_to_template"],
            metadata=func_metadata,
            regressor_set=regressors,  # type: ignore[arg-type]
        )

        func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=func_run.entities)
        mni = export_functional(func, outputs, regressors=regressors)
        results.append((outputs, mni, func_metadata))

    return results


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    tr: float | None = None,
    verbose: bool = False,
) -> None:
    """Run the functional pipeline for all matching subjects/sessions.

    Args:
        input_dir: BIDS dataset directory.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        tr: TR override in seconds.
        verbose: Show progress bar.
    """
    df = load_table(
        dataset_dir=input_dir, index_fpath=None, max_workers=0, verbose=verbose
    )

    filter_exprs = [
        pl.col("ses") != "longitudinal",
        pl.col("space").is_null(),
    ]
    if len(filters.participant_label) > 0:
        filter_exprs.append(pl.col("sub").is_in(filters.participant_label))
    if len(filters.session_label) > 0:
        filter_exprs.append(pl.col("ses").is_in(filters.session_label))
    if filters.task is not None:
        filter_exprs.append(pl.col("task") == filters.task)
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
        process_session(session, pipe_ctx, regressors=regressors, tr=tr)
        pipe_ctx.ensure_dataset_description()
