"""Per-group anatomical longitudinal processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from rbc.bids import Datatype, extract_entities
from rbc.bids.longitudinal.anatomical import (
    export_longitudinal_anat,
    resolve_longitudinal_anat,
)
from rbc.workflows.longitudinal.anatomical import (
    longitudinal_process as anatomical_longitudinal,
)
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from pathlib import Path

    from rbc.context import RunContext


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
