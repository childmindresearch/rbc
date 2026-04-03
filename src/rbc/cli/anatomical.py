"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.bids.session import load_session

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

from pathlib import Path

import polars as pl
from tqdm import tqdm

from rbc.bids import (
    ANAT_GROUP_ENTITIES,
    SUB_SES_QUERY,
    Datatype,
    extract_entities,
    load_table,
)
from rbc.bids.anatomical import export_anatomical
from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.base import BaseArgs
from rbc.context import RunContext
from rbc.core.niwrap import setup_runner
from rbc.workflows.anatomical import single_session_preprocess


@dataclass(frozen=True)
class AnatomicalArgs(BaseArgs):
    """Arguments for single-session anatomical CLI."""

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AnatomicalArgs:
        """Validation of anatomical workflow specific arguments to NamedTuple."""
        return cls(**BaseArgs.validate_namespace(ns).__dict__)


def main(args: AnatomicalArgs) -> int:
    """Main entrypoint of anatomical workflow."""
    # Setup
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC anatomical workflow")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = [
        pl.col("ses") != "longitudinal",
        pl.col("space").is_null(),
        pl.col("desc").is_null(),
    ]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    df = df.filter(pl.all_horizontal(filters))

    for _, sub_ses_group in tqdm(
        df.group_by(SUB_SES_QUERY, maintain_order=True), disable=not ctx.verbose
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        for _, anat_df in session.anat.filter(pl.col("suffix") == "T1w").group_by(
            ANAT_GROUP_ENTITIES, maintain_order=True
        ):
            row = anat_df.row(0, named=True)
            t1w_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["run", "acq", "rec", "echo"])
            ctx.logger.info(f"Processing {t1w_fpath}")

            outputs = single_session_preprocess(in_t1w=t1w_fpath)

            anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=ents)
            export_anatomical(anat, outputs)
        pipe_ctx.ensure_dataset_description()

    ctx.logger.info("RBC anatomical workflow complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register anatomical workflow to parser."""
    parser = subparsers.add_parser(
        "anatomical",
        parents=parents,
        description="RBC anatomical workflow",
        help="Anatomical workflow",
        usage="rbc input_dir output_dir anatomical [-h] [options]",
    )

    parser.set_defaults(func=lambda args: main(args))
