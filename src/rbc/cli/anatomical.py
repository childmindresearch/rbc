"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

from pathlib import Path

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.__main__ import BaseArgs
from rbc.core.bids2table import load_table
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
    group_entity = ("sub", "ses", "run")

    # Setup
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC anatomical workflow")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = [
        pl.col("datatype") == "anat",
        pl.col("suffix") == "T1w",
        pl.col("ext").str.contains(".nii"),
    ]
    if len(args.participant_labels) > 0:
        filters.append(pl.col("sub").is_in(args.participant_labels))
    if len(args.session_labels) > 0:
        filters.append(pl.col("ses").is_in(args.session_labels))
    df = df.filter(pl.all_horizontal(filters))

    for _, group in tqdm(df.group_by(group_entity), disable=not ctx.verbose):
        for row in group.iter_rows(named=True):
            t1w_fpath = Path(row["root"]) / row["path"]
            ctx.logger.info(f"Processing {t1w_fpath}")
            single_session_preprocess(in_t1w=t1w_fpath, output_dir=args.output_dir)

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
