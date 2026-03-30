"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli.query import iter_session_files, load_session

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

from pathlib import Path

import polars as pl
from tqdm import tqdm

from rbc.cli import _ANAT_GROUP_ENTITIES, _DEFAULT_ENV_VARS, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix, TemplateSpace, extract_entities
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
        sorted(df.group_by(_SUB_SES_QUERY)), disable=not ctx.verbose
    ):
        pipe_ctx = PipelineContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        for _, anat_df in iter_session_files(session, groupby=_ANAT_GROUP_ENTITIES):
            row = anat_df.filter(suffix="T1w").row(0, named=True)
            t1w_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["run", "acq", "rec", "echo"])
            ctx.logger.info(f"Processing {t1w_fpath}")

            outputs = single_session_preprocess(in_t1w=t1w_fpath)

            pipe_ctx = PipelineContext(
                sub=row["sub"], ses=row.get("ses"), output_dir=args.output_dir
            )
            anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=ents)
            anat.save(outputs.brain, suffix=Suffix.T1W, desc="brain")
            anat.save(outputs.brain_mask, suffix=Suffix.MASK, desc="T1w")
            anat.save(outputs.csf_mask, suffix=Suffix.MASK, desc="csf")
            anat.save(outputs.gm_mask, suffix=Suffix.MASK, desc="gm")
            anat.save(outputs.wm_mask, suffix=Suffix.MASK, desc="wm")
            anat.save(outputs.wm_bbr_mask, suffix=Suffix.MASK, desc="wmBBR")
            anat.save(
                outputs.forward_xfm,
                suffix="xfm",
                extra={
                    "from": "T1w",
                    "to": TemplateSpace.MNI152NLIN6ASYM,
                    "mode": "image",
                },
            )
            anat.save(
                outputs.inverse_xfm,
                suffix="xfm",
                extra={
                    "from": TemplateSpace.MNI152NLIN6ASYM,
                    "to": "T1w",
                    "mode": "image",
                },
            )
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
