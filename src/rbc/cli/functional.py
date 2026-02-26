"""CLI subcommand for functional processing.

Parses subject/session/task arguments and delegates to
``rbc.workflows.functional.single_session_preprocess``, which runs the functional
stream (reorientation -> TR truncation -> motion correction). Anatomical
preprocessing must be completed first since coregistration and template
warping depend on the anatomical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _SUB_SES_QUERY
from rbc.cli.main import BaseArgs
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids2table import get_file_path, load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.functional import single_session_preprocess

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class FunctionalArgs(BaseArgs):
    """Arguments for single-session functional CLI."""

    regressor: Literal["36-parameter", "aCompCor"]

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalArgs:
        """Validation of functional workflow specific arguments to NamedTuple."""
        return cls(**BaseArgs.validate_namespace(ns).__dict__, regressor=ns.regressor)


def main(args: FunctionalArgs) -> int:
    """Main entrypoint of functional workflow."""
    # Setup
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC functional workflow")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = []
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    df = df.filter(pl.all_horizontal(filters))

    for _, sub_ses_group in tqdm(df.group_by(_SUB_SES_QUERY), disable=not ctx.verbose):
        pipe_ctx = PipelineContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        for func_df, anat_df in iter_session_files(session, groupby=("run", "task")):
            row = func_df.row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            ctx.logger.info(f"Processing {bold_fpath}")

            get_anat_file = partial(
                get_file_path,
                df=anat_df,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses,
                datatype="anat",
            )
            outputs = single_session_preprocess(
                in_bold=bold_fpath,
                t1w_brain=get_anat_file(suffix="T1w", desc="brain"),
                wm_bbr_mask=get_anat_file(suffix="mask", desc="wmBBR"),
                brain_mask=get_anat_file(suffix="mask", desc="T1w"),
                csf_mask=get_anat_file(suffix="mask", desc="csf"),
                wm_mask=get_anat_file(suffix="mask", desc="wm"),
                anat_to_template=get_anat_file(
                    suffix="xfm",
                    extra={"from": "template", "to": "T1w", "mode": "image"},
                ),
                regressor_set=args.regressor,
            )

            pipe_ctx.export(outputs.sbref, datatype="func", suffix="sbref")
            pipe_ctx.export(
                outputs.motion_corrected_bold,
                datatype="func",
                desc="preproc",
                suffix="bold",
            )
            pipe_ctx.export(
                outputs.motion_params,
                datatype="func",
                desc="motionParams",
                suffix="motion",
                extension=".1D",
            )
            pipe_ctx.export(
                outputs.rms_rel,
                datatype="func",
                desc="relsDisplacement",
                suffix="motion",
                extension=".rms",
            )
            pipe_ctx.export(
                outputs.rms_abs,
                datatype="func",
                desc="maxDisplacement",
                suffix="motion",
                extension=".rms",
            )
            pipe_ctx.export(
                outputs.bold_mask, datatype="func", suffix="mask", desc="brain"
            )
            pipe_ctx.export(
                outputs.bold_to_anat_matrix,
                datatype="func",
                suffix="xfm",
                desc="linear",
                extension=".mat",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            pipe_ctx.export(
                outputs.regressor_file,
                datatype="func",
                desc=args.regressor,
                suffix="regressors",
                extension=".1D",
            )

    ctx.logger.info("RBC functional workflow complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register functional workflow to parser."""
    parser = subparsers.add_parser(
        "functional",
        parents=parents,
        description="RBC functional workflow",
        help="Functional workflow",
        usage="rbc input_dir output_dir functional [-h] [options]",
    )
    parser.add_argument(
        "--regressor",
        choices=["36-parameter", "aCompCor"],
        default="36-parameter",
        help="Nuisance regression method.",
    )

    parser.set_defaults(func=lambda args: main(args))
