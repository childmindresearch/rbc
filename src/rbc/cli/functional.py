"""CLI subcommand for functional processing.

Parses subject/session/task arguments and delegates to
``rbc.workflows.functional.single_session_preprocess``, which runs the functional
stream (reorientation -> TR truncation -> motion correction). Anatomical
preprocessing must be completed first since coregistration and template
warping depend on the anatomical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.bids import (
    FUNC_GROUP_ENTITIES,
    SUB_SES_QUERY,
    Datatype,
    extract_entities,
    load_table,
)
from rbc.bids.functional import export_functional, resolve_functional
from rbc.bids.session import iter_session_files, load_session
from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.context import RunContext
from rbc.core.niwrap import setup_runner
from rbc.metadata import FunctionalMetadata
from rbc.workflows.functional import single_session_preprocess

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class FunctionalArgs(BaseArgs):
    """Arguments for single-session functional CLI."""

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    tr: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalArgs:
        """Validation of functional workflow specific arguments to NamedTuple."""
        _validate_task(ns.task)
        _validate_positive(ns.tr, "TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,  # Validated by argparse choices
            task=ns.task,
            tr=ns.tr,
        )


def main(args: FunctionalArgs) -> int:
    """Main entrypoint of functional workflow."""
    # Setup
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC functional workflow")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = [pl.col("ses") != "longitudinal", pl.col("space").is_null()]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    if args.task is not None:
        filters.append(pl.col("task") == args.task)
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

        for func_df, anat_df in iter_session_files(
            session, groupby=FUNC_GROUP_ENTITIES
        ):
            func_df = func_df.filter(pl.col("desc").is_null())
            row = func_df.filter(suffix="bold").row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])
            ctx.logger.info(f"Processing {bold_fpath}")

            anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
            resolved = resolve_functional(anat_q, anat_df)

            func_metadata = FunctionalMetadata.load(bold_fpath, tr_override=args.tr)

            outputs = single_session_preprocess(
                in_bold=bold_fpath,
                t1w_brain=resolved["t1w_brain"],
                wm_bbr_mask=resolved["wm_bbr_mask"],
                brain_mask=resolved["brain_mask"],
                csf_mask=resolved["csf_mask"],
                wm_mask=resolved["wm_mask"],
                anat_to_template=resolved["anat_to_template"],
                metadata=func_metadata,
                regressor_set=args.regressor,
            )

            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            export_functional(func, outputs, regressors=args.regressor)

        pipe_ctx.ensure_dataset_description()

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
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Space-delimited nuisance regression method(s) to apply.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides BIDS sidecar and NIfTI header.",
    )

    parser.set_defaults(func=lambda args: main(FunctionalArgs.validate_namespace(args)))
