"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids2table import get_file_path, load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.anatomical import longitudinal_process as anatomical_longitudinal


@dataclass(frozen=True)
class LongitudinalArgs(BaseArgs):
    """Arguments for longitudinal CLI."""

    anatomical: bool
    functional: bool

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> LongitudinalArgs:
        """Validation of longitudinal workflow specific arguments to NamedTuple."""
        if ns.functional:
            raise NotImplementedError(
                "Functional longitudinal pipeline not yet implemented."
            )
        if not ns.functional and not ns.anatomical:
            raise ValueError(
                "At least one of '--anatomical' or '--functional' is required."
            )
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            anatomical=ns.anatomical,
            functional=ns.functional,
        )


def _require_file(path: Path | None, field: str) -> Path:
    if path is None:
        raise ValueError(f"Expected output {field!r} is missing.")
    return path


def _process_anat(
    pipe_ctx: PipelineContext, anat_df: pl.DataFrame, tpl_df: pl.DataFrame
) -> None:
    """Handle anatomical longitudinal processing."""
    row = anat_df.filter(suffix="T1w").row(0, named=True)
    t1w_run: int | None = row.get("run")

    # Grab files
    def _get_anat_file(**kwargs) -> Path | None:  # noqa: ANN003 - bids arguments
        try:
            return get_file_path(
                df=anat_df,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses,
                datatype="anat",
                **kwargs,
            )
        except FileNotFoundError:
            return None

    _get_tpl_file = partial(
        get_file_path, df=tpl_df, sub=pipe_ctx.sub, ses="longitudinal", datatype="anat"
    )

    outputs = anatomical_longitudinal(
        template=_get_tpl_file(suffix="T1w"),
        subj_to_template_xfm=_get_tpl_file(
            suffix="xfm",
            extension=".mat",
            extra={"from": pipe_ctx.ses},  # type: ignore [dict-item]
        ),
        brain=_require_file(_get_anat_file(suffix="T1w", desc="brain"), "brain"),
        brain_mask=_get_anat_file(suffix="mask", desc="T1w"),
        csf_mask=_get_anat_file(suffix="mask", desc="csf"),
        gm_mask=_get_anat_file(suffix="mask", desc="gm"),
        wm_mask=_get_anat_file(suffix="mask", desc="wm"),
    )
    # Save longitudinal outputs
    pipe_ctx.export(
        outputs.brain,
        datatype="anat",
        space="longitudinal",
        desc="brain",
        suffix="T1w",
        run=t1w_run,
    )
    pipe_ctx.export(
        _require_file(outputs.brain_mask, "brain_mask"),
        datatype="anat",
        space="longitudinal",
        desc="T1w",
        suffix="mask",
        run=t1w_run,
    )
    pipe_ctx.export(
        _require_file(outputs.csf_mask, "csf_mask"),
        datatype="anat",
        space="longitudinal",
        desc="csf",
        suffix="mask",
        run=t1w_run,
    )
    pipe_ctx.export(
        _require_file(outputs.gm_mask, "gm_mask"),
        datatype="anat",
        space="longitudinal",
        desc="gm",
        suffix="mask",
        run=t1w_run,
    )
    pipe_ctx.export(
        _require_file(outputs.wm_mask, "wm_mask"),
        datatype="anat",
        space="longitudinal",
        desc="wm",
        suffix="mask",
        run=t1w_run,
    )
    pipe_ctx.export(
        outputs.forward_xfm,
        datatype="anat",
        suffix="xfm",
        extra={"from": "T1w", "to": "longitudinal", "mode": "image"},
        run=t1w_run,
    )
    pipe_ctx.export(
        outputs.inverse_xfm,
        datatype="anat",
        suffix="xfm",
        extra={"from": "longitudinal", "to": "T1w", "mode": "image"},
        run=t1w_run,
    )


def _process_func(
    pipe_ctx: PipelineContext, func_df: pl.DataFrame, tpl_df: pl.DataFrame
) -> None:
    """Handle longitudinal functional processing."""


def main(args: LongitudinalArgs) -> int:
    """Main entrypoint of longitudinal workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.warning(
        "This workflow is experimental and may be sensitive to input file "
        "naming conventions."
    )
    ctx.logger.info("Preparing to run RBC-based longitudinal workflow")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    group_df = df
    filters = [pl.col("ses") != "longitudinal", pl.col("space") != "longitudinal"]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    if filters:
        group_df = df.filter(pl.all_horizontal(filters))

    for _, sub_ses_group in tqdm(
        group_df.group_by(_SUB_SES_QUERY), disable=not ctx.verbose
    ):
        pipe_ctx = PipelineContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0],
            output_dir=args.output_dir,
        )
        if pipe_ctx.ses is None:
            raise ValueError(
                "No session data - unable to perform longitudinal processing"
            )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)
        tpl_df = df.filter(
            pl.all_horizontal(
                pl.col("sub") == pipe_ctx.sub, pl.col("ses") == "longitudinal"
            )
        )
        if tpl_df.is_empty():
            raise ValueError("No longitudinal template found")

        for func_df, anat_df in iter_session_files(session, groupby=("run", "task")):
            if args.anatomical:
                _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)
            if args.functional:
                _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
        pipe_ctx.ensure_dataset_description()

    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register longitudinal workflow to parser."""
    parser = subparsers.add_parser(
        "longitudinal",
        parents=parents,
        description="RBC-based longitudinal workflow",
        help="Longitudinal workflow",
        usage="rbc input_dir output_dir longitudinal [-h] [options]",
    )
    parser.add_argument(
        "--anatomical",
        default=False,
        action="store_true",
        help="Use anatomical longitudinal pipeline for processing",
    )
    parser.add_argument(
        "--functional",
        default=False,
        action="store_true",
        help="Use functional longitudinal pipeline for processing",
    )

    parser.set_defaults(
        func=lambda args: main(LongitudinalArgs.validate_namespace(args))
    )
