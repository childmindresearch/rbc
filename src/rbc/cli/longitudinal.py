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

from rbc.cli import _DEFAULT_ENV_VARS, _FUNC_GROUP_ENTITIES, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Extension, Suffix
from rbc.core.bids2table import get_file_path, load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.anatomical import longitudinal_process as anatomical_longitudinal
from rbc.workflows.functional import longitudinal_process as functional_longitudinal


@dataclass(frozen=True)
class LongitudinalArgs(BaseArgs):
    """Arguments for longitudinal CLI."""

    anatomical: bool
    functional: bool

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> LongitudinalArgs:
        """Validation of longitudinal workflow specific arguments to NamedTuple."""
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
                datatype=Datatype.ANAT,
                **kwargs,
            )
        except FileNotFoundError:
            return None

    _get_tpl_file = partial(
        get_file_path,
        df=tpl_df,
        sub=pipe_ctx.sub,
        ses="longitudinal",
        datatype=Datatype.ANAT,
    )

    outputs = anatomical_longitudinal(
        template=_get_tpl_file(suffix=Suffix.T1W),
        subj_to_template_xfm=_get_tpl_file(
            suffix="xfm",
            extension=".txt",
            extra={"from": pipe_ctx.ses},  # type: ignore [dict-item]
        ),
        brain=_require_file(_get_anat_file(suffix=Suffix.T1W, desc="brain"), "brain"),
        brain_mask=_get_anat_file(suffix=Suffix.MASK, desc="T1w"),
        csf_mask=_get_anat_file(suffix=Suffix.MASK, desc="csf"),
        gm_mask=_get_anat_file(suffix=Suffix.MASK, desc="gm"),
        wm_mask=_get_anat_file(suffix=Suffix.MASK, desc="wm"),
    )
    # Save longitudinal outputs
    _aex = partial(
        pipe_ctx.export, datatype=Datatype.ANAT, space="longitudinal", run=t1w_run
    )
    _aex(outputs.brain, desc="brain", suffix=Suffix.T1W)
    _aex(
        _require_file(outputs.brain_mask, "brain_mask"), desc="T1w", suffix=Suffix.MASK
    )
    _aex(_require_file(outputs.csf_mask, "csf_mask"), desc="csf", suffix=Suffix.MASK)
    _aex(_require_file(outputs.csf_mask, "gm_mask"), desc="gm", suffix=Suffix.MASK)
    _aex(_require_file(outputs.csf_mask, "wm_mask"), desc="wm", suffix=Suffix.MASK)
    _aex(
        outputs.forward_xfm,
        suffix="xfm",
        extra={"from": "T1w", "to": "longitudinal", "mode": "image"},
    )
    _aex(
        outputs.inverse_xfm,
        suffix="xfm",
        extra={"from": "longitudinal", "to": "T1w", "mode": "image"},
    )


def _process_func(
    pipe_ctx: PipelineContext, func_df: pl.DataFrame, tpl_df: pl.DataFrame
) -> None:
    """Handle functional longitudinal processing."""
    row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
    bold_task: str | None = row.get("task")
    bold_run: int | None = row.get("run")

    # Grab files
    def _get_func_file(**kwargs) -> Path | None:  # noqa: ANN003 - bids arguments
        try:
            return get_file_path(
                df=func_df,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses,
                datatype=Datatype.FUNC,
                run=bold_run,
                task=bold_task,
                **kwargs,
            )
        except FileNotFoundError:
            return None

    get_tpl_file = partial(
        get_file_path, df=tpl_df, ses="longitudinal", sub=pipe_ctx.sub, datatype="anat"
    )
    # Use native space data
    outputs = functional_longitudinal(
        template=get_tpl_file(suffix="T1w"),
        anat_to_template_xfm=get_tpl_file(
            suffix="xfm",
            extension=".mat",
            extra={"from": pipe_ctx.ses},  # type: ignore [dict-item]
        ),
        bold_to_anat_xfm=_require_file(
            _get_func_file(
                suffix="xfm",
                desc="linear",
                extension=".mat",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            ),
            "bold_to_anat_xfm",
        ),
        sbref=_require_file(
            _get_func_file(space=False, suffix=Suffix.SBREF), Suffix.SBREF
        ),
        bold=_require_file(
            _get_func_file(space=False, desc="preproc", suffix=Suffix.BOLD), Suffix.BOLD
        ),
        bold_mask=_get_func_file(space=False, desc="brain", suffix=Suffix.MASK),
    )
    # Save longitudinal outputs
    pipe_ctx.export(
        outputs.sbref,
        datatype=Datatype.FUNC,
        space="longitudinal",
        suffix=Suffix.SBREF,
        task=bold_task,
        run=bold_run,
    )
    pipe_ctx.export(
        outputs.bold,
        datatype=Datatype.FUNC,
        space="longitudinal",
        desc="preproc",
        suffix=Suffix.BOLD,
        task=bold_task,
        run=bold_run,
    )
    pipe_ctx.export(
        outputs.forward_xfm,
        datatype=Datatype.FUNC,
        suffix="xfm",
        desc="composite",
        extension=Extension.NII_GZ,
        extra={"from": "bold", "to": "longitudinal", "mode": "image"},
        task=bold_task,
        run=bold_run,
    )
    if outputs.bold_mask:
        pipe_ctx.export(
            outputs.bold_mask,
            datatype=Datatype.FUNC,
            space="longitudinal",
            desc="brain",
            suffix=Suffix.MASK,
            task=bold_task,
            run=bold_run,
        )


def main(args: LongitudinalArgs) -> int:
    """Main entrypoint of longitudinal workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
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
    filters = [pl.col("ses") != "longitudinal"]
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

        for func_df, anat_df in iter_session_files(
            session, groupby=_FUNC_GROUP_ENTITIES
        ):
            if args.anatomical:
                anat_df = anat_df.filter(
                    pl.col("space").is_null() | (pl.col("space") != "longitudinal")
                )
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
