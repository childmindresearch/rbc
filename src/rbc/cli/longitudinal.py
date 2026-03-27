"""CLI subcommand for longitudinal processing."""

from __future__ import annotations

from dataclasses import dataclass
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
from rbc.core.bids import Datatype, Extension, Suffix, extract_entities
from rbc.core.bids2table import load_table
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
    ents = extract_entities(row, ["run"])

    anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
    tpl_pipe_ctx = PipelineContext(
        sub=pipe_ctx.sub, ses="longitudinal", output_dir=pipe_ctx.output_dir
    )
    tpl_q = tpl_pipe_ctx.bids(datatype=Datatype.ANAT)

    outputs = anatomical_longitudinal(
        template=tpl_q.expect(tpl_df, suffix=Suffix.T1W),
        subj_to_template_xfm=tpl_q.expect(
            tpl_df,
            suffix="xfm",
            extension=".txt",
            extra={"from": pipe_ctx.ses},  # type: ignore[dict-item]
        ),
        brain=anat_q.expect(anat_df, suffix=Suffix.T1W, desc="brain"),
        brain_mask=anat_q.find(anat_df, suffix=Suffix.MASK, desc="T1w"),
        csf_mask=anat_q.find(anat_df, suffix=Suffix.MASK, desc="csf"),
        gm_mask=anat_q.find(anat_df, suffix=Suffix.MASK, desc="gm"),
        wm_mask=anat_q.find(anat_df, suffix=Suffix.MASK, desc="wm"),
    )

    aex = pipe_ctx.bids(datatype=Datatype.ANAT, entities=ents, space="longitudinal")
    aex.save(outputs.brain, suffix=Suffix.T1W, desc="brain")
    aex.save(
        _require_file(outputs.brain_mask, "brain_mask"),
        suffix=Suffix.MASK,
        desc="T1w",
    )
    aex.save(
        _require_file(outputs.csf_mask, "csf_mask"), suffix=Suffix.MASK, desc="csf"
    )
    aex.save(_require_file(outputs.gm_mask, "gm_mask"), suffix=Suffix.MASK, desc="gm")
    aex.save(_require_file(outputs.wm_mask, "wm_mask"), suffix=Suffix.MASK, desc="wm")
    aex.save(
        outputs.forward_xfm,
        suffix="xfm",
        extra={"from": "T1w", "to": "longitudinal", "mode": "image"},
    )
    aex.save(
        outputs.inverse_xfm,
        suffix="xfm",
        extra={"from": "longitudinal", "to": "T1w", "mode": "image"},
    )


def _process_func(
    pipe_ctx: PipelineContext, func_df: pl.DataFrame, tpl_df: pl.DataFrame
) -> None:
    """Handle functional longitudinal processing."""
    row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
    ents = extract_entities(row, ["task", "run"])

    func_q = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
    tpl_pipe_ctx = PipelineContext(
        sub=pipe_ctx.sub, ses="longitudinal", output_dir=pipe_ctx.output_dir
    )
    tpl_q = tpl_pipe_ctx.bids(datatype=Datatype.ANAT)

    outputs = functional_longitudinal(
        template=tpl_q.expect(tpl_df, suffix="T1w"),
        anat_to_template_xfm=tpl_q.expect(
            tpl_df,
            suffix="xfm",
            extension=".txt",
            extra={"from": pipe_ctx.ses},  # type: ignore[dict-item]
        ),
        bold_to_anat_itk=func_q.expect(
            func_df,
            suffix="xfm",
            desc="linearITK",
            extension=".txt",
            extra={"from": "bold", "to": "T1w", "mode": "image"},
        ),
        sbref=func_q.expect(func_df, suffix=Suffix.SBREF, without=["space"]),
        bold=func_q.expect(
            func_df, suffix=Suffix.BOLD, desc="preproc", without=["space"]
        ),
        bold_mask=func_q.find(
            func_df, suffix=Suffix.MASK, desc="brain", without=["space"]
        ),
    )

    fex = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents, space="longitudinal")
    fex.save(outputs.sbref, suffix=Suffix.SBREF)
    fex.save(outputs.bold, suffix=Suffix.BOLD, desc="preproc")
    fex.save(
        outputs.forward_xfm,
        suffix="xfm",
        desc="composite",
        extension=Extension.NII_GZ,
        extra={"from": "bold", "to": "longitudinal", "mode": "image"},
    )
    if outputs.bold_mask:
        fex.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")


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
    filters = [
        pl.col("ses") != "longitudinal",
        pl.col("space").is_null(),
        pl.col("desc").is_null(),
    ]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
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
        session = load_session(df, pipe_ctx.sub, pipe_ctx.ses)
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
