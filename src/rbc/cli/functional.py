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

from rbc.cli import _DEFAULT_ENV_VARS, _FUNC_GROUP_ENTITIES, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs, _validate_task
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix, TemplateSpace, extract_entities
from rbc.core.bids2table import load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.functional import single_session_preprocess

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class FunctionalArgs(BaseArgs):
    """Arguments for single-session functional CLI."""

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalArgs:
        """Validation of functional workflow specific arguments to NamedTuple."""
        _validate_task(ns.task)
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,  # Validated by argparse choices
            task=ns.task,
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

    filters = [pl.col("space").is_null(), pl.col("desc").is_null()]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    if args.task is not None:
        filters.append(pl.col("task") == args.task)
    if filters:
        df = df.filter(pl.all_horizontal(filters))

    for _, sub_ses_group in tqdm(df.group_by(_SUB_SES_QUERY), disable=not ctx.verbose):
        pipe_ctx = PipelineContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        for func_df, anat_df in iter_session_files(
            session, groupby=_FUNC_GROUP_ENTITIES
        ):
            row = func_df.filter(suffix="bold").row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])
            ctx.logger.info(f"Processing {bold_fpath}")

            anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)

            outputs = single_session_preprocess(
                in_bold=bold_fpath,
                t1w_brain=anat_q.expect(anat_df, suffix=Suffix.T1W, desc="brain"),
                wm_bbr_mask=anat_q.expect(anat_df, suffix=Suffix.MASK, desc="wmBBR"),
                brain_mask=anat_q.expect(anat_df, suffix=Suffix.MASK, desc="T1w"),
                csf_mask=anat_q.expect(anat_df, suffix=Suffix.MASK, desc="csf"),
                wm_mask=anat_q.expect(anat_df, suffix=Suffix.MASK, desc="wm"),
                anat_to_template=anat_q.expect(
                    anat_df,
                    suffix="xfm",
                    extra={
                        "from": TemplateSpace.MNI152NLIN6ASYM,
                        "to": "T1w",
                        "mode": "image",
                    },
                ),
                regressor_set=args.regressor,
            )

            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func.save(outputs.sbref, suffix=Suffix.SBREF)
            func.save(outputs.preproc_bold, suffix=Suffix.BOLD, desc="preproc")
            func.save(
                outputs.motion_params,
                suffix=Suffix.MOTION,
                desc="motionParams",
                extension=".1D",
            )
            func.save(
                outputs.rms_rel,
                suffix=Suffix.MOTION,
                desc="relsDisplacement",
                extension=".rms",
            )
            func.save(
                outputs.rms_abs,
                suffix=Suffix.MOTION,
                desc="maxDisplacement",
                extension=".rms",
            )
            func.save(outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
            func.save(
                outputs.bold_to_anat_matrix,
                suffix="xfm",
                desc="linear",
                extension=".txt",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            func.save(
                outputs.bold_to_anat_itk,
                suffix="xfm",
                desc="linearITK",
                extension=".txt",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            for regressor in args.regressor:
                func.save(
                    outputs.regressor_file[regressor],
                    suffix="regressors",
                    desc=regressor,
                    extension=".1D",
                )

            mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)
            for regressor in args.regressor:
                mni.save(
                    outputs.cleaned_bold[regressor],
                    suffix=Suffix.BOLD,
                    desc="preproc",
                    extra={"reg": regressor},
                )
            mni.save(outputs.template_bold, suffix=Suffix.BOLD, desc="preproc")
            mni.save(outputs.template_brain_mask, suffix=Suffix.MASK, desc="bold")

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

    parser.set_defaults(func=lambda args: main(FunctionalArgs.validate_namespace(args)))
