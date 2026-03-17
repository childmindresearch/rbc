"""CLI subcommand for quality control.

Reads preprocessed derivatives from ``output_dir`` and computes QC metrics
(framewise displacement, DVARS, registration overlap, etc.), generating
per-run XCP-D-format TSVs with pass/fail flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix, TemplateSpace
from rbc.core.bids2table import get_file_path, load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.qc import single_session_qc

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class QCArgs(BaseArgs):
    """Arguments for the QC CLI."""

    task: str | None
    start_tr: int
    regressor: Literal["36-parameter", "aCompCor"]

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> QCArgs:
        """Validate QC-specific arguments."""
        _validate_task(ns.task)
        _validate_positive(ns.start_tr, "Start TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            task=ns.task,
            start_tr=ns.start_tr,
            regressor=ns.regressor,
        )


def main(args: QCArgs) -> int:
    """Main entrypoint of QC workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC QC workflow")
    df = load_table(
        dataset_dir=args.output_dir,
        index_fpath=None,
        max_workers=0,
        verbose=ctx.verbose,
    )

    filters = [
        pl.col("datatype") == "func",
        pl.col("suffix") == "bold",
        pl.col("desc") == "preproc",
        pl.col("space") == TemplateSpace.MNI152NLIN6ASYM,
    ]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    if args.task is not None:
        filters.append(pl.col("task") == args.task)
    df = df.filter(pl.all_horizontal(filters))

    for _, group in tqdm(df.group_by(_SUB_SES_QUERY), disable=not ctx.verbose):
        sub: str = group["sub"][0]
        ses: str | None = group["ses"][0] or None
        pipe_ctx = PipelineContext(sub=sub, ses=ses, output_dir=args.output_dir)

        deriv_df = load_table(
            dataset_dir=args.output_dir, index_fpath=None, max_workers=0, verbose=False
        )
        get_deriv = partial(get_file_path, df=deriv_df, sub=sub, ses=ses)

        for _, run_group in group.group_by(("run", "task")):
            row = run_group.row(0, named=True)
            bold_task: str | None = row.get("task")
            bold_run: int | None = row.get("run")

            template_bold = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.BOLD,
                desc="preproc",
                space=TemplateSpace.MNI152NLIN6ASYM,
                task=bold_task,
                run=bold_run,
            )
            cleaned_bold = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.BOLD,
                desc="preproc",
                space=TemplateSpace.MNI152NLIN6ASYM,
                task=bold_task,
                run=bold_run,
                extra={"reg": args.regressor},
            )
            motion_params = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.MOTION,
                desc="motionParams",
                extension=".1D",
                task=bold_task,
                run=bold_run,
            )
            rms_rel = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.MOTION,
                desc="relsDisplacement",
                extension=".rms",
                task=bold_task,
                run=bold_run,
            )
            bold_mask = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.MASK,
                desc="brain",
                task=bold_task,
                run=bold_run,
            )
            brain_mask = get_deriv(
                datatype=Datatype.ANAT,
                suffix=Suffix.MASK,
                desc="T1w",
            )
            bold_to_anat_matrix = get_deriv(
                datatype=Datatype.FUNC,
                suffix="xfm",
                desc="linear",
                extension=".mat",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
                task=bold_task,
                run=bold_run,
            )
            template_brain_mask = get_deriv(
                datatype=Datatype.FUNC,
                suffix=Suffix.MASK,
                desc="bold",
                space=TemplateSpace.MNI152NLIN6ASYM,
                task=bold_task,
                run=bold_run,
            )

            qc_outputs = single_session_qc(
                template_bold=template_bold,
                cleaned_bold=cleaned_bold,
                motion_params=motion_params,
                rms_rel=rms_rel,
                bold_mask=bold_mask,
                brain_mask=brain_mask,
                bold_to_anat_matrix=bold_to_anat_matrix,
                template_brain_mask=template_brain_mask,
                sub=sub,
                ses=ses or "",
                task=bold_task or "",
                run=bold_run or 0,
                start_tr=args.start_tr,
                regressor_set=args.regressor,
            )

            pipe_ctx.export(
                qc_outputs.qc_file,
                datatype=Datatype.FUNC,
                suffix="quality",
                desc="xcp",
                extension=".tsv",
                space=TemplateSpace.MNI152NLIN6ASYM,
                task=bold_task,
                run=bold_run,
                extra={"reg": args.regressor},
            )

            status = "PASSED" if qc_outputs.passed else "FAILED"
            ctx.logger.info(
                f"QC {status} for sub-{sub} ses-{ses} task-{bold_task} run-{bold_run}"
            )
        pipe_ctx.ensure_dataset_description()

    ctx.logger.info("RBC QC workflow complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register QC workflow to parser."""
    parser = subparsers.add_parser(
        "qc",
        parents=parents,
        description="RBC quality control workflow",
        help="QC workflow (motion, DVARS, registration overlap)",
        usage="rbc input_dir output_dir qc [-h] [options]",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--start-tr",
        type=int,
        default=2,
        help="Number of initial TRs discarded during preprocessing.",
    )
    parser.add_argument(
        "--regressor",
        choices=["36-parameter", "aCompCor"],
        default="36-parameter",
        help="Nuisance regression method used in functional preprocessing.",
    )

    parser.set_defaults(func=lambda args: main(QCArgs.validate_namespace(args)))
