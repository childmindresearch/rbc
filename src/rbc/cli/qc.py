"""CLI subcommand for quality control.

Reads preprocessed derivatives from ``output_dir`` and computes QC metrics
(framewise displacement, DVARS, registration overlap, etc.), generating
per-run XCP-D-format TSVs with pass/fail flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _FUNC_GROUP_ENTITIES, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix, TemplateSpace, extract_entities
from rbc.core.bids2table import load_table
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

        for _, run_group in group.group_by(_FUNC_GROUP_ENTITIES):
            row = run_group.row(0, named=True)
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])

            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func_mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)

            template_bold = func_mni.find(deriv_df, suffix=Suffix.BOLD, desc="preproc")
            cleaned_bold = func_mni.find(
                deriv_df,
                suffix=Suffix.BOLD,
                desc="preproc",
                extra={"reg": args.regressor},
            )
            motion_params = func.find(
                deriv_df, suffix=Suffix.MOTION, desc="motionParams", extension=".1D"
            )
            rms_rel = func.find(
                deriv_df,
                suffix=Suffix.MOTION,
                desc="relsDisplacement",
                extension=".rms",
            )
            bold_mask = func.find(deriv_df, suffix=Suffix.MASK, desc="brain")
            brain_mask = pipe_ctx.bids(datatype=Datatype.ANAT).find(
                deriv_df, suffix=Suffix.MASK, desc="T1w"
            )
            bold_to_anat_matrix = func.find(
                deriv_df,
                suffix="xfm",
                desc="linear",
                extension=".mat",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            template_brain_mask = func_mni.find(
                deriv_df, suffix=Suffix.MASK, desc="bold"
            )

            bold_task: str | None = row.get("task")
            bold_run: int | None = row.get("run")

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

            func_mni.save(
                qc_outputs.qc_file,
                suffix="quality",
                desc="xcp",
                extension=".tsv",
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
