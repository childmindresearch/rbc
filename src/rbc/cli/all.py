"""CLI subcommand that chains all four workflows.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory without disk round-trips
between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs, _validate_atlas, _validate_positive, _validate_task
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix, TemplateSpace
from rbc.core.bids2table import load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.anatomical import single_session_preprocess as anatomical_preprocess
from rbc.workflows.functional import single_session_preprocess as functional_preprocess
from rbc.workflows.metrics import single_session_metrics as metrics_pipeline
from rbc.workflows.qc import single_session_qc as qc_pipeline

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from rbc_resources import AtlasName


@dataclass(frozen=True)
class AllArgs(BaseArgs):
    """Arguments for the combined pipeline CLI."""

    regressor: Literal["36-parameter", "aCompCor"]
    task: str | None
    atlas: AtlasName
    fwhm: float
    start_tr: int

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AllArgs:
        """Validate all-workflow arguments."""
        _validate_task(ns.task)
        _validate_atlas(ns.atlas)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.start_tr, "Start TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,
            task=ns.task,
            atlas=ns.atlas,
            fwhm=ns.fwhm,
            start_tr=ns.start_tr,
        )


def main(args: AllArgs) -> int:
    """Main entrypoint of combined pipeline."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC full pipeline")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = []
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

        # --- Anatomical (once per session, first T1w) ---
        anat_row = session.anat.row(0, named=True)
        t1w_fpath = Path(anat_row["root"]) / anat_row["path"]
        ctx.logger.info(f"Anatomical: {t1w_fpath}")

        anat_outputs = anatomical_preprocess(in_t1w=t1w_fpath)

        pipe_ctx.export(
            anat_outputs.brain,
            datatype=Datatype.ANAT,
            suffix=Suffix.T1W,
            desc="brain",
        )
        pipe_ctx.export(
            anat_outputs.brain_mask,
            datatype=Datatype.ANAT,
            suffix=Suffix.MASK,
            desc="T1w",
        )
        pipe_ctx.export(
            anat_outputs.csf_mask,
            datatype=Datatype.ANAT,
            suffix=Suffix.MASK,
            desc="csf",
        )
        pipe_ctx.export(
            anat_outputs.gm_mask,
            datatype=Datatype.ANAT,
            suffix=Suffix.MASK,
            desc="gm",
        )
        pipe_ctx.export(
            anat_outputs.wm_mask,
            datatype=Datatype.ANAT,
            suffix=Suffix.MASK,
            desc="wm",
        )
        pipe_ctx.export(
            anat_outputs.wm_bbr_mask,
            datatype=Datatype.ANAT,
            suffix=Suffix.MASK,
            desc="wmBBR",
        )
        pipe_ctx.export(
            anat_outputs.forward_xfm,
            datatype=Datatype.ANAT,
            suffix="xfm",
            extra={"from": "T1w", "to": "template", "mode": "image"},
        )
        pipe_ctx.export(
            anat_outputs.inverse_xfm,
            datatype=Datatype.ANAT,
            suffix="xfm",
            extra={"from": "template", "to": "T1w", "mode": "image"},
        )

        # --- Functional + Metrics + QC (per BOLD run) ---
        for func_df, _anat_df in iter_session_files(session, groupby=("run", "task")):
            row = func_df.row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            bold_task: str | None = row.get("task")
            bold_run: int | None = row.get("run")
            ctx.logger.info(f"Functional: {bold_fpath}")

            func_outputs = functional_preprocess(
                in_bold=bold_fpath,
                t1w_brain=anat_outputs.brain,
                wm_bbr_mask=anat_outputs.wm_bbr_mask,
                brain_mask=anat_outputs.brain_mask,
                csf_mask=anat_outputs.csf_mask,
                wm_mask=anat_outputs.wm_mask,
                anat_to_template=anat_outputs.inverse_xfm,
                start_tr=args.start_tr,
                regressor_set=args.regressor,
            )

            # Export functional outputs
            _fex = partial(
                pipe_ctx.export,
                datatype=Datatype.FUNC,
                task=bold_task,
                run=bold_run,
            )
            _fex(func_outputs.sbref, suffix=Suffix.SBREF)
            _fex(
                func_outputs.preproc_bold,
                desc="preproc",
                suffix=Suffix.BOLD,
            )
            _fex(
                func_outputs.motion_params,
                desc="motionParams",
                suffix=Suffix.MOTION,
                extension=".1D",
            )
            _fex(
                func_outputs.rms_rel,
                desc="relsDisplacement",
                suffix=Suffix.MOTION,
                extension=".rms",
            )
            _fex(
                func_outputs.rms_abs,
                desc="maxDisplacement",
                suffix=Suffix.MOTION,
                extension=".rms",
            )
            _fex(func_outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
            _fex(
                func_outputs.bold_to_anat_matrix,
                suffix="xfm",
                desc="linear",
                extension=".mat",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            _fex(
                func_outputs.regressor_file,
                desc=args.regressor,
                suffix="regressors",
                extension=".1D",
            )
            _fex(
                func_outputs.template_bold,
                space=TemplateSpace.MNI152NLIN6ASYM,
                desc="preproc",
                suffix=Suffix.BOLD,
            )
            _fex(
                func_outputs.regressed_bold,
                space=TemplateSpace.MNI152NLIN6ASYM,
                desc="regressed",
                suffix=Suffix.BOLD,
                extra={"reg": args.regressor},
            )
            _fex(
                func_outputs.cleaned_bold,
                space=TemplateSpace.MNI152NLIN6ASYM,
                desc="preproc",
                suffix=Suffix.BOLD,
                extra={"reg": args.regressor},
            )
            _fex(
                func_outputs.template_brain_mask,
                space=TemplateSpace.MNI152NLIN6ASYM,
                desc="bold",
                suffix=Suffix.MASK,
            )

            # --- Metrics ---
            ctx.logger.info(
                f"Metrics: sub-{pipe_ctx.sub} task-{bold_task} run-{bold_run}"
            )
            metrics_outputs = metrics_pipeline(
                regressed_bold=func_outputs.regressed_bold,
                cleaned_bold=func_outputs.cleaned_bold,
                template_brain_mask=func_outputs.template_brain_mask,
                atlas=args.atlas,
                fwhm=args.fwhm,
            )

            reg_extra: dict[str, str | int] = {"reg": args.regressor}
            _mex = partial(
                pipe_ctx.export,
                datatype=Datatype.FUNC,
                space=TemplateSpace.MNI152NLIN6ASYM,
                task=bold_task,
                run=bold_run,
                extra=reg_extra,
            )
            _mex(metrics_outputs.alff, suffix="alff")
            _mex(metrics_outputs.falff, suffix="falff")
            _mex(metrics_outputs.alff_smooth, suffix="alff", desc="smooth")
            _mex(metrics_outputs.falff_smooth, suffix="falff", desc="smooth")
            _mex(
                metrics_outputs.alff_zscored,
                suffix="alff",
                desc="smoothZstd",
            )
            _mex(
                metrics_outputs.falff_zscored,
                suffix="falff",
                desc="smoothZstd",
            )
            _mex(metrics_outputs.reho, suffix="reho")
            _mex(metrics_outputs.reho_smooth, suffix="reho", desc="smooth")
            _mex(
                metrics_outputs.reho_zscored,
                suffix="reho",
                desc="smoothZstd",
            )
            _mex(
                metrics_outputs.timeseries,
                suffix="timeseries",
                desc="mean",
                extension=".tsv",
                atlas=args.atlas,
            )
            _mex(
                metrics_outputs.correlation_matrix,
                suffix="correlations",
                desc="pearson",
                extension=".tsv",
                atlas=args.atlas,
            )

            # --- QC ---
            ctx.logger.info(f"QC: sub-{pipe_ctx.sub} task-{bold_task} run-{bold_run}")
            qc_outputs = qc_pipeline(
                template_bold=func_outputs.template_bold,
                cleaned_bold=func_outputs.cleaned_bold,
                motion_params=func_outputs.motion_params,
                rms_rel=func_outputs.rms_rel,
                bold_mask=func_outputs.bold_mask,
                brain_mask=anat_outputs.brain_mask,
                bold_to_anat_matrix=func_outputs.bold_to_anat_matrix,
                template_brain_mask=func_outputs.template_brain_mask,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses or "",
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
                f"QC {status} for sub-{pipe_ctx.sub} task-{bold_task} run-{bold_run}"
            )
        pipe_ctx.ensure_dataset_description()

    ctx.logger.info("RBC full pipeline complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register combined pipeline to parser."""
    parser = subparsers.add_parser(
        "all",
        parents=parents,
        description="RBC full pipeline (anatomical + functional + metrics + QC)",
        help="Full pipeline (all workflows)",
        usage="rbc input_dir output_dir all [-h] [options]",
    )
    parser.add_argument(
        "--regressor",
        choices=["36-parameter", "aCompCor"],
        default="36-parameter",
        help="Nuisance regression method.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--atlas",
        choices=[
            "schaefer_200",
            "schaefer_300",
            "schaefer_400",
            "schaefer_1000",
            "aal",
        ],
        default="schaefer_200",
        help="Atlas for timeseries extraction.",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=6.0,
        help="Smoothing kernel FWHM in mm.",
    )
    parser.add_argument(
        "--start-tr",
        type=int,
        default=2,
        help="Number of initial TRs to discard.",
    )

    parser.set_defaults(func=lambda args: main(AllArgs.validate_namespace(args)))
