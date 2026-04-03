"""CLI subcommand that chains all four workflows.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory without disk round-trips
between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from tqdm import tqdm

from rbc.bids import Datatype, extract_entities, load_table
from rbc.bids.anatomical import export_anatomical
from rbc.bids.functional import export_functional
from rbc.bids.metrics import export_metrics
from rbc.bids.qc import export_qc
from rbc.cli import (
    _ANAT_GROUP_ENTITIES,
    _DEFAULT_ENV_VARS,
    _FUNC_GROUP_ENTITIES,
    _SUB_SES_QUERY,
)
from rbc.cli.base import BaseArgs, _validate_atlas, _validate_positive, _validate_task
from rbc.cli.query import iter_session_files, load_session
from rbc.context import RunContext
from rbc.core.niwrap import setup_runner
from rbc.metadata import FunctionalMetadata
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

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    atlas: Sequence[AtlasName]
    fwhm: float
    start_tr: int
    tr: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AllArgs:
        """Validate all-workflow arguments."""
        _validate_task(ns.task)
        for atlas in ns.atlas:
            _validate_atlas(atlas)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.start_tr, "Start TR")
        _validate_positive(ns.tr, "TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,
            task=ns.task,
            atlas=ns.atlas,
            fwhm=ns.fwhm,
            start_tr=ns.start_tr,
            tr=ns.tr,
        )


def main(args: AllArgs) -> int:
    """Main entrypoint of combined pipeline."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC full pipeline")
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
    if args.task is not None:
        filters.append(pl.col("task") == args.task)
    df = df.filter(pl.all_horizontal(filters))

    for _, sub_ses_group in tqdm(
        df.group_by(_SUB_SES_QUERY, maintain_order=True), disable=not ctx.verbose
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        # --- Anatomical (once per session, first T1w) ---
        for _, anat_df in session.anat.filter(pl.col("suffix") == "T1w").group_by(
            _ANAT_GROUP_ENTITIES, maintain_order=True
        ):
            anat_row = anat_df.filter(suffix="T1w").row(0, named=True)
            t1w_fpath = Path(anat_row["root"]) / anat_row["path"]
            ents = extract_entities(anat_row, ["run", "acq", "rec", "echo"])
            ctx.logger.info(f"Anatomical: {t1w_fpath}")

            anat_outputs = anatomical_preprocess(in_t1w=t1w_fpath)

            anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=ents)
            export_anatomical(anat, anat_outputs)

        # --- Functional + Metrics + QC (per BOLD run) ---
        for func_df, _anat_df in iter_session_files(
            session, groupby=_FUNC_GROUP_ENTITIES
        ):
            row = func_df.row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])
            ctx.logger.info(f"Functional: {bold_fpath}")

            func_metadata = FunctionalMetadata.load(bold_fpath, tr_override=args.tr)

            func_outputs = functional_preprocess(
                in_bold=bold_fpath,
                t1w_brain=anat_outputs.brain,
                wm_bbr_mask=anat_outputs.wm_bbr_mask,
                brain_mask=anat_outputs.brain_mask,
                csf_mask=anat_outputs.csf_mask,
                wm_mask=anat_outputs.wm_mask,
                anat_to_template=anat_outputs.inverse_xfm,
                metadata=func_metadata,
                start_tr=args.start_tr,
                regressor_set=args.regressor,
            )

            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            mni = export_functional(func, func_outputs, regressors=args.regressor)

            # --- Metrics ---
            for regressor in args.regressor:
                ctx.logger.info(
                    f"Metrics: sub-{pipe_ctx.sub} task-{ents.get('task', '')} "
                    f"run-{ents.get('run', 0)} regressor-{regressor}"
                )
                metrics_outputs = metrics_pipeline(
                    regressed_bold=func_outputs.regressed_bold[regressor],
                    cleaned_bold=func_outputs.cleaned_bold[regressor],
                    template_brain_mask=func_outputs.template_brain_mask,
                    tr=func_metadata.tr,
                    atlas=args.atlas,
                    fwhm=args.fwhm,
                )

                export_metrics(
                    mni,
                    metrics_outputs,
                    regressor=regressor,
                    atlases=args.atlas,
                )

            # --- QC ---
            ctx.logger.info(
                f"QC: sub-{pipe_ctx.sub} "
                f"task-{ents.get('task', '')} run-{ents.get('run', 0)}"
            )
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
                task=ents.get("task", ""),
                run=ents.get("run", 0),
                start_tr=args.start_tr,
                regressor_set=args.regressor,
            )

            export_qc(mni, qc_outputs, regressors=args.regressor)

            status = "PASSED" if qc_outputs.passed else "FAILED"
            ctx.logger.info(
                f"QC {status} for sub-{pipe_ctx.sub} task-{ents.get('task', '')} "
                f"run-{ents.get('run', 0)}"
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
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Nuisance regression method.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--atlas",
        nargs="+",
        choices=[
            "schaefer_200",
            "schaefer_300",
            "schaefer_400",
            "schaefer_1000",
            "aal",
        ],
        default=["schaefer_200"],
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
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides BIDS sidecar and NIfTI header.",
    )

    parser.set_defaults(func=lambda args: main(AllArgs.validate_namespace(args)))
