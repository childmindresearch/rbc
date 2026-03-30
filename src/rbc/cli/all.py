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

from rbc.cli import (
    _ANAT_GROUP_ENTITIES,
    _DEFAULT_ENV_VARS,
    _FUNC_GROUP_ENTITIES,
    _SUB_SES_QUERY,
)
from rbc.cli.base import BaseArgs, _validate_atlas, _validate_positive, _validate_task
from rbc.cli.query import iter_session_files, load_session
from rbc.context import PipelineContext
from rbc.core.bids import (
    Datatype,
    Suffix,
    TemplateSpace,
    bids_safe_label,
    extract_entities,
)
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

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    atlas: Sequence[AtlasName]
    fwhm: float
    start_tr: int

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AllArgs:
        """Validate all-workflow arguments."""
        _validate_task(ns.task)
        for atlas in ns.atlas:
            _validate_atlas(atlas)
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


def main(args: AllArgs) -> int:  # noqa: C901
    """Main entrypoint of combined pipeline."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS
    ctx.runner.image_overrides = {"antsx/ants:v2.5.3": "antsx/ants:latest"}

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
        sorted(df.group_by(_SUB_SES_QUERY)), disable=not ctx.verbose
    ):
        pipe_ctx = PipelineContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=args.output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        # --- Anatomical (once per session, first T1w) ---
        anat_outputs = None
        for _, anat_df in iter_session_files(session, groupby=_ANAT_GROUP_ENTITIES):
            anat_row = anat_df.filter(suffix="T1w").row(0, named=True)
            t1w_fpath = Path(anat_row["root"]) / anat_row["path"]
            ents = extract_entities(anat_row, ["run", "acq", "rec", "echo"])
            ctx.logger.info(f"Anatomical: {t1w_fpath}")

            anat_outputs = anatomical_preprocess(in_t1w=t1w_fpath)

            anat = pipe_ctx.bids(datatype=Datatype.ANAT, entities=ents)
            anat.save(anat_outputs.brain, suffix=Suffix.T1W, desc="brain")
            anat.save(anat_outputs.brain_mask, suffix=Suffix.MASK, desc="T1w")
            anat.save(anat_outputs.csf_mask, suffix=Suffix.MASK, desc="csf")
            anat.save(anat_outputs.gm_mask, suffix=Suffix.MASK, desc="gm")
            anat.save(anat_outputs.wm_mask, suffix=Suffix.MASK, desc="wm")
            anat.save(anat_outputs.wm_bbr_mask, suffix=Suffix.MASK, desc="wmBBR")
            anat.save(
                anat_outputs.forward_xfm,
                suffix="xfm",
                extra={
                    "from": "T1w",
                    "to": TemplateSpace.MNI152NLIN6ASYM,
                    "mode": "image",
                },
            )
            anat.save(
                anat_outputs.inverse_xfm,
                suffix="xfm",
                extra={
                    "from": TemplateSpace.MNI152NLIN6ASYM,
                    "to": "T1w",
                    "mode": "image",
                },
            )

        if anat_outputs is None:
            raise ValueError(f"No T1w found for sub-{pipe_ctx.sub} ses-{pipe_ctx.ses}")

        # --- Functional + Metrics + QC (per BOLD run) ---
        for func_df, _anat_df in iter_session_files(
            session, groupby=_FUNC_GROUP_ENTITIES
        ):
            row = func_df.row(0, named=True)
            bold_fpath = Path(row["root"]) / row["path"]
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])
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
            func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func.save(func_outputs.sbref, suffix=Suffix.SBREF)
            func.save(func_outputs.preproc_bold, suffix=Suffix.BOLD, desc="preproc")
            func.save(
                func_outputs.motion_params,
                suffix=Suffix.MOTION,
                desc="motionParams",
                extension=".1D",
            )
            func.save(
                func_outputs.rms_rel,
                suffix=Suffix.MOTION,
                desc="relsDisplacement",
                extension=".rms",
            )
            func.save(
                func_outputs.rms_abs,
                suffix=Suffix.MOTION,
                desc="maxDisplacement",
                extension=".rms",
            )
            func.save(func_outputs.bold_mask, suffix=Suffix.MASK, desc="brain")
            func.save(
                func_outputs.bold_to_anat_matrix,
                suffix="xfm",
                desc="linear",
                extension=".txt",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            func.save(
                func_outputs.bold_to_anat_itk,
                suffix="xfm",
                desc="linearITK",
                extension=".txt",
                extra={"from": "bold", "to": "T1w", "mode": "image"},
            )
            for regressor in args.regressor:
                func.save(
                    func_outputs.regressor_file[regressor],
                    suffix="regressors",
                    desc=bids_safe_label(regressor),
                    extension=".1D",
                )

            mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)
            for regressor in args.regressor:
                mni.save(
                    func_outputs.regressed_bold[regressor],
                    suffix=Suffix.BOLD,
                    desc="regressed",
                    extra={"reg": regressor},
                )
                mni.save(
                    func_outputs.cleaned_bold[regressor],
                    suffix=Suffix.BOLD,
                    desc="preproc",
                    extra={"reg": regressor},
                )

            mni.save(func_outputs.template_bold, suffix=Suffix.BOLD, desc="preproc")
            mni.save(func_outputs.template_brain_mask, suffix=Suffix.MASK, desc="bold")

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
                    atlas=args.atlas,
                    fwhm=args.fwhm,
                )

                mex = mni.derive(extra={"reg": regressor})
                mex.save(metrics_outputs.alff, suffix="alff")
                mex.save(metrics_outputs.falff, suffix="falff")
                mex.save(metrics_outputs.alff_smooth, suffix="alff", desc="smooth")
                mex.save(metrics_outputs.falff_smooth, suffix="falff", desc="smooth")
                mex.save(metrics_outputs.alff_zscored, suffix="alff", desc="smoothZstd")
                mex.save(
                    metrics_outputs.falff_zscored, suffix="falff", desc="smoothZstd"
                )
                mex.save(metrics_outputs.reho, suffix="reho")
                mex.save(metrics_outputs.reho_smooth, suffix="reho", desc="smooth")
                mex.save(metrics_outputs.reho_zscored, suffix="reho", desc="smoothZstd")
                for atlas in args.atlas:
                    mex.save(
                        metrics_outputs.timeseries[atlas],
                        suffix="timeseries",
                        desc="mean",
                        extension=".tsv",
                        atlas=atlas,
                    )
                    mex.save(
                        metrics_outputs.correlation_matrix[atlas],
                        suffix="correlations",
                        desc="pearson",
                        extension=".tsv",
                        atlas=atlas,
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

            for regressor in args.regressor:
                mni.save(
                    qc_outputs.qc_file[regressor],
                    suffix="quality",
                    desc="xcp",
                    extension=".tsv",
                    extra={"reg": regressor},
                )

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

    parser.set_defaults(func=lambda args: main(AllArgs.validate_namespace(args)))
