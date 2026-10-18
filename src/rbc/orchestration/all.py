"""Orchestration for the combined cross-sectional pipeline.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory between stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, load_table
from rbc.bids.metrics import export_metrics
from rbc.bids.qc import export_qc
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.anatomical import process_session as process_anat
from rbc.orchestration.functional import process_session as process_func
from rbc.workflows.metrics import single_session_metrics
from rbc.workflows.qc import single_session_qc
from rbc_resources import (
    BRAIN_EXTRACTION_TEMPLATES,
    REGISTRATION_TEMPLATES,
    BrainExtractionTemplates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_logger = logging.getLogger(__name__)


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    atlas_files: Mapping[str, Path],
    smooth: float | None = None,
    start_tr: int,
    tr: float | None = None,
    brain_extraction_templates: BrainExtractionTemplates = BRAIN_EXTRACTION_TEMPLATES,
    registration_template: Path = REGISTRATION_TEMPLATES.brain_1mm,
    func_template: Path = REGISTRATION_TEMPLATES.brain_2mm,
    func_template_mask: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
    func_template_ref: Path = REGISTRATION_TEMPLATES.bold_ref,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the full pipeline (anat + func + metrics + QC) per session.

    Unlike running each workflow separately, this passes outputs in-memory
    between stages without disk round-trips.

    Args:
        input_dirs: BIDS dataset directories.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        atlas_files: Mapping of atlas labels to resolved NIfTI file paths.
        smooth: Smoothing kernel FWHM in mm.
        start_tr: Number of initial TRs discarded during preprocessing.
        tr: TR override in seconds, or ``None`` to read from headers.
        brain_extraction_templates: Brain extraction template bundle.
        registration_template: Brain template for ANTs registration.
        func_template: Brain template for functional resampling (default: MNI152 2 mm).
        func_template_mask: Brain mask for functional masking (default: MNI152 2 mm).
        func_template_ref: BOLD reference image for functional masking.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC full pipeline")
    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )

    df = filters.apply(
        df,
        pl.col("datatype").is_not_null(),
        pl.col("ses").ne_missing("longitudinal"),
        pl.col("space").is_null(),
        pl.col("desc").is_null(),
    )

    for _, sub_ses_group in tqdm(
        df.group_by(SUB_SES_QUERY, maintain_order=True), disable=not verbose
    ):
        pipe_ctx = RunContext(
            sub=sub_ses_group["sub"][0],
            ses=sub_ses_group["ses"][0] or None,
            output_dir=output_dir,
        )
        session = load_session(sub_ses_group, pipe_ctx.sub, pipe_ctx.ses)

        # --- Anatomical ---
        anat_outputs = process_anat(
            session,
            pipe_ctx,
            brain_extraction_templates=brain_extraction_templates,
            registration_template=registration_template,
        )

        # --- Functional + Metrics + QC (per BOLD run) ---
        func_results = process_func(
            session,
            pipe_ctx,
            regressors=regressors,
            anat_inputs={
                "t1w_brain": anat_outputs.brain,
                "wm_bbr_mask": anat_outputs.wm_bbr_mask,
                "brain_mask": anat_outputs.brain_mask,
                "csf_mask": anat_outputs.csf_mask,
                "wm_mask": anat_outputs.wm_mask,
                "anat_to_template": anat_outputs.anat_to_template_xfm,
            },
            tr=tr,
            func_template=func_template,
            func_template_mask=func_template_mask,
            func_template_ref=func_template_ref,
        )

        for func_outputs, mni, func_metadata in func_results:
            # Metrics (per regressor)
            for regressor in regressors:
                _logger.info(
                    "Metrics: sub-%s regressor-%s",
                    pipe_ctx.sub,
                    regressor,
                )
                metrics_outputs = single_session_metrics(
                    regressed_bold=func_outputs.regressed_bold[regressor],
                    cleaned_bold=func_outputs.cleaned_bold[regressor],
                    template_brain_mask=func_outputs.template_brain_mask,
                    tr=func_metadata.tr,
                    atlas_files=atlas_files,
                    smooth=smooth,
                )
                export_metrics(
                    mni,
                    metrics_outputs,
                    regressor=regressor,
                    atlases=list(atlas_files),
                    smooth=smooth,
                )

            # QC
            _logger.info("QC: sub-%s", pipe_ctx.sub)
            qc_outputs = single_session_qc(
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
                task="",
                run=0,
                start_tr=start_tr,
                regressor_set=regressors,
                mni_brain_mask_2mm=func_template_mask,
            )
            export_qc(mni, qc_outputs, regressors=regressors)

            status = "PASSED" if qc_outputs.passed else "FAILED"
            _logger.info("QC %s for sub-%s", status, pipe_ctx.sub)

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC full pipeline complete")
