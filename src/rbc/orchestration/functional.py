"""Orchestration for the cross-sectional functional workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from rbc.bids import SUB_SES_QUERY, Datatype, load_table
from rbc.bids.functional import (
    FunctionalInputs,
    discover_functional,
    export_functional,
    resolve_functional,
)
from rbc.bids.session import load_session
from rbc.context import RunContext
from rbc.core.nifti import log_image_summary
from rbc.metadata import FunctionalMetadata
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.functional import single_session_preprocess
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids import Bids
    from rbc.bids.session import SessionTables
    from rbc.workflows.functional import FunctionalOutputs

_logger = logging.getLogger(__name__)


def process_session(
    session: SessionTables,
    pipe_ctx: RunContext,
    *,
    regressors: Sequence[str],
    anat_inputs: FunctionalInputs | None = None,
    tr: float | None = None,
    smooth: float | None = None,
    func_template: Path = REGISTRATION_TEMPLATES.brain_2mm,
    func_template_mask: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
    func_template_ref: Path = REGISTRATION_TEMPLATES.bold_ref,
) -> list[tuple[FunctionalOutputs, Bids, FunctionalMetadata]]:
    """Run functional preprocessing for one session.

    Args:
        session: Session tables for a single subject/session.
        pipe_ctx: RunContext bound to this subject/session.
        regressors: Regressor names.
        anat_inputs: Pre-resolved anatomical inputs. When provided (e.g. from
            the combined ``all`` pipeline), skips the DataFrame-based resolve
            and uses these paths directly for every BOLD run.
        tr: TR override in seconds, or ``None`` to read from headers.
        smooth: Smoothing kernel FWHM in mm, or ``None`` to skip smoothing.
        func_template: Brain template for functional resampling (default: MNI152 2 mm).
        func_template_mask: Brain mask for functional masking (default: MNI152 2 mm).
        func_template_ref: BOLD reference image for functional masking.

    Returns:
        List of (outputs, mni_builder, metadata) per BOLD run,
        for use by downstream workflows (metrics, QC).
    """
    results = []
    for func_run in discover_functional(session):
        log_image_summary(func_run.path, label="Functional BOLD")

        if anat_inputs is not None:
            resolved = anat_inputs
        else:
            anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
            resolved = resolve_functional(anat_q, func_run.anat_df)

        func_metadata = FunctionalMetadata.load(func_run.path, tr_override=tr)

        outputs = single_session_preprocess(
            in_bold=func_run.path,
            t1w_brain=resolved["t1w_brain"],
            wm_bbr_mask=resolved["wm_bbr_mask"],
            brain_mask=resolved["brain_mask"],
            csf_mask=resolved["csf_mask"],
            wm_mask=resolved["wm_mask"],
            anat_to_template=resolved["anat_to_template"],
            metadata=func_metadata,
            smooth=smooth,
            regressor_set=regressors,  # type: ignore[arg-type]
            func_template=func_template,
            func_template_mask=func_template_mask,
            func_template_ref=func_template_ref,
        )

        func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=func_run.entities)
        mni = export_functional(func, outputs, regressors=regressors, smooth=smooth)
        results.append((outputs, mni, func_metadata))

    return results


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    tr: float | None = None,
    smooth: float | None = None,
    func_template: Path = REGISTRATION_TEMPLATES.brain_2mm,
    func_template_mask: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
    func_template_ref: Path = REGISTRATION_TEMPLATES.bold_ref,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the functional pipeline for all matching subjects/sessions.

    Args:
        input_dirs: BIDS dataset directories.
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor names.
        tr: TR override in seconds.
        smooth: Smoothing kernel FWHM in mm, or ``None`` to skip smoothing.
        func_template: Brain template for functional resampling (default: MNI152 2 mm).
        func_template_mask: Brain mask for functional masking (default: MNI152 2 mm).
        func_template_ref: BOLD reference image for functional masking.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC functional workflow")
    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )

    df = filters.apply(
        df,
        pl.col("datatype").is_not_null(),
        pl.col("ses").ne_missing("longitudinal"),
        pl.col("space").is_null(),
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
        process_session(
            session,
            pipe_ctx,
            regressors=regressors,
            tr=tr,
            smooth=smooth,
            func_template=func_template,
            func_template_mask=func_template_mask,
            func_template_ref=func_template_ref,
        )
        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC functional workflow complete")
