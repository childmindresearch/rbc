"""Orchestration for the combined longitudinal pipeline.

Runs template, anatomical, functional, metrics, and QC in sequence for
each subject, passing outputs in-memory between stages where possible.

The template step is inherently cross-session and writes to disk.
Subsequent per-session stages (anat, func) resolve their own inputs
from disk but return outputs for in-memory handoff to metrics and QC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

from rbc.bids import FUNC_GROUP_ENTITIES, Datatype, Suffix, extract_entities, load_table
from rbc.bids.longitudinal.template import discover_template_inputs
from rbc.bids.metrics import export_metrics
from rbc.bids.session import _FUNC_ENTITY_KEYS, iter_session_files
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.longitudinal._iter import iter_sessions_with_template
from rbc.orchestration.longitudinal.anatomical import process_anat
from rbc.orchestration.longitudinal.functional import process_func
from rbc.orchestration.longitudinal.metrics import _read_derivative_tr
from rbc.orchestration.longitudinal.qc import process_qc
from rbc.orchestration.longitudinal.template import (
    process_subject,
    setup_freesurfer_auth,
)
from rbc.workflows.metrics import single_session_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Literal

__all__ = ["run"]

_logger = logging.getLogger(__name__)


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[Literal["36-parameter", "aCompCor"]] = ("36-parameter",),
    fs_license: Path | None = None,
    atlas_files: Mapping[str, Path] | None = None,
    smooth: float | None = None,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run the full longitudinal pipeline (template -> anat -> func -> metrics -> qc).

    Unlike running each workflow separately, this passes functional
    outputs in-memory to the metrics and QC stages without disk
    round-trips.

    Args:
        input_dirs: BIDS dataset directories (must include preprocessed
            cross-sectional derivatives).
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor strategies to apply.
        fs_license: Optional FreeSurfer license for template generation.
        atlas_files: Mapping of atlas labels to NIfTI file paths.
            If ``None``, metrics are skipped.
        smooth: Smoothing kernel FWHM in mm, or ``None`` to skip smoothing.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.warning(
        "This workflow is experimental and may be sensitive to input file "
        "naming conventions."
    )
    _logger.info("Preparing to run RBC full longitudinal pipeline")

    # --- Step 1: Template generation (cross-session, writes to disk) ---
    setup_freesurfer_auth(fs_license)

    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )
    # Template discovery needs all sessions (not just the filtered ones),
    # so apply only the participant filter here.
    tpl_filters = Filters(participant_label=filters.participant_label)
    tpl_inputs, skipped = discover_template_inputs(tpl_filters.apply(df))
    for sub in skipped:
        _logger.warning(
            "Skipping sub-%s: only one preprocessed T1w brain volume found.",
            sub,
        )

    for subject_inputs in tqdm(tpl_inputs, desc="Templates", disable=not verbose):
        pipe_ctx = RunContext(sub=subject_inputs.sub, ses=None, output_dir=output_dir)
        process_subject(pipe_ctx, subject_inputs)
        pipe_ctx.ensure_dataset_description()

    _logger.info("Template generation complete; proceeding to per-session stages")

    # --- Step 2: Per-session anat -> func -> metrics -> qc ---
    for pipe_ctx, session, tpl_df in iter_sessions_with_template(
        input_dirs, output_dir, filters=filters, verbose=verbose
    ):
        # Anatomical
        for _, anat_df in session.anat.group_by(("run", "acq"), maintain_order=True):
            anat_outputs = process_anat(
                pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df
            )

        # Functional + Metrics + QC (per BOLD run)
        for func_df, _ in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
            func_outputs = process_func(
                pipe_ctx=pipe_ctx,
                func_df=func_df,
                tpl_df=tpl_df,
                regressors=regressors,
            )

            row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
            ents = extract_entities(row, _FUNC_ENTITY_KEYS)
            func_q = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func_long = func_q.derive(space="longitudinal")

            # Metrics (per regressor, in-memory from func_outputs)
            if atlas_files:
                tr = _read_derivative_tr(func_outputs.bold)
                for regressor in regressors:
                    _logger.info(
                        "Longitudinal metrics: sub-%s ses-%s regressor-%s",
                        pipe_ctx.sub,
                        pipe_ctx.ses,
                        regressor,
                    )
                    metrics_outputs = single_session_metrics(
                        regressed_bold=func_outputs.regressed_bold[regressor],
                        cleaned_bold=func_outputs.cleaned_bold[regressor],
                        template_brain_mask=func_outputs.bold_mask,
                        tr=tr,
                        atlas_files=atlas_files,
                        smooth=smooth,
                    )
                    export_metrics(
                        func_long,
                        metrics_outputs,
                        regressor=regressor,
                        atlases=list(atlas_files),
                        smooth=smooth,
                    )

            # QC (in-memory from func_outputs + anat_outputs)
            if anat_outputs.brain_mask is None:
                _logger.warning(
                    "Skipping longitudinal QC for sub-%s ses-%s: "
                    "no anatomical brain mask in longitudinal space.",
                    pipe_ctx.sub,
                    pipe_ctx.ses,
                )
                continue

            _logger.info("Longitudinal QC: sub-%s ses-%s", pipe_ctx.sub, pipe_ctx.ses)
            process_qc(
                func_long,
                anat_brain_mask=anat_outputs.brain_mask,
                bold_mask=func_outputs.bold_mask,
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses or "",
                task=ents.get("task", ""),
                run=ents.get("run", 0),
            )

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC full longitudinal pipeline complete")
