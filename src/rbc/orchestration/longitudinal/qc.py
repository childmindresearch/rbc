"""Orchestration for the longitudinal QC workflow.

Minimal scope: Dice/Jaccard overlap between the anatomical brain mask
and BOLD brain mask in longitudinal template space.  No visualizations;
see #303/#304 for future viz pipelines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import polars as pl

from rbc.bids import FUNC_GROUP_ENTITIES, Datatype, Suffix, extract_entities, load_table
from rbc.bids.longitudinal.qc import (
    LongitudinalQCOutputs,
    export_longitudinal_qc,
    resolve_longitudinal_qc,
    write_longitudinal_qc_tsv,
)
from rbc.bids.session import iter_session_files
from rbc.core.longitudinal.resampling import resample_img_to_bold_grid
from rbc.core.niwrap import generate_exec_folder
from rbc.core.qc.registration import registration_qc_metrics
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.longitudinal._iter import iter_sessions_with_template

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids import Bids

__all__ = ["process_qc", "run"]

_logger = logging.getLogger(__name__)

# Dice threshold for pass/fail on longitudinal registration QC.
DICE_THRESHOLD = 0.85


def process_qc(
    func_long: Bids,
    *,
    anat_brain_mask: Path,
    bold_mask: Path,
    sub: str,
    ses: str,
    task: str,
    run: int | str,
) -> LongitudinalQCOutputs:
    """Compute longitudinal registration QC for a single BOLD run.

    Compares the anatomical brain mask with the BOLD brain mask, both in
    longitudinal template space, and writes a single-row QC TSV.

    Args:
        func_long: Bids builder with ``space="longitudinal"`` for this run.
        anat_brain_mask: Anatomical brain mask in longitudinal space.
        bold_mask: BOLD brain mask in longitudinal space.
        sub: Subject ID.
        ses: Session label.
        task: Task label.
        run: Run number.

    Returns:
        QC outputs with overlap metrics and pass/fail flag.
    """
    # Resample longitudinal anatomical mask to bold grid for QC purposes.
    # Longitudinal processed data are registered to the longitudinal template with
    # respective modality's native resolution
    anat_brain_mask = resample_img_to_bold_grid(bold_mask, anat_brain_mask)
    anat_mask_arr = nib.nifti1.load(anat_brain_mask).get_fdata()
    bold_mask_arr = nib.nifti1.load(bold_mask).get_fdata()
    reg_metrics = registration_qc_metrics(anat_mask_arr, bold_mask_arr)

    passed = reg_metrics.dice >= DICE_THRESHOLD

    work_dir = generate_exec_folder("longitudinal_qc")
    qc_file = write_longitudinal_qc_tsv(
        work_dir / "registration_quality.tsv",
        sub=sub,
        ses=ses,
        task=task,
        run=run,
        dice=reg_metrics.dice,
        jaccard=reg_metrics.jaccard,
        coverage=reg_metrics.coverage,
        cross_corr=reg_metrics.cross_corr,
        passed=passed,
    )

    outputs = LongitudinalQCOutputs(
        dice=reg_metrics.dice,
        jaccard=reg_metrics.jaccard,
        coverage=reg_metrics.coverage,
        cross_corr=reg_metrics.cross_corr,
        passed=passed,
        qc_file=qc_file,
    )
    export_longitudinal_qc(func_long, outputs)

    status = "PASSED" if passed else "FAILED"
    _logger.info(
        "Longitudinal QC %s (Dice=%.4f) for sub-%s ses-%s task-%s run-%s",
        status,
        reg_metrics.dice,
        sub,
        ses,
        task,
        run,
    )
    return outputs


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Run registration QC for longitudinal derivatives.

    For each BOLD run, computes Dice/Jaccard overlap between the
    anatomical brain mask and BOLD brain mask in longitudinal space.

    Args:
        input_dirs: BIDS dataset directories (must include longitudinal
            anatomical and functional derivatives).
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC longitudinal QC workflow")
    full_df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )

    for pipe_ctx, session, _tpl_df in iter_sessions_with_template(
        input_dirs, output_dir, filters=filters, verbose=verbose
    ):
        anat_q = pipe_ctx.bids(datatype=Datatype.ANAT)
        anat_long_q = anat_q.derive(space="longitudinal")
        anat_long_df = full_df.filter(
            pl.col("datatype") == "anat",
            pl.col("space") == "longitudinal",
        )

        func_long_df = full_df.filter(
            pl.col("datatype") == "func",
            pl.col("space") == "longitudinal",
        )

        for func_df, _ in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
            row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
            ents = extract_entities(row, ["task", "run"])

            func_q = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func_long_q = func_q.derive(space="longitudinal")

            resolved = resolve_longitudinal_qc(
                anat_long_q, func_long_q, anat_long_df, func_long_df
            )

            process_qc(
                func_long_q,
                anat_brain_mask=resolved["anat_brain_mask"],
                bold_mask=resolved["bold_mask"],
                sub=pipe_ctx.sub,
                ses=pipe_ctx.ses or "",
                task=ents.get("task", ""),
                run=ents.get("run", 0),
            )

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC longitudinal QC workflow complete")
