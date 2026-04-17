"""Orchestration for the longitudinal metrics workflow.

Computes resting-state metrics (ALFF, fALFF, ReHo, atlas timeseries)
on longitudinal-space functional derivatives produced by the functional
stage.  Reuses :func:`~rbc.workflows.metrics.single_session_metrics`
unchanged; only the input resolution targets ``space=longitudinal``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import polars as pl

from rbc.bids import FUNC_GROUP_ENTITIES, Datatype, Suffix, extract_entities
from rbc.bids.longitudinal.metrics import resolve_longitudinal_metrics
from rbc.bids.metrics import export_metrics
from rbc.bids.session import iter_session_files
from rbc.metadata import resolve_tr, warn_implausible_tr
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.orchestration.longitudinal._iter import iter_sessions_with_template
from rbc.workflows.metrics import single_session_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from rbc.bids import Bids
    from rbc.workflows.longitudinal.functional import FunctionalLongOutputs
    from rbc.workflows.metrics import MetricsOutputs

__all__ = ["process_metrics", "run"]

_logger = logging.getLogger(__name__)


def _read_derivative_tr(nifti_path: Path, *, override: float | None = None) -> float:
    """Resolve TR from a derivative NIfTI, with optional CLI override.

    Pipes through :func:`~rbc.metadata.resolve_tr` for validation and
    plausibility warnings, matching the cross-sectional metadata path.
    Derivatives have no BIDS sidecar, so ``sidecar_tr`` is always None.
    """
    hdr = nib.nifti1.load(nifti_path).header
    raw_pixdim = float(hdr["pixdim"][4])  # type: ignore[index]
    header_tr = raw_pixdim if raw_pixdim > 0 else None
    tr = resolve_tr(sidecar_tr=None, header_tr=header_tr, override=override)
    warn_implausible_tr(tr)
    return tr


def process_metrics(
    func_long: Bids,
    *,
    func_outputs: FunctionalLongOutputs,
    tr: float,
    regressor: str,
    atlas_files: Mapping[str, Path],
    fwhm: float,
) -> MetricsOutputs:
    """Run metrics for a single regressor on a single longitudinal BOLD run.

    Used by ``orchestration.longitudinal.all`` to process in-memory
    functional outputs.

    Args:
        func_long: Bids builder with ``space="longitudinal"`` for this run.
        func_outputs: Functional outputs (in-memory from ``all`` pipeline).
        tr: Repetition time in seconds.
        regressor: Regressor name.
        atlas_files: Mapping of atlas labels to resolved NIfTI file paths.
        fwhm: Smoothing kernel FWHM in mm.

    Returns:
        Metrics outputs for this run/regressor.
    """
    outputs = single_session_metrics(
        regressed_bold=func_outputs.regressed_bold[regressor],
        cleaned_bold=func_outputs.cleaned_bold[regressor],
        template_brain_mask=func_outputs.bold_mask,
        tr=tr,
        atlas_files=atlas_files,
        fwhm=fwhm,
    )
    export_metrics(func_long, outputs, regressor=regressor, atlases=list(atlas_files))
    return outputs


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    regressors: Sequence[str],
    atlas_files: Mapping[str, Path],
    fwhm: float,
    tr: float | None = None,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Compute resting-state metrics in longitudinal space.

    Resolves per-regressor ``regressed_bold``/``cleaned_bold`` and the
    brain mask from longitudinal-space functional derivatives, reads TR
    from the NIfTI header (with optional CLI override), and calls
    :func:`~rbc.workflows.metrics.single_session_metrics` unchanged.

    Args:
        input_dirs: BIDS dataset directories (must include longitudinal
            functional derivatives).
        output_dir: Output directory for derivatives.
        filters: Participant/session/task filters.
        regressors: Regressor strategy names to compute metrics for.
        atlas_files: Mapping of atlas labels to resolved NIfTI file paths.
        fwhm: Smoothing kernel FWHM in mm.
        tr: TR override in seconds, or ``None`` to read from headers.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    verbose = config.verbose

    _logger.info("Preparing to run RBC longitudinal metrics workflow")

    for pipe_ctx, session, tpl_df in iter_sessions_with_template(
        input_dirs, output_dir, filters=filters, verbose=verbose
    ):
        for func_df, _ in iter_session_files(session, groupby=FUNC_GROUP_ENTITIES):
            row = func_df.filter(suffix=Suffix.BOLD).row(0, named=True)
            ents = extract_entities(row, ["task", "run"])

            func_q = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
            func_long_q = func_q.derive(space="longitudinal")

            # Build a DataFrame of longitudinal-space functional derivatives
            full_df = pl.concat([func_df, tpl_df])
            func_long_df = full_df.filter(pl.col("space") == "longitudinal")

            for regressor in regressors:
                resolved = resolve_longitudinal_metrics(
                    func_long_q, func_long_df, regressor=regressor
                )
                run_tr = (
                    tr
                    if tr is not None
                    else _read_derivative_tr(resolved["regressed_bold"])
                )
                outputs = single_session_metrics(
                    regressed_bold=resolved["regressed_bold"],
                    cleaned_bold=resolved["cleaned_bold"],
                    template_brain_mask=resolved["template_brain_mask"],
                    tr=run_tr,
                    atlas_files=atlas_files,
                    fwhm=fwhm,
                )
                export_metrics(
                    func_long_q,
                    outputs,
                    regressor=regressor,
                    atlases=list(atlas_files),
                )

        pipe_ctx.ensure_dataset_description()

    _logger.info("RBC longitudinal metrics workflow complete")
