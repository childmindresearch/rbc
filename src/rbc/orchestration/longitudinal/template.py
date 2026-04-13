"""Per-subject orchestration for longitudinal template construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import niwrap
from tqdm import tqdm

from rbc.bids import Datatype, load_table
from rbc.bids.longitudinal.template import (
    discover_template_inputs,
    export_template,
)
from rbc.context import RunContext
from rbc.orchestration import Filters, RunnerConfig, init_runner
from rbc.workflows.longitudinal.template import (
    LongitudinalTemplateOutputs,
    generate_subject_template,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.bids.longitudinal.template import TemplateInputs

__all__ = ["process_subject", "run"]

_logger = logging.getLogger(__name__)


def process_subject(
    pipe_ctx: RunContext, inputs: TemplateInputs
) -> LongitudinalTemplateOutputs:
    """Build and export a robust longitudinal template for one subject.

    Args:
        pipe_ctx: RunContext bound to this subject (``ses=None``).
        inputs: Per-session preprocessed T1w volumes.

    Returns:
        Workflow outputs (template + ITK transforms).
    """
    outputs = generate_subject_template(
        sub=inputs.sub,
        sessions=inputs.sessions,
        in_files=inputs.files,
    )
    tpl = pipe_ctx.bids(datatype=Datatype.ANAT).derive(ses="longitudinal")
    export_template(tpl, outputs)
    return outputs


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Build a longitudinal template per subject across all matching sessions.

    Args:
        input_dirs: BIDS dataset directories (must include preprocessed
            cross-sectional anatomical derivatives).
        output_dir: Output directory for derivatives.
        filters: Participant/session filters applied before grouping.
        runner_config: Execution backend configuration.
    """
    config = runner_config or RunnerConfig()
    init_runner(config)
    # Bypass FreeSurfer's chklc() so no FS license is needed.
    niwrap.get_global_runner().environ["SURFER_SIDEDOOR"] = "1"
    verbose = config.verbose

    _logger.warning(
        "This workflow is experimental and may be sensitive to input file "
        "naming conventions."
    )
    _logger.info("Preparing to build longitudinal templates")
    df = load_table(
        dataset_dirs=input_dirs, index_fpath=None, max_workers=0, verbose=verbose
    )
    df = filters.apply(df)

    inputs = discover_template_inputs(df)
    if not inputs:
        raise ValueError(
            "No subject has multiple sessions of preprocessed T1w brain "
            "volumes; cannot build a longitudinal template."
        )

    for subject_inputs in tqdm(inputs, disable=not verbose):
        pipe_ctx = RunContext(sub=subject_inputs.sub, ses=None, output_dir=output_dir)
        process_subject(pipe_ctx, subject_inputs)
        pipe_ctx.ensure_dataset_description()

    _logger.info("Longitudinal template construction complete")
