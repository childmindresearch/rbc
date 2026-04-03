"""CLI subcommand for derivative metrics.

Reads functional derivatives from ``output_dir`` and computes ALFF, fALFF,
ReHo, smoothing, z-scoring, and atlas-based timeseries extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import polars as pl
from tqdm import tqdm

from rbc.cli import _DEFAULT_ENV_VARS, _FUNC_GROUP_ENTITIES, _SUB_SES_QUERY
from rbc.cli.base import BaseArgs, _validate_atlas, _validate_positive, _validate_task
from rbc.context import RunContext
from rbc.core.bids import (
    Datatype,
    Suffix,
    TemplateSpace,
    bids_safe_label,
    extract_entities,
)
from rbc.core.bids2table import load_table
from rbc.core.niwrap import setup_runner
from rbc.workflows.metrics import single_session_metrics

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

    from rbc_resources import AtlasName


@dataclass(frozen=True)
class MetricsArgs(BaseArgs):
    """Arguments for the metrics CLI."""

    atlas: Sequence[AtlasName]
    fwhm: float
    task: str | None
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    tr: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> MetricsArgs:
        """Validate metrics-specific arguments."""
        for atlas in ns.atlas:
            _validate_atlas(atlas)
        _validate_task(ns.task)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.tr, "TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            atlas=ns.atlas,
            fwhm=ns.fwhm,
            task=ns.task,
            regressor=ns.regressor,
            tr=ns.tr,
        )


_logger = logging.getLogger(__name__)


def _read_header_tr(nifti_path: Path) -> float:
    """Read TR from a NIfTI header, raising on missing/zero values."""
    hdr = nib.nifti1.load(nifti_path).header
    tr = float(hdr["pixdim"][4])  # type: ignore[index]
    if tr <= 0:
        msg = (
            f"NIfTI header TR is {tr} for {nifti_path}. Pass --tr to specify manually."
        )
        raise ValueError(msg)
    _logger.info("TR: %.4f s (from NIfTI header)", tr)
    return tr


def main(args: MetricsArgs) -> int:
    """Main entrypoint of metrics workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS

    ctx.logger.info("Preparing to run RBC metrics workflow")
    if args.tr is not None:
        ctx.logger.info("Using CLI-provided TR: %.4f s", args.tr)
    else:
        ctx.logger.info("TR will be read from NIfTI headers")
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
        pipe_ctx = RunContext(sub=sub, ses=ses, output_dir=args.output_dir)

        deriv_df = load_table(
            dataset_dir=args.output_dir, index_fpath=None, max_workers=0, verbose=False
        )

        for _, run_group in group.group_by(_FUNC_GROUP_ENTITIES):
            row = run_group.row(0, named=True)
            ents = extract_entities(row, ["task", "run", "acq", "rec", "dir", "echo"])

            mni_q = pipe_ctx.bids(
                datatype=Datatype.FUNC,
                entities=ents,
                space=TemplateSpace.MNI152NLIN6ASYM,
            )

            template_brain_mask = mni_q.expect(
                deriv_df, suffix=Suffix.MASK, desc="bold"
            )
            for regressor in args.regressor:
                regressed_bold = mni_q.expect(
                    deriv_df,
                    suffix=Suffix.BOLD,
                    desc="regressed",
                    extra={"reg": bids_safe_label(regressor)},
                )
                cleaned_bold = mni_q.expect(
                    deriv_df,
                    suffix=Suffix.BOLD,
                    desc="preproc",
                    extra={"reg": bids_safe_label(regressor)},
                )

                tr = args.tr if args.tr is not None else _read_header_tr(regressed_bold)

                outputs = single_session_metrics(
                    regressed_bold=regressed_bold,
                    cleaned_bold=cleaned_bold,
                    template_brain_mask=template_brain_mask,
                    tr=tr,
                    atlas=args.atlas,
                    fwhm=args.fwhm,
                )

                mex = mni_q.derive(extra={"reg": bids_safe_label(regressor)})
                mex.save(outputs.alff, suffix="alff")
                mex.save(outputs.falff, suffix="falff")
                mex.save(outputs.alff_smooth, suffix="alff", desc="smooth")
                mex.save(outputs.falff_smooth, suffix="falff", desc="smooth")
                mex.save(outputs.alff_zscored, suffix="alff", desc="smoothZstd")
                mex.save(outputs.falff_zscored, suffix="falff", desc="smoothZstd")
                mex.save(outputs.reho, suffix="reho")
                mex.save(outputs.reho_smooth, suffix="reho", desc="smooth")
                mex.save(outputs.reho_zscored, suffix="reho", desc="smoothZstd")
                for atlas in args.atlas:
                    mex.save(
                        outputs.timeseries[atlas],
                        suffix="timeseries",
                        desc="mean",
                        extension=".tsv",
                        atlas=bids_safe_label(atlas),
                    )
                    mex.save(
                        outputs.correlation_matrix[atlas],
                        suffix="correlations",
                        desc="pearson",
                        extension=".tsv",
                        atlas=bids_safe_label(atlas),
                    )
        pipe_ctx.ensure_dataset_description()

    ctx.logger.info("RBC metrics workflow complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register metrics workflow to parser."""
    parser = subparsers.add_parser(
        "metrics",
        parents=parents,
        description="RBC metrics workflow",
        help="Metrics workflow (ALFF, ReHo, timeseries)",
        usage="rbc input_dir output_dir metrics [-h] [options]",
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
        help="Space-delimited atlas(es) for timeseries extraction.",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=6.0,
        help="Smoothing kernel FWHM in mm.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help=(
            "Space-delimited nuisance regression method(s) used in "
            "functional preprocessing."
        ),
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides NIfTI header value for ALFF.",
    )

    parser.set_defaults(func=lambda args: main(MetricsArgs.validate_namespace(args)))
