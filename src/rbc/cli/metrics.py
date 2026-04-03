"""CLI subcommand for derivative metrics.

Reads functional derivatives from ``output_dir`` and computes ALFF, fALFF,
ReHo, smoothing, z-scoring, and atlas-based timeseries extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.base import BaseArgs, _validate_atlas, _validate_positive, _validate_task
from rbc.core.niwrap import setup_runner
from rbc.orchestration import Filters
from rbc.orchestration.metrics import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

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


def main(args: MetricsArgs) -> int:
    """Main entrypoint of metrics workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS
    ctx.logger.info("Preparing to run RBC metrics workflow")

    run(
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        atlases=args.atlas,
        fwhm=args.fwhm,
        tr=args.tr,
        verbose=ctx.verbose,
    )

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
