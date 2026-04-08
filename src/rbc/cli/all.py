"""CLI subcommand that chains all four workflows.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory without disk round-trips
between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.all import run
from rbc_resources import ATLAS_REGISTRY

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
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        atlases=args.atlas,
        fwhm=args.fwhm,
        start_tr=args.start_tr,
        tr=args.tr,
        runner_config=RunnerConfig(
            runner=args.runner,
            verbose=bool(args.verbose),
            tmp_dir=args.tmp_dir,
        ),
    )
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
        choices=list(ATLAS_REGISTRY.keys()),
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
