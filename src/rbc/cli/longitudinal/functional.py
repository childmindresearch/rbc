"""``rbc longitudinal functional`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import _validate_positive, _validate_task
from rbc.cli.longitudinal._base import LongitudinalBaseArgs, add_fs_license_argument
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.functional import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class FunctionalLongArgs(LongitudinalBaseArgs):
    """Arguments for ``rbc longitudinal functional``."""

    task: str | None
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    smooth: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalLongArgs:
        """Validate namespace for the longitudinal functional subcommand."""
        _validate_task(ns.task)
        if ns.smooth is not None:
            _validate_positive(ns.smooth, "smooth")
        return cls(
            **LongitudinalBaseArgs.validate_namespace(ns).__dict__,
            task=ns.task,
            regressor=ns.regressor,
            smooth=ns.smooth,
        )


def main(args: FunctionalLongArgs) -> int:
    """Run the longitudinal functional stage."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        smooth=args.smooth,
        runner_config=RunnerConfig(
            runner=args.runner,
            verbose=bool(args.verbose),
            tmp_dir=args.tmp_dir,
            ants_threads=args.ants_threads,
        ),
    )
    return 0


def register_command(
    subparsers: argparse._SubParsersAction,
    parents: Sequence[argparse.ArgumentParser],
) -> None:
    """Register ``rbc longitudinal functional`` on a longitudinal subparser group."""
    parser = subparsers.add_parser(
        "functional",
        parents=parents,
        description=(
            "Warp preprocessed BOLD derivatives into each subject's "
            "longitudinal template space and re-run nuisance regression."
        ),
        help="Longitudinal functional stage",
        usage=(
            "rbc longitudinal functional INPUT_DIR [INPUT_DIR ...] "
            "-o OUTPUT_DIR [options]"
        ),
    )
    add_fs_license_argument(parser)
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        metavar="FWHM",
        help="Smoothing with the kernel of specified FWHM in mm (e.g. --smooth 6.0). "
        "If omitted, no smoothing is applied.",
    )
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Space-delimited nuisance regression method(s) to apply.",
    )

    parser.set_defaults(
        func=lambda args: main(FunctionalLongArgs.validate_namespace(args))
    )
