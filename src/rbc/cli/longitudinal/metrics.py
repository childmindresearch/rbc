"""``rbc longitudinal metrics`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import _validate_positive, _validate_task
from rbc.cli.longitudinal._base import LongitudinalBaseArgs, add_fs_license_argument
from rbc.cli.metrics import _resolve_atlas_args
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.metrics import run
from rbc_resources import ATLAS_REGISTRY

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class MetricsLongArgs(LongitudinalBaseArgs):
    """Arguments for ``rbc longitudinal metrics``."""

    atlas_files: dict[str, Path]
    fwhm: float
    tr: float | None
    task: str | None
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> MetricsLongArgs:
        """Validate namespace for the longitudinal metrics subcommand."""
        _validate_task(ns.task)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.tr, "TR")
        return cls(
            **LongitudinalBaseArgs.validate_namespace(ns).__dict__,
            atlas_files=_resolve_atlas_args(ns.atlas),
            fwhm=ns.fwhm,
            tr=ns.tr,
            task=ns.task,
            regressor=ns.regressor,
        )


def main(args: MetricsLongArgs) -> int:
    """Run resting-state metrics in longitudinal space."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        atlas_files=args.atlas_files,
        fwhm=args.fwhm,
        tr=args.tr,
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
    """Register ``rbc longitudinal metrics``."""
    parser = subparsers.add_parser(
        "metrics",
        parents=parents,
        description=(
            "Compute resting-state metrics (ALFF, fALFF, ReHo, atlas "
            "timeseries) on longitudinal-space functional derivatives."
        ),
        help="Compute resting-state metrics in longitudinal space",
        usage=(
            "rbc longitudinal metrics INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]"
        ),
    )
    add_fs_license_argument(parser)
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help=("Space-delimited nuisance regression method(s) to compute metrics for."),
    )
    parser.add_argument(
        "--atlas",
        nargs="+",
        default=["schaefer_200"],
        metavar="ATLAS",
        help=(
            "Atlas(es) for timeseries extraction. Accepts registry names "
            f"({', '.join(sorted(ATLAS_REGISTRY))}) or paths to custom NIfTI "
            "atlas files."
        ),
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=6.0,
        help="Smoothing kernel FWHM in mm.",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides NIfTI header.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )

    parser.set_defaults(
        func=lambda args: main(MetricsLongArgs.validate_namespace(args))
    )
