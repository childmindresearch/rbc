"""CLI subcommand for derivative metrics.

Reads functional derivatives from ``output_dir`` and computes ALFF, fALFF,
ReHo, smoothing, z-scoring, and atlas-based timeseries extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import (
    BaseArgs,
    _validate_atlas_nifti,
    _validate_positive,
    _validate_task,
)
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.metrics import run
from rbc_resources import ATLAS_REGISTRY, resolve_atlas

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


def _resolve_atlas_args(raw_atlases: list[str]) -> dict[str, Path]:
    """Resolve a list of atlas names or paths to a label-to-path mapping.

    Registry atlases are trusted; custom paths are validated for integer
    dtype and 3-D shape.
    """
    atlas_files: dict[str, Path] = {}
    for entry in raw_atlases:
        label, path = resolve_atlas(entry)
        if label in atlas_files:
            raise ValueError(
                f"Duplicate atlas label {label!r}. Use distinct file names "
                f"for custom atlases."
            )
        if entry not in ATLAS_REGISTRY:
            _validate_atlas_nifti(path)
        atlas_files[label] = path
    return atlas_files


@dataclass(frozen=True)
class MetricsArgs(BaseArgs):
    """Arguments for the metrics CLI."""

    atlas_files: dict[str, Path]
    fwhm: float
    task: str | None
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    tr: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> MetricsArgs:
        """Validate metrics-specific arguments."""
        _validate_task(ns.task)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.tr, "TR")
        atlas_files = _resolve_atlas_args(ns.atlas)
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            atlas_files=atlas_files,
            fwhm=ns.fwhm,
            task=ns.task,
            regressor=ns.regressor,
            tr=ns.tr,
        )


def main(args: MetricsArgs) -> int:
    """Main entrypoint of metrics workflow."""
    run(
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
        ),
    )
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
