"""CLI subcommand for quality control.

Reads preprocessed derivatives from ``output_dir`` and computes QC metrics
(framewise displacement, DVARS, registration overlap, etc.), generating
per-run XCP-D-format TSVs with pass/fail flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.qc import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class QCArgs(BaseArgs):
    """Arguments for the QC CLI."""

    task: str | None
    start_tr: int
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> QCArgs:
        """Validate QC-specific arguments."""
        _validate_task(ns.task)
        _validate_positive(ns.start_tr, "Start TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            task=ns.task,
            start_tr=ns.start_tr,
            regressor=ns.regressor,
        )


def main(args: QCArgs) -> int:
    """Main entrypoint of QC workflow."""
    run(
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        start_tr=args.start_tr,
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
    """Register QC workflow to parser."""
    parser = subparsers.add_parser(
        "qc",
        parents=parents,
        description="RBC quality control workflow",
        help="QC workflow (motion, DVARS, registration overlap)",
        usage="rbc qc INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--start-tr",
        type=int,
        default=2,
        help="Number of initial TRs discarded during preprocessing.",
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

    parser.set_defaults(func=lambda args: main(QCArgs.validate_namespace(args)))
