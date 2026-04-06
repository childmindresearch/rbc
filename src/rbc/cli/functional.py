"""CLI subcommand for functional processing.

Parses subject/session/task arguments and delegates to
``rbc.orchestration.functional.run``, which runs the functional
stream (reorientation, TR truncation, motion correction, etc.). Anatomical
preprocessing must be completed first since coregistration and template
warping depend on the anatomical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import BaseArgs, _validate_positive, _validate_task
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.functional import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class FunctionalArgs(BaseArgs):
    """Arguments for single-session functional CLI."""

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    tr: float | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalArgs:
        """Validation of functional workflow specific arguments to NamedTuple."""
        _validate_task(ns.task)
        _validate_positive(ns.tr, "TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,  # Validated by argparse choices
            task=ns.task,
            tr=ns.tr,
        )


def main(args: FunctionalArgs) -> int:
    """Main entrypoint of functional workflow."""
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
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
    """Register functional workflow to parser."""
    parser = subparsers.add_parser(
        "functional",
        parents=parents,
        description="RBC functional workflow",
        help="Functional workflow",
        usage="rbc input_dir output_dir functional [-h] [options]",
    )
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Space-delimited nuisance regression method(s) to apply.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides BIDS sidecar and NIfTI header.",
    )

    parser.set_defaults(func=lambda args: main(FunctionalArgs.validate_namespace(args)))
