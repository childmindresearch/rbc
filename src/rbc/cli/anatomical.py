"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.base import BaseArgs
from rbc.core.niwrap import setup_runner
from rbc.orchestration import Filters
from rbc.orchestration.anatomical import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class AnatomicalArgs(BaseArgs):
    """Arguments for single-session anatomical CLI."""

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AnatomicalArgs:
        """Validation of anatomical workflow specific arguments to NamedTuple."""
        return cls(**BaseArgs.validate_namespace(ns).__dict__)


def main(args: AnatomicalArgs) -> int:
    """Main entrypoint of anatomical workflow."""
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = _DEFAULT_ENV_VARS
    ctx.logger.info("Preparing to run RBC anatomical workflow")

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
        verbose=ctx.verbose,
    )

    ctx.logger.info("RBC anatomical workflow complete")
    return 0


def register_command(
    subparsers: argparse._SubParsersAction, parents: Sequence[argparse.ArgumentParser]
) -> None:
    """Register anatomical workflow to parser."""
    parser = subparsers.add_parser(
        "anatomical",
        parents=parents,
        description="RBC anatomical workflow",
        help="Anatomical workflow",
        usage="rbc input_dir output_dir anatomical [-h] [options]",
    )

    parser.set_defaults(func=lambda args: main(args))
