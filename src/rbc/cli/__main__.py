"""Command-line interface for RBC processing pipelines.

This module provides the main entry point for the RBC command-line tool.
It handles argument parsing, workflow dispatch, and execution of BIDS-organized
neuroimaging data processing pipelines.

The CLI supports multiple workflows (e.g., anatomical, functional) with shared global
options. Each workflow can define its own specific parameters while inheriting common
options.

Usage:
    rbc input_dir output_dir {workflow} [options]

Example:
    rbc /data/bids /data/output anatomical --participant-label 01
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rbc.cli import anatomical

_VALID_RUNNERS = frozenset({"local", "docker", "singularity"})


@dataclass(frozen=True)
class BaseArgs:
    """Base (global) arguments shared across all workflow CLIs."""

    input_dir: Path
    output_dir: Path
    runner: Literal["local", "docker", "singularity"]
    participant_labels: list[str]
    session_labels: list[str]
    verbose: int

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> "BaseArgs":
        """Validation of base arguments."""
        if not ns.input_dir.exists():
            raise ValueError(f"Input path does not exist: {ns.input_dir}")
        if ns.runner not in _VALID_RUNNERS:
            raise ValueError(
                f"Expected one of {_VALID_RUNNERS} for runner, got: {ns.runner!r}"
            )

        for labels, prefix in (
            (ns.participant_labels, "sub-"),
            (ns.session_labels, "ses-"),
        ):
            if bad := next(
                (label for label in labels if label.startswith(prefix)), None
            ):
                raise ValueError(f"Label must not start with {prefix!r}: {bad!r}")

        return cls(
            input_dir=ns.input_dir,
            output_dir=ns.output_dir,
            runner=ns.runner,
            participant_labels=ns.participant_labels,
            session_labels=ns.session_labels,
            verbose=ns.verbose,
        )


def _global_opts() -> argparse.ArgumentParser:
    """Shared global options across workflows."""
    global_opts = argparse.ArgumentParser(add_help=False)
    global_opts.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v, -vv, -vvv)",
    )
    global_opts.add_argument(
        "--participant-labels",
        nargs="+",
        default=[],
        type=lambda x: x.removeprefix("sub-"),
        help="Space-delimited participant identifier ('sub-' prefix can be removed)",
    )
    global_opts.add_argument(
        "--session-labels",
        nargs="+",
        default=[],
        type=lambda x: x.removeprefix("ses-"),
        help="Space-delimited session identifier ('ses-' prefix can be removed)",
    )
    global_opts.add_argument(
        "--runner",
        choices=["local", "docker", "singularity"],
        default="local",
        type=lambda x: x.lower(),
        help="NiWrap runner to use for executing workflow",
    )
    return global_opts


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="rbc",
        description="RBC processing pipelines (developed using NiWrap)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s input_dir output_dir {workflow} [options]",
    )
    # Global arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="BIDS-organized input dataset directory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where output data should be stored",
    )
    global_opts = _global_opts()

    # Subcommands
    subparsers = parser.add_subparsers(
        title="workflows",
        dest="workflow",
        required=True,
        description="Available workflows",
        help="Workflow help",
    )
    anatomical.register_command(subparsers, parents=[global_opts])

    for action in global_opts._actions:
        parser._add_action(action)

    return parser


def cli(argv: Sequence[str] | None = None) -> str | int:
    """Main CLI entry point."""
    parser = create_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code is not None else 1

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


def main() -> None:
    """Main entry point for console."""
    sys.exit(cli())


if __name__ == "__main__":
    main()
