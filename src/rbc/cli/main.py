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
from pathlib import Path

from rbc.cli import all as all_
from rbc.cli import anatomical, functional, longitudinal, metrics, qc
from rbc.cli.base import BaseArgs

__all__ = ["BaseArgs"]


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
        "--participant-label",
        nargs="+",
        default=[],
        type=lambda x: x.removeprefix("sub-"),
        help="Space-delimited participant identifier ('sub-' prefix can be removed)",
    )
    global_opts.add_argument(
        "--session-label",
        nargs="+",
        default=[],
        type=lambda x: x.removeprefix("ses-"),
        help="Space-delimited session identifier ('ses-' prefix can be removed)",
    )
    global_opts.add_argument(
        "--runner",
        choices=["local", "docker", "podman", "singularity"],
        default="local",
        type=lambda x: x.lower(),
        help="NiWrap runner to use for executing workflow",
    )
    global_opts.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Directory for intermediate files (default: system temp)",
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
    functional.register_command(subparsers, parents=[global_opts])
    metrics.register_command(subparsers, parents=[global_opts])
    qc.register_command(subparsers, parents=[global_opts])
    all_.register_command(subparsers, parents=[global_opts])
    # Experimental subcommand
    longitudinal.register_command(subparsers, parents=[global_opts])

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
