"""``rbc longitudinal qc`` subcommand (placeholder for Stage 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.cli.longitudinal._base import LongitudinalBaseArgs, add_fs_license_argument
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.qc import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


QCLongArgs = LongitudinalBaseArgs


def main(args: QCLongArgs) -> int:
    """Run registration QC for longitudinal derivatives."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
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
    """Register ``rbc longitudinal qc`` (Stage 6 placeholder)."""
    parser = subparsers.add_parser(
        "qc",
        parents=parents,
        description=(
            "Run registration QC for longitudinal derivatives. Placeholder "
            "wired up by Stage 3; full implementation ships in Stage 6."
        ),
        help="Longitudinal QC stage (Stage 6)",
        usage=("rbc longitudinal qc INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]"),
    )
    add_fs_license_argument(parser)
    parser.set_defaults(
        func=lambda args: main(LongitudinalBaseArgs.validate_namespace(args))
    )
