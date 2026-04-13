"""``rbc longitudinal template`` subcommand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.cli.base import BaseArgs
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.template import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


def main(args: BaseArgs) -> int:
    """Build a robust longitudinal template per matching subject."""
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
    """Register ``rbc longitudinal template`` on a longitudinal subparser group."""
    parser = subparsers.add_parser(
        "template",
        parents=parents,
        description=(
            "Build a robust within-subject T1w template for longitudinal "
            "analysis using FreeSurfer's mri_robust_template."
        ),
        help="Build longitudinal T1w template",
        usage=(
            "rbc longitudinal template INPUT_DIR [INPUT_DIR ...] "
            "-o OUTPUT_DIR [options]"
        ),
    )

    parser.set_defaults(func=lambda args: main(BaseArgs.validate_namespace(args)))
