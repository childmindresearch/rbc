"""CLI subcommand for longitudinal processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli.base import BaseArgs, _or_default, _validate_nifti_path
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal import run
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class LongitudinalArgs(BaseArgs):
    """Arguments for longitudinal CLI."""

    anatomical: bool
    functional: bool
    registration_template: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> LongitudinalArgs:
        """Validation of longitudinal workflow specific arguments to NamedTuple."""
        if not ns.functional and not ns.anatomical:
            raise ValueError(
                "At least one of '--anatomical' or '--functional' is required."
            )
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            anatomical=ns.anatomical,
            functional=ns.functional,
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
        )


def main(args: LongitudinalArgs) -> int:
    """Main entrypoint of longitudinal workflow."""
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
        anatomical=args.anatomical,
        functional=args.functional,
        registration_template=args.registration_template,
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
    """Register longitudinal workflow to parser."""
    parser = subparsers.add_parser(
        "longitudinal",
        parents=parents,
        description="RBC-based longitudinal workflow",
        help="Longitudinal workflow",
        usage="rbc input_dir output_dir longitudinal [-h] [options]",
    )
    parser.add_argument(
        "--anatomical",
        default=False,
        action="store_true",
        help="Use anatomical longitudinal pipeline for processing",
    )
    parser.add_argument(
        "--functional",
        default=False,
        action="store_true",
        help="Use functional longitudinal pipeline for processing",
    )

    templates = parser.add_argument_group("template overrides")
    templates.add_argument(
        "--anat-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain template for anatomical registration.",
    )

    parser.set_defaults(
        func=lambda args: main(LongitudinalArgs.validate_namespace(args))
    )
