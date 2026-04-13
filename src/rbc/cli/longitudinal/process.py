"""Legacy ``rbc longitudinal process`` subcommand.

Carries the pre-Stage-2 ``--anatomical --functional`` flag flow under a
nested subcommand so the new ``rbc longitudinal template`` can sit beside
it. Stage 3 will replace this with dedicated ``anatomical`` / ``functional``
subcommands.
"""

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
    """Arguments for the legacy ``longitudinal process`` subcommand."""

    anatomical: bool
    functional: bool
    registration_template: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> LongitudinalArgs:
        """Validate the legacy longitudinal namespace."""
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
    """Run the legacy longitudinal anat+func dispatcher."""
    run(
        input_dirs=args.input_dirs,
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
            ants_threads=args.ants_threads,
        ),
    )
    return 0


def register_command(
    subparsers: argparse._SubParsersAction,
    parents: Sequence[argparse.ArgumentParser],
) -> None:
    """Register ``rbc longitudinal process`` on a longitudinal subparser group."""
    parser = subparsers.add_parser(
        "process",
        parents=parents,
        description=(
            "Legacy longitudinal anat/func dispatcher (will be split into "
            "dedicated subcommands in Stage 3)."
        ),
        help="Legacy longitudinal anat/func dispatcher",
        usage=(
            "rbc longitudinal process INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]"
        ),
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
