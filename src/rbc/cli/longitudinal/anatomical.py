"""``rbc longitudinal anatomical`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli.base import _or_default, _validate_nifti_path
from rbc.cli.longitudinal._base import LongitudinalBaseArgs, add_fs_license_argument
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.anatomical import run
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class AnatomicalLongArgs(LongitudinalBaseArgs):
    """Arguments for ``rbc longitudinal anatomical``."""

    registration_template: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AnatomicalLongArgs:
        """Validate namespace for the longitudinal anatomical subcommand."""
        return cls(
            **LongitudinalBaseArgs.validate_namespace(ns).__dict__,
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
        )


def main(args: AnatomicalLongArgs) -> int:
    """Run the longitudinal anatomical stage."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
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
    """Register ``rbc longitudinal anatomical`` on a longitudinal subparser group."""
    parser = subparsers.add_parser(
        "anatomical",
        parents=parents,
        description=(
            "Warp preprocessed anatomical derivatives into each subject's "
            "longitudinal template space."
        ),
        help="Longitudinal anatomical stage",
        usage=(
            "rbc longitudinal anatomical INPUT_DIR [INPUT_DIR ...] "
            "-o OUTPUT_DIR [options]"
        ),
    )
    add_fs_license_argument(parser)

    templates = parser.add_argument_group("template overrides")
    templates.add_argument(
        "--anat-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain template for anatomical registration.",
    )

    parser.set_defaults(
        func=lambda args: main(AnatomicalLongArgs.validate_namespace(args))
    )
