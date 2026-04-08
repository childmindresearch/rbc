"""CLI subcommand for anatomical processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli.base import (
    BaseArgs,
    _build_brain_extraction_templates,
    _or_default,
    _validate_nifti_path,
)
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.anatomical import run
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

    from rbc_resources import BrainExtractionTemplates


@dataclass(frozen=True)
class AnatomicalArgs(BaseArgs):
    """Arguments for single-session anatomical CLI."""

    brain_extraction_templates: BrainExtractionTemplates
    registration_template: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AnatomicalArgs:
        """Validation of anatomical workflow specific arguments to NamedTuple."""
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            brain_extraction_templates=_build_brain_extraction_templates(ns),
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
        )


def main(args: AnatomicalArgs) -> int:
    """Main entrypoint of anatomical workflow."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
        brain_extraction_templates=args.brain_extraction_templates,
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
    """Register anatomical workflow to parser."""
    parser = subparsers.add_parser(
        "anatomical",
        parents=parents,
        description="RBC anatomical workflow",
        help="Anatomical workflow",
        usage="rbc anatomical INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]",
    )

    templates = parser.add_argument_group("template overrides")
    templates.add_argument(
        "--anat-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain template for anatomical registration.",
    )
    templates.add_argument(
        "--brain-extraction-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain extraction template (replaces OASIS template).",
    )
    templates.add_argument(
        "--brain-extraction-prob-mask",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain extraction probability mask.",
    )
    templates.add_argument(
        "--brain-extraction-reg-mask",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain extraction registration mask.",
    )

    parser.set_defaults(func=lambda args: main(AnatomicalArgs.validate_namespace(args)))
