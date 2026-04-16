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

from rbc.cli.base import (
    BaseArgs,
    _or_default,
    _validate_nifti_path,
    _validate_positive,
    _validate_task,
)
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.functional import run
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class FunctionalArgs(BaseArgs):
    """Arguments for single-session functional CLI."""

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    tr: float | None
    smooth: float | None
    func_template: Path
    func_template_mask: Path
    func_template_ref: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> FunctionalArgs:
        """Validation of functional workflow specific arguments to NamedTuple."""
        _validate_task(ns.task)
        _validate_positive(ns.tr, "TR")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,
            task=ns.task,
            tr=ns.tr,
            smooth=ns.smooth,
            func_template=_or_default(
                ns.func_template, REGISTRATION_TEMPLATES.brain_2mm
            ),
            func_template_mask=_or_default(
                ns.func_template_mask, REGISTRATION_TEMPLATES.brain_mask_2mm
            ),
            func_template_ref=_or_default(
                ns.func_template_ref, REGISTRATION_TEMPLATES.bold_ref
            ),
        )


def main(args: FunctionalArgs) -> int:
    """Main entrypoint of functional workflow."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        tr=args.tr,
        smooth=args.smooth,
        func_template=args.func_template,
        func_template_mask=args.func_template_mask,
        func_template_ref=args.func_template_ref,
        runner_config=RunnerConfig(
            runner=args.runner,
            verbose=bool(args.verbose),
            tmp_dir=args.tmp_dir,
            ants_threads=args.ants_threads,
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
        usage="rbc functional INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]",
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
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        metavar="FWHM",
        help="Smooth the cleaned (post-regression, bandpass-filtered) BOLD with "
        "the kernel of specified FWHM in mm (e.g. --smooth 6.0) "
        "If omitted, no smoothing is applied.",
    )

    templates = parser.add_argument_group("template overrides")
    templates.add_argument(
        "--func-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain template for functional resampling (default: MNI152 2 mm).",
    )
    templates.add_argument(
        "--func-template-mask",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain mask for functional masking (default: MNI152 2 mm).",
    )
    templates.add_argument(
        "--func-template-ref",
        type=_validate_nifti_path,
        default=None,
        help="Custom BOLD reference image for functional masking.",
    )

    parser.set_defaults(func=lambda args: main(FunctionalArgs.validate_namespace(args)))
