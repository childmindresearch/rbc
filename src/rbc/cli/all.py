"""CLI subcommand that chains all four workflows.

Runs anatomical, functional, metrics, and QC in sequence for each
subject-session, passing outputs in-memory without disk round-trips
between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import (
    BaseArgs,
    _build_brain_extraction_templates,
    _or_default,
    _validate_nifti_path,
    _validate_positive,
    _validate_task,
)
from rbc.cli.metrics import _resolve_atlas_args
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.all import run
from rbc_resources import ATLAS_REGISTRY, REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

    from rbc_resources import BrainExtractionTemplates


@dataclass(frozen=True)
class AllArgs(BaseArgs):
    """Arguments for the combined pipeline CLI."""

    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    task: str | None
    atlas_files: dict[str, Path]
    fwhm: float
    start_tr: int
    tr: float | None
    brain_extraction_templates: BrainExtractionTemplates
    registration_template: Path
    func_template: Path
    func_template_mask: Path
    func_template_ref: Path

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AllArgs:
        """Validate all-workflow arguments."""
        _validate_task(ns.task)
        _validate_positive(ns.fwhm, "FWHM")
        _validate_positive(ns.start_tr, "Start TR")
        _validate_positive(ns.tr, "TR")
        atlas_files = _resolve_atlas_args(ns.atlas)
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            regressor=ns.regressor,
            task=ns.task,
            atlas_files=atlas_files,
            fwhm=ns.fwhm,
            start_tr=ns.start_tr,
            tr=ns.tr,
            brain_extraction_templates=_build_brain_extraction_templates(ns),
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
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


def main(args: AllArgs) -> int:
    """Main entrypoint of combined pipeline."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
            task=args.task,
        ),
        regressors=args.regressor,
        atlas_files=args.atlas_files,
        fwhm=args.fwhm,
        start_tr=args.start_tr,
        tr=args.tr,
        brain_extraction_templates=args.brain_extraction_templates,
        registration_template=args.registration_template,
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
    """Register combined pipeline to parser."""
    parser = subparsers.add_parser(
        "all",
        parents=parents,
        description="RBC full pipeline (anatomical + functional + metrics + QC)",
        help="Full pipeline (all workflows)",
        usage="rbc all INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]",
    )
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Nuisance regression method.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task label to filter BOLD runs (without 'task-' prefix).",
    )
    parser.add_argument(
        "--atlas",
        nargs="+",
        default=["schaefer_200"],
        metavar="ATLAS",
        help=(
            "Atlas(es) for timeseries extraction. Accepts registry names "
            f"({', '.join(sorted(ATLAS_REGISTRY))}) or paths to custom NIfTI "
            "atlas files."
        ),
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=6.0,
        help="Smoothing kernel FWHM in mm.",
    )
    parser.add_argument(
        "--start-tr",
        type=int,
        default=2,
        help="Number of initial TRs to discard.",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds. Overrides BIDS sidecar and NIfTI header.",
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

    parser.set_defaults(func=lambda args: main(AllArgs.validate_namespace(args)))
