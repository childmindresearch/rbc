"""``rbc longitudinal all`` subcommand (placeholder for Stage 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.cli.base import _or_default, _validate_nifti_path, _validate_positive
from rbc.cli.longitudinal._base import LongitudinalBaseArgs, add_fs_license_argument
from rbc.cli.metrics import _resolve_atlas_args
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.all import run
from rbc_resources import ATLAS_REGISTRY, REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class AllLongArgs(LongitudinalBaseArgs):
    """Arguments for ``rbc longitudinal all``."""

    registration_template: Path
    atlas_files: dict[str, Path]
    fwhm: float

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> AllLongArgs:
        """Validate namespace for the full longitudinal pipeline subcommand."""
        _validate_positive(ns.fwhm, "FWHM")
        return cls(
            **LongitudinalBaseArgs.validate_namespace(ns).__dict__,
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
            atlas_files=_resolve_atlas_args(ns.atlas),
            fwhm=ns.fwhm,
        )


def main(args: AllLongArgs) -> int:
    """Run the combined longitudinal pipeline."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
        fs_license=args.fs_license,
        atlas_files=args.atlas_files,
        fwhm=args.fwhm,
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
    """Register ``rbc longitudinal all`` (Stage 6 placeholder)."""
    parser = subparsers.add_parser(
        "all",
        parents=parents,
        description=(
            "Run the full longitudinal pipeline (template → anat → func → "
            "metrics → qc). Placeholder wired up by Stage 3; full "
            "implementation ships in Stage 6."
        ),
        help="Full longitudinal pipeline (Stage 6)",
        usage=(
            "rbc longitudinal all INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]"
        ),
    )
    add_fs_license_argument(parser)
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

    templates = parser.add_argument_group("template overrides")
    templates.add_argument(
        "--anat-template",
        type=_validate_nifti_path,
        default=None,
        help="Custom brain template for anatomical registration.",
    )

    parser.set_defaults(func=lambda args: main(AllLongArgs.validate_namespace(args)))
