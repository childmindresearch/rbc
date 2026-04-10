"""CLI subcommand for longitudinal processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc.cli.base import BaseArgs, _or_default, _validate_nifti_path
from rbc.cli.metrics import _resolve_atlas_args
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal import run
from rbc_resources import ATLAS_REGISTRY, REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class LongitudinalArgs(BaseArgs):
    """Arguments for longitudinal CLI."""

    anatomical: bool
    functional: bool
    metrics: bool
    registration_template: Path
    regressor: Sequence[Literal["36-parameter", "aCompCor"]]
    atlas_files: dict[str, Path]

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
            metrics=ns.metrics,
            registration_template=_or_default(
                ns.anat_template, REGISTRATION_TEMPLATES.brain_1mm
            ),
            regressor=ns.regressor,
            atlas_files=_resolve_atlas_args(ns.atlas),
        )


def main(args: LongitudinalArgs) -> int:
    """Main entrypoint of longitudinal workflow."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        filters=Filters(
            participant_label=args.participant_label,
            session_label=args.session_label,
        ),
        anatomical=args.anatomical,
        functional=args.functional,
        metrics=args.metrics,
        registration_template=args.registration_template,
        regressors=args.regressor,
        atlas_files=args.atlas_files,
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
    """Register longitudinal workflow to parser."""
    parser = subparsers.add_parser(
        "longitudinal",
        parents=parents,
        description="RBC-based longitudinal workflow",
        help="Longitudinal workflow",
        usage="rbc longitudinal INPUT_DIR [INPUT_DIR ...] -o OUTPUT_DIR [options]",
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
    parser.add_argument(
        "--regressor",
        nargs="+",
        choices=["36-parameter", "aCompCor"],
        default=["36-parameter"],
        help="Space-delimited nuisance regression method(s) to apply.",
    )
    parser.add_argument(
        "--metrics",
        default=False,
        action="store_true",
        help="Compute longitudinal metrics (ALFF, ReHo, timeseries).",
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
