"""``rbc longitudinal template`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rbc.cli.base import BaseArgs
from rbc.orchestration import Filters, RunnerConfig
from rbc.orchestration.longitudinal.template import run

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


@dataclass(frozen=True)
class TemplateArgs(BaseArgs):
    """Arguments for ``rbc longitudinal template``."""

    fs_license: Path | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> TemplateArgs:
        """Validate base args and the optional --fs-license path."""
        fs_license: Path | None = ns.fs_license
        if fs_license is not None and not fs_license.exists():
            raise ValueError(f"FreeSurfer license not found: {fs_license}")
        return cls(**BaseArgs.validate_namespace(ns).__dict__, fs_license=fs_license)


def main(args: TemplateArgs) -> int:
    """Build a robust longitudinal template per matching subject."""
    run(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        fs_license=args.fs_license,
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
    parser.add_argument(
        "--fs-license",
        type=Path,
        default=None,
        help=(
            "Optional path to a FreeSurfer license file. Falls back to the "
            "FS_LICENSE environment variable, then to a license-free bypass "
            "if neither is set."
        ),
    )
    parser.set_defaults(func=lambda args: main(TemplateArgs.validate_namespace(args)))
