"""Shared base arguments for ``rbc longitudinal`` subcommands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rbc.cli.base import BaseArgs

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class LongitudinalBaseArgs(BaseArgs):
    """Base args for longitudinal subcommands.

    Adds ``--fs-license`` on top of :class:`~rbc.cli.base.BaseArgs`.
    Only the template stage currently consumes the license, but the flag is
    accepted across all longitudinal subcommands for a consistent surface.
    """

    fs_license: Path | None

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> LongitudinalBaseArgs:
        """Validate base args plus the optional ``--fs-license`` path."""
        fs_license: Path | None = ns.fs_license
        if fs_license is not None and not fs_license.exists():
            raise ValueError(f"FreeSurfer license not found: {fs_license}")
        return cls(**BaseArgs.validate_namespace(ns).__dict__, fs_license=fs_license)


def add_fs_license_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the ``--fs-license`` argument to a subcommand parser."""
    parser.add_argument(
        "--fs-license",
        type=Path,
        default=None,
        help=(
            "Optional path to a FreeSurfer license file. Falls back to the "
            "FS_LICENSE environment variable, then to a license-free bypass "
            "if neither is set. Only the ``template`` stage consumes it."
        ),
    )
