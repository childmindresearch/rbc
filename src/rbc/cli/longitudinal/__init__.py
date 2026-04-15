"""``rbc longitudinal`` parent command and nested subcommand registration.

Stage 3 lands the full nested subcommand layout. Every stage of the
longitudinal pipeline has a dedicated subcommand; ``metrics``, ``qc``, and
``all`` are registered so they surface in ``--help`` but raise
``NotImplementedError`` until Stage 6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.cli.longitudinal import (
    all as all_,
)
from rbc.cli.longitudinal import (
    anatomical,
    functional,
    metrics,
    qc,
    template,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


def register_command(
    subparsers: argparse._SubParsersAction,
    parents: Sequence[argparse.ArgumentParser],
) -> None:
    """Register the ``longitudinal`` parent command and nested subcommands."""
    parser = subparsers.add_parser(
        "longitudinal",
        aliases=["long"],
        description="RBC longitudinal workflows",
        help="Longitudinal workflows",
        usage=("rbc longitudinal {template,anatomical,functional,metrics,qc,all} ..."),
    )
    nested = parser.add_subparsers(
        title="longitudinal stages",
        dest="long_stage",
        required=True,
        description="Available longitudinal stages",
        help="Stage help",
    )
    template.register_command(nested, parents=parents)
    anatomical.register_command(nested, parents=parents)
    functional.register_command(nested, parents=parents)
    metrics.register_command(nested, parents=parents)
    qc.register_command(nested, parents=parents)
    all_.register_command(nested, parents=parents)
