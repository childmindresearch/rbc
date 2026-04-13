"""``rbc longitudinal`` parent command and nested subcommand registration.

Stage 2 introduces the nested subcommand layout. The pre-existing
``--anatomical --functional`` flow lives under ``rbc longitudinal process``
until Stage 3 splits it into per-stage subcommands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.cli.longitudinal import process, template

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
        usage="rbc longitudinal {template,process} ...",
    )
    nested = parser.add_subparsers(
        title="longitudinal stages",
        dest="long_stage",
        required=True,
        description="Available longitudinal stages",
        help="Stage help",
    )
    template.register_command(nested, parents=parents)
    process.register_command(nested, parents=parents)
