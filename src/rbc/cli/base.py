"""Base arguments shared across all workflow CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

_VALID_RUNNERS = frozenset({"local", "docker", "singularity"})


@dataclass(frozen=True)
class BaseArgs:
    """Base (global) arguments shared across all workflow CLIs."""

    input_dir: Path
    output_dir: Path
    runner: Literal["local", "docker", "singularity"]
    participant_label: list[str]
    session_label: list[str]
    verbose: int

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> BaseArgs:
        """Validation of base arguments."""
        if not ns.input_dir.exists():
            raise ValueError(f"Input path does not exist: {ns.input_dir}")
        if ns.runner not in _VALID_RUNNERS:
            raise ValueError(
                f"Expected one of {_VALID_RUNNERS} for runner, got: {ns.runner!r}"
            )

        for labels, prefix in (
            (ns.participant_label, "sub-"),
            (ns.session_label, "ses-"),
        ):
            if bad := next(
                (label for label in labels if label.startswith(prefix)), None
            ):
                raise ValueError(f"Label must not start with {prefix!r}: {bad!r}")

        return cls(
            input_dir=ns.input_dir,
            output_dir=ns.output_dir,
            runner=ns.runner,
            participant_label=ns.participant_label,
            session_label=ns.session_label,
            verbose=ns.verbose,
        )
