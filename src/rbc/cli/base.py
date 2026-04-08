"""Base arguments shared across all workflow CLIs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

_VALID_RUNNERS = frozenset({"auto", "local", "docker", "podman", "singularity"})


@dataclass(frozen=True)
class BaseArgs:
    """Base (global) arguments shared across all workflow CLIs."""

    input_dir: Path
    output_dir: Path
    runner: Literal["auto", "local", "docker", "podman", "singularity"]
    participant_label: list[str]
    session_label: list[str]
    verbose: int
    tmp_dir: Path | None

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

        tmp_dir: Path | None = ns.tmp_dir
        if tmp_dir is not None and tmp_dir.exists() and not tmp_dir.is_dir():
            raise ValueError(
                f"Temporary path exists, but is not a directory: {tmp_dir}"
            )

        return cls(
            input_dir=ns.input_dir,
            output_dir=ns.output_dir,
            runner=ns.runner,
            participant_label=ns.participant_label,
            session_label=ns.session_label,
            verbose=ns.verbose,
            tmp_dir=tmp_dir,
        )


def _validate_task(task: str | None) -> None:
    """Validate BIDS task label contains only alphanumeric characters and '+'."""
    if task is not None and not re.fullmatch(r"[0-9a-zA-Z+]+", task):
        raise ValueError(
            f"Task must contain only alphanumeric characters and '+', got: {task!r}."
        )


def _validate_positive(value: float | int | None, name: str) -> None:
    """Validate that a numeric value is strictly positive."""
    if value is not None and value <= 0:
        raise ValueError(f"{name} should be greater than 0, got: {value!r}.")
