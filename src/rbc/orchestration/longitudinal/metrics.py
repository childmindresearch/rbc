"""Orchestration for the longitudinal metrics workflow (not yet implemented)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rbc.orchestration import Filters, RunnerConfig

__all__ = ["run"]


def run(
    input_dirs: Sequence[Path],
    output_dir: Path,
    *,
    filters: Filters,
    atlas_files: dict[str, Path],
    fwhm: float,
    runner_config: RunnerConfig | None = None,
) -> None:
    """Compute resting-state metrics in longitudinal space.

    Placeholder wired up by Stage 3; full implementation ships in Stage 6.
    """
    del input_dirs, output_dir, filters, atlas_files, fwhm, runner_config
    raise NotImplementedError(
        "rbc longitudinal metrics is planned for Stage 6 of the longitudinal "
        "refactor (tracker: #301). It is not yet implemented."
    )
