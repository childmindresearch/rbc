"""Pipeline orchestration layer.

Provides ``run()`` entry points for each workflow that handle runner setup,
BIDS table loading, filtering, sub/ses iteration, and the
discover-process-export loop. CLI modules delegate to these after parsing
arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from rbc.core import CPAC_ANTS_SEED
from rbc.core.niwrap import setup_runner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal

_DEFAULT_ENV_VARS = {
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
}


@dataclass(frozen=True)
class Filters:
    """User-level filters applied to the BIDS table before processing.

    Attributes:
        participant_label: Subject labels to include (empty = all).
        session_label: Session labels to include (empty = all).
        task: Task label to filter BOLD runs (None = all).
    """

    participant_label: Sequence[str] = field(default_factory=tuple)
    session_label: Sequence[str] = field(default_factory=tuple)
    task: str | None = None

    def apply(self, df: pl.DataFrame, *base_exprs: pl.Expr) -> pl.DataFrame:
        """Apply user-level and workflow-specific filters to a BIDS table.

        Args:
            df: BIDS table to filter.
            *base_exprs: Workflow-specific filter expressions
                (e.g. space, datatype constraints).

        Returns:
            Filtered DataFrame.
        """
        exprs = list(base_exprs)
        if len(self.participant_label) > 0:
            exprs.append(pl.col("sub").is_in(self.participant_label))
        if len(self.session_label) > 0:
            exprs.append(pl.col("ses").is_in(self.session_label))
        if self.task is not None:
            exprs.append(pl.col("task") == self.task)
        if not exprs:
            return df
        return df.filter(pl.all_horizontal(exprs))


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for the execution backend.

    Attributes:
        runner: Execution backend (local, docker, podman, singularity).
        verbose: Enable verbose output (progress bars, info logging).
        tmp_dir: Temporary directory for intermediate files.
        ants_threads: Number of threads for ANTs (ITK) operations.
    """

    runner: Literal["auto", "local", "docker", "podman", "singularity"] = "local"
    verbose: bool = False
    tmp_dir: Path | None = None
    ants_threads: int = 1


def init_runner(config: RunnerConfig) -> None:
    """Set up the execution backend and environment variables.

    Args:
        config: Runner configuration.
    """
    ctx = setup_runner(
        runner=config.runner, verbose=config.verbose, tmp_dir=config.tmp_dir
    )
    ctx.runner.environ = {
        **_DEFAULT_ENV_VARS,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(config.ants_threads),
    }
