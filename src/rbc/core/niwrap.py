"""Niwrap helpers used throughout the pipeline.

Provides utilities for setting up and tearing down runners. Allows us to use runner
of choice depending on what is available on the system.
"""

import logging
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

import niwrap

_LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: niwrap.Runner


def setup_runner(
    runner: Literal["local", "docker", "singularity"] = "local",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    verbose: int = 0,
    **kwargs,  # noqa: ANN003 (ignore annotation for kwargs)
) -> StyxContext:
    """Setup Styx with appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use - choices include
            ['local', 'docker', 'singularity']
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        Configured logger instance and initialized runner
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()

    match runner_exec := runner.lower():
        case "local":
            niwrap.use_local()
        case "docker":
            niwrap.use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case "singularity":
            niwrap.use_singularity(
                singularity_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Unknown runner selection '{runner}' - please select one of "
                "'local', 'docker', or 'singularity"
            )

    styx_runner = niwrap.get_global_runner()
    styx_runner.data_dir = Path(tmp_dir)
    logger = logging.getLogger(styx_runner.logger_name)
    log_level = min(verbose, len(_LOG_LEVELS) - 1)
    logger.setLevel(_LOG_LEVELS[log_level])
    return StyxContext(logger=logger, runner=styx_runner)


def generate_exec_folder(suffix: str = "python") -> Path:
    """Generate an execution folder following Styx hash pattern.

    Args:
        suffix: Task to append to suffix of folder (default: 'python')

    Returns:
        Path to created execution folder
    """
    runner = niwrap.get_global_runner()
    dir_path = (
        Path(runner.data_dir) / f"{runner.uid}_{runner.execution_counter}_{suffix}"
    )
    dir_path.mkdir(parents=True)
    runner.execution_counter += 1
    return dir_path
