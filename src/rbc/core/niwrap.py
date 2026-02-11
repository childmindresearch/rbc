"""Niwrap helpers used throughout the pipeline.

Provides utilities for setting up and tearing down runners. Allows us to use runner
of choice depending on what is available on the system.
"""

import logging
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

import niwrap


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: niwrap.Runner


def setup_runner(
    runner: Literal["local", "docker", "singularity"] = "local",
    tmp_dir: Path | None = None,
    image_overrides: dict[str, str] | None = None,
    **kwargs,  # noqa: ANN003 (ignore annotation for kwargs)
) -> StyxContext:
    """Setup Styx with appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use - choices include
            ['local', 'docker', 'singularity']
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        Configured logger instance and initialized runner
    """
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())

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
    styx_runner.data_dir = tmp_dir
    logger = logging.getLogger(styx_runner.logger_name)
    logger.setLevel(logging.INFO)
    return StyxContext(logger=logger, runner=styx_runner)
