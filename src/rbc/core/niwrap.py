"""Niwrap helpers used throughout the pipeline.

Provides utilities for setting up and tearing down runners. Allows us to use runner
of choice depending on what is available on the system.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

import niwrap
from styxcache import CachePolicy, CachingRunner
from styxcache.backends import docker_digest_resolver, podman_digest_resolver
from styxpodman import PodmanRunner

_LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]

RunnerType = Literal["local", "docker", "podman", "singularity"]

_RUNNER_EXECUTABLES: list[tuple[RunnerType, list[str]]] = [
    ("docker", ["docker"]),
    ("podman", ["podman"]),
    ("singularity", ["apptainer", "singularity"]),
]


# styxcache's CachingRunner doesn't proxy base-runner attributes, but rbc
# touches data_dir / uid / execution_counter on the global runner. This shim
# forwards anything it doesn't own to self.base.
class _CacheProxyingRunner(CachingRunner):
    _OWN_ATTRS = frozenset({"base", "store", "policy"})

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["base"], name)

    def __setattr__(self, name: str, value: object) -> None:
        if "base" not in self.__dict__ or name in self._OWN_ATTRS:
            super().__setattr__(name, value)
        else:
            setattr(self.__dict__["base"], name, value)


def maybe_wrap_with_cache(
    runner: niwrap.Runner, runner_type: RunnerType
) -> niwrap.Runner:
    """Wrap *runner* with a styxcache CachingRunner if RBC_STYXCACHE_DIR is set.

    Registers the wrapped runner as the niwrap global and returns it. When the
    env var is unset, or the runner type isn't a container runner with a
    digest resolver, the runner is returned unchanged.
    """
    cache_dir = os.environ.get("RBC_STYXCACHE_DIR")
    if not cache_dir or runner_type not in {"docker", "podman"}:
        return runner
    resolver = (
        docker_digest_resolver if runner_type == "docker" else podman_digest_resolver
    )
    wrapped = _CacheProxyingRunner(
        base=runner,
        cache_dir=cache_dir,
        policy=CachePolicy(
            image_digest=resolver,
            # Bump to invalidate when styxcache storage semantics change.
            extra={"cache_generation": "2026-1"},
        ),
    )
    niwrap.set_global_runner(wrapped)
    return wrapped


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: niwrap.Runner
    verbose: bool


def resolve_runner(
    runner: RunnerType | Literal["auto"] = "auto",
) -> tuple[RunnerType, str]:
    """Resolve runner selection, auto-detecting if needed.

    When runner is "auto", checks for available container runtimes on PATH
    in order of preference: docker > podman > apptainer/singularity > local.

    Args:
        runner: Runner type or "auto" for auto-detection.

    Returns:
        Tuple of (runner_type, executable_name).
    """
    if runner != "auto":
        return runner, runner

    for runner_type, executables in _RUNNER_EXECUTABLES:
        for exe in executables:
            if shutil.which(exe):
                return runner_type, exe
    return "local", "local"


def setup_runner(
    runner: RunnerType | Literal["auto"] = "auto",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    verbose: int = 0,
    **kwargs,  # noqa: ANN003 (ignore annotation for kwargs)
) -> StyxContext:
    """Setup Styx with appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use. "auto" detects the first available
            container runtime, falling back to "local".
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        Configured logger instance and initialized runner
    """
    runner_type, runner_exec = resolve_runner(runner)

    match runner_type:
        case "local":
            niwrap.use_local()
        case "docker":
            niwrap.use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                docker_user_id=0,
                **kwargs,
            )
        case "podman":
            niwrap.set_global_runner(
                runner=PodmanRunner(
                    podman_executable=runner_exec,
                    image_overrides=image_overrides,
                    podman_user_id=0,
                    **kwargs,
                )
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
                "'auto', 'local', 'docker', 'podman', or 'singularity'"
            )

    styx_runner = niwrap.get_global_runner()
    if tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        data_parent = Path(tmp_dir)
    else:
        data_parent = _work_root() / "niwrap"
        data_parent.mkdir(parents=True, exist_ok=True)
    styx_runner.data_dir = Path(tempfile.mkdtemp(dir=data_parent))
    styx_logger = logging.getLogger(styx_runner.logger_name)
    log_level = min(verbose, len(_LOG_LEVELS) - 1)
    styx_logger.setLevel(_LOG_LEVELS[log_level])

    # Opt-in persistent caching: subprocess invocations of `rbc` in CI (e.g.
    # tests/integration/test_all.py's `rbc all` spawn) pick this up through
    # the RBC_STYXCACHE_DIR env var inherited from the parent pytest process.
    styx_runner = maybe_wrap_with_cache(styx_runner, runner_type)

    rbc_logger = logging.getLogger("rbc")
    rbc_logger.setLevel(_LOG_LEVELS[log_level])
    if not rbc_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        rbc_logger.addHandler(handler)

    return StyxContext(logger=rbc_logger, runner=styx_runner, verbose=verbose > 0)


_CONTAINER_FS_LICENSE_PATH = "/opt/freesurfer/license.txt"


def mount_fs_license(runner: niwrap.Runner, fs_license: Path) -> None:
    """Make a FreeSurfer license available to the active runner.

    Local runner: sets ``FS_LICENSE`` in the ambient environment.
    Container runners: bind-mounts the license at a stable path and points
    ``FS_LICENSE`` at it on the runner environ.
    """
    license_path = fs_license.resolve()
    runner_name = type(runner).__name__.lower().replace("runner", "")

    if runner_name == "local":
        os.environ["FS_LICENSE"] = str(license_path)
        return

    if runner_name in ("docker", "podman"):
        mount_args = [
            "--mount",
            f"type=bind,source={license_path},"
            f"target={_CONTAINER_FS_LICENSE_PATH},readonly",
        ]
    elif runner_name == "singularity":
        mount_args = ["--bind", f"{license_path}:{_CONTAINER_FS_LICENSE_PATH}"]
    else:
        raise ValueError(f"Unsupported runner for FS license mount: {runner_name!r}")

    getattr(runner, f"{runner_name}_extra_args").extend(mount_args)
    runner.environ["FS_LICENSE"] = _CONTAINER_FS_LICENSE_PATH


_WORK_ROOT: Path | None = None


def _work_root() -> Path:
    """Return the shared rbc work-dir root, creating it on first use.

    Honors ``RBC_WORK_DIR`` if set; otherwise creates a process-unique
    temp dir. Both niwrap runners' ``data_dir`` and rbc-owned scratch
    folders live under this root, so a single env var controls disk
    layout and cleanup covers both in one sweep.
    """
    global _WORK_ROOT
    if _WORK_ROOT is not None and _WORK_ROOT.exists():
        return _WORK_ROOT
    override = os.environ.get("RBC_WORK_DIR")
    if override:
        root = Path(override)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="rbc_work_"))
    _WORK_ROOT = root
    return root


def generate_exec_folder(suffix: str = "python") -> Path:
    """Create a fresh scratch directory for intermediate rbc-owned outputs.

    Returns a unique directory under ``RBC_WORK_DIR`` (or a process-unique
    temp dir). Never touches a niwrap runner's data_dir, so writes here
    are guaranteed not to scribble into a cached tool output shard.
    """
    scratch = _work_root() / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{suffix}_", dir=scratch))
