"""Shared fixtures for tests data."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

import niwrap
import pytest
from styxcache import CachePolicy, CachingRunner
from styxcache.backends import docker_digest_resolver, podman_digest_resolver
from styxpodman import PodmanRunner

from rbc.core.niwrap import resolve_runner
from rbc.orchestration import _DEFAULT_ENV_VARS


class _AttrProxyCachingRunner(CachingRunner):
    # CachingRunner doesn't proxy base-runner attrs, but rbc.core.niwrap
    # reads/mutates data_dir, uid, execution_counter on the global runner.
    _OWN_ATTRS = frozenset({"base", "store", "policy"})

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["base"], name)

    def __setattr__(self, name: str, value: object) -> None:
        if "base" not in self.__dict__ or name in self._OWN_ATTRS:
            super().__setattr__(name, value)
        else:
            setattr(self.__dict__["base"], name, value)


class TestSubjectData(NamedTuple):
    """Test subject file paths."""

    subject_id: str
    subject_dir: Path
    t1w: Path
    bold: Path
    tasks: Path
    events: Path


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply appropriate markers based on test location."""
    markers = {"unit", "integration", "full_pipeline"}

    for item in items:
        test_path = Path(item.fspath)
        for marker in markers & set(test_path.parts):
            item.add_marker(getattr(pytest.mark, marker))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add option(s) to pytest parser."""
    parser.addoption(
        "--runner",
        action="store",
        default="docker",
        help="Styx runner type to use: "
        "['auto', 'local', 'docker', 'podman', 'singularity']",
    )


@pytest.fixture(scope="session", autouse=True)
def niwrap_runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> niwrap.Runner:
    """Globally set test niwrap runner."""
    # Set up niwrap runner
    runner_type, runner_exec = resolve_runner(
        request.config.getoption("--runner").lower()
    )
    match runner_type:
        case "docker":
            niwrap.use_docker(docker_executable=runner_exec)
        case "podman":
            niwrap.set_global_runner(
                # UserID = 0 currently necessary for some containers used
                runner=PodmanRunner(podman_executable=runner_exec, podman_user_id=0)
            )
        case "singularity":
            niwrap.use_singularity(singularity_executable=runner_exec)
        case _:
            niwrap.use_local()
    runner = niwrap.get_global_runner()
    # Override single-threaded ANTs for testing — deterministic results
    # aren't needed here, and multi-threading cuts registration time ~3-5x.
    runner.environ = {
        **_DEFAULT_ENV_VARS,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(min(os.cpu_count() or 1, 4)),
    }
    runner.data_dir = tmp_path_factory.mktemp(f"{os.urandom(8).hex()}")
    # Set up logging for debugging
    logger = logging.getLogger(runner.logger_name)
    logger.setLevel(logging.DEBUG)

    cache_dir = os.environ.get("RBC_STYXCACHE_DIR")
    if cache_dir and runner_type in {"docker", "podman"}:
        resolver = (
            docker_digest_resolver
            if runner_type == "docker"
            else podman_digest_resolver
        )
        wrapped = _AttrProxyCachingRunner(
            base=runner,
            cache_dir=cache_dir,
            policy=CachePolicy(
                image_digest=resolver,
                # Bump to invalidate when styxcache storage semantics change
                # (e.g. 0.1.x entries lacked persisted stdout).
                extra={"cache_generation": "2026-2"},
            ),
        )
        niwrap.set_global_runner(wrapped)
        return wrapped
    return runner


@pytest.fixture(scope="session")
def test_dataset_dir() -> Path:
    """Return path to test dataset directory."""
    return Path(__file__).parent / "data" / "ds000001"


@pytest.fixture(scope="session")
def test_subject(test_dataset_dir: Path) -> TestSubjectData:
    """Return namespace containing file paths to test subject data."""
    subject_id = "01"
    task_id = "balloonanalogrisktask"

    subject_dir = test_dataset_dir / f"sub-{subject_id}"
    anat_dir = subject_dir / "anat"
    func_dir = subject_dir / "func"

    subject_data = TestSubjectData(
        subject_id=subject_id,
        subject_dir=subject_dir,
        t1w=anat_dir / f"sub-{subject_id}_T1w.nii.gz",
        bold=func_dir / f"sub-{subject_id}_task-{task_id}_run-01_bold.nii.gz",
        tasks=test_dataset_dir / f"task-{task_id}_bold.json",
        events=func_dir / f"sub-{subject_id}_task-{task_id}_run-01_events.tsv",
    )

    required_files = {
        "T1w": subject_data.t1w,
        "BOLD": subject_data.bold,
        "task": subject_data.tasks,
        "events": subject_data.events,
    }
    for name, fpath in required_files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"{name} file not found: {fpath}")
    return subject_data
