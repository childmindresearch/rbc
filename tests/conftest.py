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
from styxpodman import PodmanRunner

from rbc.cli import _DEFAULT_ENV_VARS


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
        help="Styx runner type to use: ['local', 'docker', 'singularity']",
    )


@pytest.fixture(scope="session", autouse=True)
def niwrap_runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> niwrap.Runner:
    """Globally set test niwrap runner."""
    # Set up niwrap runner
    match request.config.getoption("--runner").lower():
        case "docker":
            niwrap.use_docker()
        case "singularity":
            niwrap.use_singularity()
        case "podman":
            niwrap.set_global_runner(  # using docker executable to fix mounting
                runner=PodmanRunner(podman_user_id=0)
            )
        case _:
            niwrap.use_local()
    runner = niwrap.get_global_runner()
    runner.environ = _DEFAULT_ENV_VARS
    runner.data_dir = tmp_path_factory.mktemp(f"{os.urandom(8).hex()}")
    # Set up logging for debugging
    logger = logging.getLogger(runner.logger_name)
    logger.setLevel(logging.DEBUG)
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
