"""Shared fixtures for tests data."""

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply appropriate markers based on test location."""
    for item in items:
        test_path = Path(item.fspath)

        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def test_dataset_dir() -> Path:
    """Return path to test dataset directory."""
    return Path(__file__).parent / "data" / "ds000001"


@pytest.fixture
def test_subject(test_dataset_dir: Path) -> SimpleNamespace:
    """Return namespace containing file paths to test subject data."""
    subject_id = "01"
    task_id = "balloonanalogrisktask"

    subject_dir = test_dataset_dir / f"sub-{subject_id}"
    anat_dir = subject_dir / "anat"
    func_dir = subject_dir / "func"

    subject_data = SimpleNamespace(
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
