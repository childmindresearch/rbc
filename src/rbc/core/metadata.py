"""BIDS metadata extraction utilities.

Sidecar JSON files can exist at multiple levels (dataset, subject, session, run).
More specific files override more general ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from rbc.core.bids import parse_bids_name

__all__ = ["get_repetition_time"]


def get_repetition_time(bold_file: Path) -> float:
    """Get RepetitionTime (TR) from BIDS sidecar JSON.

    Args:
        bold_file: Path to BOLD file

    Returns:
        Repetition time in seconds
    """
    # Check run level json
    json_file = bold_file.parent / bold_file.name.replace(".nii.gz", ".json").replace(
        ".nii", ".json"
    )
    if json_file.exists():
        with Path.open(json_file) as f:
            return json.load(f)["RepetitionTime"]

    # Check dataset level json
    parsed = parse_bids_name(bold_file.name)
    task = parsed.entities.get("task")

    if task:
        current = bold_file.parent
        for _ in range(5):
            task_json = current / f"task-{task}_bold.json"
            if task_json.exists():
                with Path.open(task_json) as f:
                    return json.load(f)["RepetitionTime"]
            current = current.parent

    raise FileNotFoundError(f"No JSON metadata found for {bold_file.name}")
