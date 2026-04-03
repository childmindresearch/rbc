"""Pipeline orchestration layer.

Provides ``run()`` entry points for each workflow that handle BIDS table
loading, filtering, sub/ses iteration, and the discover-process-export
loop. CLI modules delegate to these after parsing arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


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
