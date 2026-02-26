"""Full e2e test for the functional preprocessing workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from full_pipeline.conftest import PipelineData


def test_single_session_preprocess(pipeline_data: PipelineData) -> None:
    """All 16 FunctionalOutputs paths must exist on disk."""
    for path in pipeline_data.func:
        assert Path(path).exists()
