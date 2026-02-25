"""Full e2e test for the derivative metrics workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbc.workflows import metrics_pipeline

if TYPE_CHECKING:
    from full_pipeline.conftest import PipelineData


def test_single_session_metrics(
    pipeline_data: PipelineData, manifest: dict[str, object]
) -> None:
    """All 11 MetricsOutputs paths must exist on disk."""
    result = metrics_pipeline(
        cleaned_bold=pipeline_data.func.cleaned_bold,
        template_brain_mask=pipeline_data.template_brain_mask,
    )
    for path in result:
        assert Path(path).exists()

    manifest["metrics"] = {k: str(v) for k, v in result._asdict().items()}
