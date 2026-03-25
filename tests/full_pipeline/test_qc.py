"""Full e2e test for the QC metrics workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.core.qc.xcp import XCPQCMetrics
from rbc.workflows import qc_pipeline

if TYPE_CHECKING:
    from conftest import TestSubjectData
    from full_pipeline.conftest import PipelineData


def test_single_session_qc(
    pipeline_data: PipelineData,
    test_subject: TestSubjectData,
    manifest: dict[str, object],
) -> None:
    """QC pipeline returns metrics, a TSV file, and a pass/fail flag."""
    result = qc_pipeline(
        template_bold=pipeline_data.func.template_bold,
        cleaned_bold=pipeline_data.func.cleaned_bold,
        motion_params=pipeline_data.func.motion_params,
        rms_rel=pipeline_data.func.rms_rel,
        bold_mask=pipeline_data.func.bold_mask,
        brain_mask=pipeline_data.anat.brain_mask,
        bold_to_anat_matrix=pipeline_data.func.bold_to_anat_matrix,
        template_brain_mask=pipeline_data.template_brain_mask,
        sub=test_subject.subject_id,
        ses="001",
        task="balloonanalogrisktask",
        run=1,
    )
    assert all(isinstance(v, XCPQCMetrics) for v in result.metrics.values())
    assert all(path.exists() for path in result.qc_file.values())
    assert isinstance(result.passed, bool)

    manifest["qc"] = {
        regressor: {
            "qc_file": str(result.qc_file[regressor]),
            "passed": result.passed,
            "metrics": result.metrics[regressor]._asdict(),
        }
        for regressor in result.metrics
    }
