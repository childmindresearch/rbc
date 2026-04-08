"""Full e2e test for the derivative metrics workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib

from rbc.workflows import metrics_pipeline
from rbc_resources import get_atlas

if TYPE_CHECKING:
    from full_pipeline.conftest import PipelineData


def _to_dict(obj: object) -> dict | str | None:
    """Convert workflow NamedTuple object to dict recursively."""
    if hasattr(obj, "_asdict"):
        return {k: _to_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return str(obj) if obj is not None else None


def test_single_session_metrics(
    pipeline_data: PipelineData,
    manifest: dict[str, object],
) -> None:
    """All 11 MetricsOutputs paths must exist on disk."""
    regressor = "36-parameter"  # default regressor
    regressed_bold = pipeline_data.func.regressed_bold[regressor]
    tr = float(nib.nifti1.load(regressed_bold).header["pixdim"][4])
    result = metrics_pipeline(
        regressed_bold=regressed_bold,
        cleaned_bold=pipeline_data.func.cleaned_bold[regressor],
        template_brain_mask=pipeline_data.template_brain_mask,
        tr=tr,
        atlas_files={"schaefer_200": get_atlas("schaefer_200")},
    )
    for output in result:
        paths = output.values() if isinstance(output, dict) else [output]
        for path in paths:
            assert Path(path).exists()

    manifest["metrics"] = _to_dict(result)
