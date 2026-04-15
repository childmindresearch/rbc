"""Shared fixtures for full-pipeline e2e tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from rbc.metadata import FunctionalMetadata
from rbc.workflows import anatomical_preprocess, functional_preprocess
from rbc.workflows.functional import _warp_mask_to_template
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Generator

    import niwrap
    from conftest import TestSubjectData

    from rbc.workflows import AnatomicalOutputs, FunctionalOutputs

MANIFEST_PATH = Path(__file__).parent / ".last_run.json"


class PipelineData(NamedTuple):
    """Shared outputs from the anatomical + functional preprocessing chain."""

    anat: AnatomicalOutputs
    func: FunctionalOutputs
    template_brain_mask: Path


@pytest.fixture(scope="session")
def manifest() -> Generator[dict[str, object], None, None]:
    """Shared manifest that tests populate; written to disk at session end."""
    data: dict[str, object] = {}
    yield data
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))


def _to_dict(obj: object) -> dict | str | None:
    """Convert workflow NamedTuple object to dict recursively."""
    if hasattr(obj, "_asdict"):
        return {k: _to_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return str(obj) if obj is not None else None


@pytest.fixture(scope="session")
def pipeline_data(
    test_subject: TestSubjectData,
    niwrap_runner: niwrap.Runner,  # noqa: ARG001 — ensures runner is configured
    manifest: dict[str, object],
) -> PipelineData:
    """Run anatomical and functional preprocessing once for all e2e tests."""
    anat = anatomical_preprocess(test_subject.t1w)
    func_metadata = FunctionalMetadata.load(test_subject.bold)
    func = functional_preprocess(
        in_bold=test_subject.bold,
        t1w_brain=anat.brain,
        wm_bbr_mask=anat.wm_bbr_mask,
        brain_mask=anat.brain_mask,
        csf_mask=anat.csf_mask,
        wm_mask=anat.wm_mask,
        anat_to_template=anat.anat_to_template_xfm,
        metadata=func_metadata,
    )
    template_brain_mask = _warp_mask_to_template(
        anat.brain_mask, REGISTRATION_TEMPLATES.brain_2mm, anat.anat_to_template_xfm
    )
    manifest["anat"] = _to_dict(anat)
    manifest["func"] = _to_dict(func)
    manifest["template_brain_mask"] = str(template_brain_mask)
    return PipelineData(anat=anat, func=func, template_brain_mask=template_brain_mask)
