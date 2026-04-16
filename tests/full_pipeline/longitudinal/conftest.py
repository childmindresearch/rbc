"""Shared fixtures for longitudinal full-pipeline e2e tests.

Runs the longitudinal template + anatomical + functional stages once
per session, caching outputs for all tests in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from rbc.metadata import FunctionalMetadata
from rbc.workflows import anatomical_preprocess, functional_preprocess
from rbc.workflows.longitudinal.functional import (
    FunctionalLongOutputs,
)
from rbc.workflows.longitudinal.functional import (
    longitudinal_process as functional_longitudinal,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import niwrap
    from conftest import TestSubjectData

    from rbc.workflows import AnatomicalOutputs, FunctionalOutputs

MANIFEST_PATH = Path(__file__).parent / ".last_run_longitudinal.json"


class LongitudinalPipelineData(NamedTuple):
    """Shared outputs from the longitudinal functional pipeline."""

    anat: AnatomicalOutputs
    func: FunctionalOutputs
    long_func: FunctionalLongOutputs


def _to_dict(obj: object) -> dict | str | None:
    """Convert workflow NamedTuple object to dict recursively."""
    if hasattr(obj, "_asdict"):
        return {k: _to_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return str(obj) if obj is not None else None


@pytest.fixture(scope="session")
def longitudinal_manifest() -> Generator[dict[str, object], None, None]:
    """Shared manifest written to disk at session end."""
    data: dict[str, object] = {}
    yield data
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))


@pytest.fixture(scope="session")
def longitudinal_pipeline_data(
    test_subject: TestSubjectData,
    niwrap_runner: niwrap.Runner,  # noqa: ARG001
    longitudinal_manifest: dict[str, object],
) -> LongitudinalPipelineData:
    """Run cross-sectional + longitudinal functional pipeline once.

    Since the full_pipeline test dataset (ds000001) is single-session,
    we simulate longitudinal processing by treating the cross-sectional
    outputs as if they came from a multi-session subject. The longitudinal
    workflow only needs a template, xfm, and the cross-sectional BOLD +
    regressors, so we can construct a synthetic longitudinal template from
    the anatomical outputs.
    """
    # Cross-sectional anat
    anat = anatomical_preprocess(test_subject.t1w)

    # Cross-sectional func
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

    # Longitudinal functional: use anat-to-template xfm as the
    # "longitudinal template" xfm and the brain as the "template" image.
    # This is not anatomically meaningful but exercises the full pipeline
    # chain through regression reuse.
    long_func = functional_longitudinal(
        template=anat.brain,
        anat_to_template_xfm=anat.anat_to_template_xfm,
        bold_to_anat_itk=func.bold_to_anat_itk,
        sbref=func.sbref,
        bold=func.preproc_bold,
        bold_mask=func.bold_mask,
        regressor_files=func.regressor_file,
        regressor_set=["36-parameter"],
    )

    longitudinal_manifest["anat"] = _to_dict(anat)
    longitudinal_manifest["func"] = _to_dict(func)
    longitudinal_manifest["long_func"] = _to_dict(long_func)

    return LongitudinalPipelineData(anat=anat, func=func, long_func=long_func)
