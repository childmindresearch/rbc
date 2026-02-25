"""Shared fixtures for full-pipeline e2e tests."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, NamedTuple

import niwrap
import pytest

from rbc.cli import _DEFAULT_ENV_VARS
from rbc.core.resources import MNI_TEMPLATES
from rbc.workflows import anatomical_preprocess, functional_preprocess
from rbc.workflows.functional import _warp_mask_to_template

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestSubjectData

    from rbc.workflows import AnatomicalOutputs, FunctionalOutputs


class PipelineData(NamedTuple):
    """Shared outputs from the anatomical + functional preprocessing chain."""

    anat: AnatomicalOutputs
    func: FunctionalOutputs
    template_brain_mask: Path


@pytest.fixture(scope="session")
def _niwrap_session_runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> niwrap.Runner:
    """Session-scoped niwrap runner for full-pipeline tests."""
    match request.config.getoption("--runner").lower():
        case "docker":
            niwrap.use_docker()
        case "singularity":
            niwrap.use_singularity()
        case _:
            niwrap.use_local()
    runner = niwrap.get_global_runner()
    runner.environ = _DEFAULT_ENV_VARS
    data_dir = tmp_path_factory.mktemp("full_pipeline") / os.urandom(8).hex()
    data_dir.mkdir(parents=True, exist_ok=False)
    runner.data_dir = data_dir
    logger = logging.getLogger(runner.logger_name)
    logger.setLevel(logging.DEBUG)
    return runner


@pytest.fixture(scope="session")
def pipeline_data(
    test_subject: TestSubjectData, _niwrap_session_runner: niwrap.Runner
) -> PipelineData:
    """Run anatomical and functional preprocessing once for all e2e tests."""
    anat = anatomical_preprocess(test_subject.t1w)
    func = functional_preprocess(
        in_bold=test_subject.bold,
        t1w_brain=anat.brain,
        wm_bbr_mask=anat.wm_bbr_mask,
        brain_mask=anat.brain_mask,
        csf_mask=anat.csf_mask,
        wm_mask=anat.wm_mask,
        anat_to_template=anat.forward_xfm,
    )
    template_brain_mask = _warp_mask_to_template(
        anat.brain_mask, MNI_TEMPLATES.brain_1mm, anat.forward_xfm
    )
    return PipelineData(anat=anat, func=func, template_brain_mask=template_brain_mask)
