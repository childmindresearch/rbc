"""Shared fixtures for full-pipeline e2e tests."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import niwrap
import pytest
from styxpodman import PodmanRunner

from rbc.cli import _DEFAULT_ENV_VARS
from rbc.workflows import anatomical_preprocess, functional_preprocess
from rbc.workflows.functional import _warp_mask_to_template
from rbc_resources import MNI_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Generator

    from conftest import TestSubjectData

    from rbc.workflows import AnatomicalOutputs, FunctionalOutputs

MANIFEST_PATH = Path(__file__).parent / ".last_run.json"


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
        case "podman":
            niwrap.set_global_runner(
                # UserID = 0 currently necessary for user mapping
                runner=PodmanRunner(podman_user_id=0)
            )
        case "singularity":
            niwrap.use_singularity()
        case _:
            niwrap.use_local()
    runner = niwrap.get_global_runner()
    # Override single-threaded ANTs for e2e tests — deterministic results
    # aren't needed here, and multi-threading cuts registration time ~3-5x.
    runner.environ = {
        **_DEFAULT_ENV_VARS,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(min(os.cpu_count() or 1, 4)),
    }
    data_dir = tmp_path_factory.mktemp("full_pipeline") / os.urandom(8).hex()
    data_dir.mkdir(parents=True, exist_ok=False)
    runner.data_dir = data_dir
    logger = logging.getLogger(runner.logger_name)
    logger.setLevel(logging.DEBUG)
    return runner


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
    _niwrap_session_runner: niwrap.Runner,
    manifest: dict[str, object],
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
        anat.brain_mask, MNI_TEMPLATES.brain_2mm, anat.forward_xfm
    )
    manifest["anat"] = _to_dict(anat)
    manifest["func"] = _to_dict(func)
    manifest["template_brain_mask"] = str(template_brain_mask)
    return PipelineData(anat=anat, func=func, template_brain_mask=template_brain_mask)
