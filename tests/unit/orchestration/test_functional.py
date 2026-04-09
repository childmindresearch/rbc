"""Unit tests for functional orchestration -- anat_inputs handoff."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import polars as pl

from rbc.bids.functional import FunctionalInputs, FunctionalRun
from rbc.context import RunContext
from rbc.metadata import FunctionalMetadata

_FAKE = Path("fake_workdir")

_ANAT_INPUTS: FunctionalInputs = {
    "t1w_brain": _FAKE / "brain.nii.gz",
    "wm_bbr_mask": _FAKE / "wm_bbr.nii.gz",
    "brain_mask": _FAKE / "brain_mask.nii.gz",
    "csf_mask": _FAKE / "csf_mask.nii.gz",
    "wm_mask": _FAKE / "wm_mask.nii.gz",
    "anat_to_template": _FAKE / "xfm.nii.gz",
}

_BOLD_DF = pl.DataFrame(
    {
        "datatype": ["func"],
        "suffix": ["bold"],
        "ext": [".nii.gz"],
        "sub": ["01"],
        "ses": ["baseline"],
        "task": ["rest"],
        "run": [None],
        "acq": [None],
        "dir": [None],
        "echo": [None],
        "part": [None],
        "rec": [None],
        "desc": [None],
        "space": [None],
        "root": ["/data"],
        "path": [
            "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz"
        ],
    }
)

_ANAT_DF = pl.DataFrame(
    {
        "datatype": ["anat"],
        "suffix": ["T1w"],
        "ext": [".nii.gz"],
        "sub": ["01"],
        "ses": ["baseline"],
        "task": [None],
        "run": [None],
        "acq": [None],
        "dir": [None],
        "echo": [None],
        "part": [None],
        "rec": [None],
        "desc": [None],
        "space": [None],
        "root": ["/data"],
        "path": [
            "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"
        ],
    }
)


def _mock_func_run() -> FunctionalRun:
    return FunctionalRun(
        path=Path("/data/sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz"),
        entities={"task": "rest"},
        anat_df=_ANAT_DF,
    )


def _mock_func_outputs() -> Mock:
    m = Mock()
    m.regressed_bold = {"36-parameter": _FAKE / "regressed.nii.gz"}
    m.cleaned_bold = {"36-parameter": _FAKE / "cleaned.nii.gz"}
    m.template_brain_mask = _FAKE / "tmpl_mask.nii.gz"
    m.template_bold = _FAKE / "tmpl_bold.nii.gz"
    m.motion_params = _FAKE / "motion.par"
    m.rms_rel = _FAKE / "rms.txt"
    m.bold_mask = _FAKE / "bold_mask.nii.gz"
    m.bold_to_anat_matrix = _FAKE / "bold2anat.mat"
    return m


def _mock_metadata() -> Mock:
    m = Mock(spec=FunctionalMetadata)
    m.tr = 2.0
    return m


@contextmanager
def _patch_process_session() -> Generator[tuple[Mock, Mock, Mock], None, None]:
    """Patch external calls made by functional.process_session()."""
    func_run = _mock_func_run()
    with (
        patch(
            "rbc.orchestration.functional.discover_functional",
            return_value=[func_run],
        ),
        patch(
            "rbc.orchestration.functional.resolve_functional",
            return_value=_ANAT_INPUTS,
        ) as mock_resolve,
        patch(
            "rbc.orchestration.functional.FunctionalMetadata.load",
            return_value=_mock_metadata(),
        ),
        patch(
            "rbc.orchestration.functional.single_session_preprocess",
            return_value=_mock_func_outputs(),
        ) as mock_preprocess,
        patch(
            "rbc.orchestration.functional.export_functional",
            return_value=Mock(),
        ),
    ):
        yield mock_resolve, mock_preprocess, func_run


class TestProcessSessionAnatInputs:
    """Tests for the anat_inputs parameter on process_session."""

    def test_anat_inputs_skips_resolve(self, tmp_path: Path) -> None:
        """When anat_inputs is provided, resolve_functional is not called."""
        from rbc.bids.session import SessionTables
        from rbc.orchestration.functional import process_session

        session = SessionTables(anat=_ANAT_DF, func=_BOLD_DF)
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)

        with _patch_process_session() as (mock_resolve, mock_preprocess, _):
            process_session(
                session,
                pipe_ctx,
                regressors=["36-parameter"],
                anat_inputs=_ANAT_INPUTS,
            )
            mock_resolve.assert_not_called()
            mock_preprocess.assert_called_once()

    def test_anat_inputs_passed_to_preprocess(self, tmp_path: Path) -> None:
        """When anat_inputs is provided, its paths are forwarded to preprocess."""
        from rbc.bids.session import SessionTables
        from rbc.orchestration.functional import process_session

        session = SessionTables(anat=_ANAT_DF, func=_BOLD_DF)
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)

        with _patch_process_session() as (_, mock_preprocess, _):
            process_session(
                session,
                pipe_ctx,
                regressors=["36-parameter"],
                anat_inputs=_ANAT_INPUTS,
            )
            call_kwargs = mock_preprocess.call_args.kwargs
            assert call_kwargs["t1w_brain"] == _ANAT_INPUTS["t1w_brain"]
            assert call_kwargs["brain_mask"] == _ANAT_INPUTS["brain_mask"]
            assert call_kwargs["wm_bbr_mask"] == _ANAT_INPUTS["wm_bbr_mask"]
            assert call_kwargs["csf_mask"] == _ANAT_INPUTS["csf_mask"]
            assert call_kwargs["wm_mask"] == _ANAT_INPUTS["wm_mask"]
            assert call_kwargs["anat_to_template"] == _ANAT_INPUTS["anat_to_template"]

    def test_no_anat_inputs_calls_resolve(self, tmp_path: Path) -> None:
        """When anat_inputs is None, resolve_functional is called."""
        from rbc.bids.session import SessionTables
        from rbc.orchestration.functional import process_session

        session = SessionTables(anat=_ANAT_DF, func=_BOLD_DF)
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)

        with _patch_process_session() as (mock_resolve, mock_preprocess, _):
            process_session(
                session,
                pipe_ctx,
                regressors=["36-parameter"],
            )
            mock_resolve.assert_called_once()
            mock_preprocess.assert_called_once()


class TestAllPipelineAnatHandoff:
    """Tests that all.run() passes anatomical outputs to functional stage."""

    def test_anat_outputs_forwarded_as_anat_inputs(self, tmp_path: Path) -> None:
        """all.run() constructs anat_inputs from AnatomicalOutputs fields."""
        from rbc.orchestration import Filters
        from rbc.orchestration.all import run
        from rbc.workflows.anatomical import AnatomicalOutputs

        anat_outputs = AnatomicalOutputs(
            brain=_FAKE / "brain.nii.gz",
            brain_mask=_FAKE / "brain_mask.nii.gz",
            brain_tpl=_FAKE / "brain_tpl.nii.gz",
            csf_mask=_FAKE / "csf_mask.nii.gz",
            gm_mask=_FAKE / "gm_mask.nii.gz",
            wm_mask=_FAKE / "wm_mask.nii.gz",
            wm_bbr_mask=_FAKE / "wm_bbr.nii.gz",
            forward_xfm=_FAKE / "fwd.nii.gz",
            inverse_xfm=_FAKE / "inv.nii.gz",
        )

        raw_df = pl.DataFrame(
            {
                "datatype": ["anat", "func"],
                "suffix": ["T1w", "bold"],
                "ext": [".nii.gz", ".nii.gz"],
                "sub": ["01", "01"],
                "ses": ["baseline", "baseline"],
                "task": [None, "rest"],
                "run": [None, None],
                "space": [None, None],
                "desc": [None, None],
                "root": ["/data", "/data"],
                "path": [
                    "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                    "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                ],
            }
        )

        with (
            patch("rbc.orchestration.all.init_runner"),
            patch("rbc.orchestration.all.load_table", return_value=raw_df),
            patch(
                "rbc.orchestration.all.process_anat",
                return_value=anat_outputs,
            ),
            patch("rbc.orchestration.all.process_func", return_value=[]) as mock_func,
            patch("rbc.orchestration.all.RunContext") as mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            run(
                input_dirs=[tmp_path],
                output_dir=tmp_path,
                filters=Filters(participant_label=["01"]),
                regressors=["36-parameter"],
                atlas_files={},
                fwhm=6.0,
                start_tr=2,
            )

            mock_func.assert_called_once()
            call_kwargs = mock_func.call_args.kwargs
            passed_inputs = call_kwargs["anat_inputs"]

            assert passed_inputs["t1w_brain"] == anat_outputs.brain
            assert passed_inputs["brain_mask"] == anat_outputs.brain_mask
            assert passed_inputs["csf_mask"] == anat_outputs.csf_mask
            assert passed_inputs["wm_mask"] == anat_outputs.wm_mask
            assert passed_inputs["wm_bbr_mask"] == anat_outputs.wm_bbr_mask
            assert passed_inputs["anat_to_template"] == anat_outputs.inverse_xfm
