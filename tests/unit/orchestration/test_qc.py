"""Unit tests for QC orchestration -- logging behaviour."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import polars as pl
import pytest

from rbc.core.qc.xcp import XCPQCMetrics
from rbc.orchestration import Filters
from rbc.workflows.qc import QCOutputs


def _mock_qc_outputs(
    *, regressor: str = "36-parameter", passed: bool = True
) -> QCOutputs:
    return QCOutputs(
        metrics={regressor: Mock(spec=XCPQCMetrics)},
        qc_file={regressor: Path("fake_workdir") / "qc.tsv"},
        passed=passed,
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Single MNI-space preprocessed BOLD run for QC tests."""
    return pl.DataFrame(
        {
            "datatype": ["func"],
            "suffix": ["bold"],
            "desc": ["preproc"],
            "space": ["MNI152NLin6Asym"],
            "sub": ["01"],
            "ses": ["baseline"],
            "task": ["rest"],
            "run": [None],
            "acq": [None],
            "dir": [None],
            "echo": [None],
            "part": [None],
            "rec": [None],
            "ext": [".nii.gz"],
            "root": ["/data"],
            "path": [
                "sub-01/ses-baseline/func/"
                "sub-01_ses-baseline_task-rest_space-MNI_bold.nii.gz"
            ],
        }
    )


@contextmanager
def _patch_qc_run(
    df: pl.DataFrame,
    *,
    qc_passed: bool = True,
) -> Generator[Mock, None, None]:
    """Patch external calls made by orchestration.qc.run()."""
    with (
        patch("rbc.orchestration.qc.init_runner"),
        patch(
            "rbc.orchestration.qc.load_table",
            side_effect=[df, *([df] * 100)],
        ),
        patch(
            "rbc.bids.query.find_file",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.orchestration.qc.single_session_qc",
            return_value=_mock_qc_outputs(passed=qc_passed),
        ) as mock_qc,
        patch("rbc.orchestration.qc.RunContext") as mock_ctx_cls,
    ):
        mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
        yield mock_qc


class TestQCLogging:
    """Tests for QC PASSED/FAILED logging in orchestration.qc.run()."""

    def test_passed_status_logged(
        self,
        sample_dataframe: pl.DataFrame,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """PASSED is logged when QC outputs indicate the run passed."""
        from rbc.orchestration.qc import run

        with (
            caplog.at_level(logging.INFO),
            _patch_qc_run(sample_dataframe, qc_passed=True),
        ):
            run(
                output_dir=tmp_path,
                filters=Filters(
                    participant_label=["01"],
                    session_label=["baseline"],
                    task="rest",
                ),
                regressors=["36-parameter"],
                start_tr=2,
            )
            assert any("PASSED" in msg for msg in caplog.messages)

    def test_failed_status_logged(
        self,
        sample_dataframe: pl.DataFrame,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """FAILED is logged when QC outputs indicate the run failed."""
        from rbc.orchestration.qc import run

        with (
            caplog.at_level(logging.INFO),
            _patch_qc_run(sample_dataframe, qc_passed=False),
        ):
            run(
                output_dir=tmp_path,
                filters=Filters(
                    participant_label=["01"],
                    session_label=["baseline"],
                    task="rest",
                ),
                regressors=["36-parameter"],
                start_tr=2,
            )
            assert any("FAILED" in msg for msg in caplog.messages)
