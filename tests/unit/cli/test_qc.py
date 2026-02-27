"""Unit tests for QC CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli import qc
from rbc.cli.main import cli, create_parser
from rbc.cli.qc import QCArgs


def _make_filtered_df(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
    task: str | None = None,
) -> pl.DataFrame:
    """Filter sample dataframe to match QC main() filtering logic."""
    return sample_dataframe.filter(
        pl.col("datatype") == "func",
        pl.col("suffix") == "bold",
        pl.col("desc") == "preproc",
        pl.col("space") == "MNI152NLin6ASym",
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        *([pl.col("task") == task] if task else []),
    )


def _mock_qc_outputs(*, passed: bool = True) -> Mock:
    """Create a mock QCOutputs with a fake qc_file path and pass/fail status."""
    outputs = Mock()
    outputs.qc_file = Path("fake_workdir") / "qc.tsv"
    outputs.passed = passed
    return outputs


@contextmanager
def _patch_qc(
    filtered_df: pl.DataFrame,
    deriv_df: pl.DataFrame,
    *,
    qc_passed: bool = True,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for QC tests.

    Patches load_table to return filtered_df on first call (primary BOLD
    filtering) and deriv_df on subsequent calls (derivative lookups per
    sub/ses group). Also patches get_file_path, single_session_qc, and
    PipelineContext.
    """
    with (
        patch("rbc.cli.qc.load_table", side_effect=[filtered_df, *([deriv_df] * 100)]),
        patch(
            "rbc.cli.qc.get_file_path", return_value=Path("fake_workdir/file.nii.gz")
        ),
        patch(
            "rbc.cli.qc.single_session_qc",
            return_value=_mock_qc_outputs(passed=qc_passed),
        ) as mock_qc,
        patch("rbc.cli.qc.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_qc, mock_ctx_cls


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.qc.setup_runner") as mock:
        ctx = Mock()
        ctx.runner = Mock()
        ctx.logger = Mock()
        ctx.verbose = False
        mock.return_value = ctx
        yield mock


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base QC argument namespace with sensible defaults."""
    input_dir = tmp_path / "input"
    input_dir.touch()
    output_dir = tmp_path / "output"
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dir=input_dir,
        output_dir=output_dir,
        participant_label=[],
        session_label=[],
        task=None,
        start_tr=2,
        regressor="36-parameter",
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Preprocessed derivatives dataframe simulating QC input.

    Contains:
    - sub-01/ses-baseline: two bold runs (task-rest, task-nback) in MNI space
    - sub-02/ses-baseline: one bold run (task-rest) in MNI space
    - sub-01/ses-vis2: one bold run (task-rest) in MNI space
    - One non-MNI bold row to verify space filtering
    """
    return pl.DataFrame(
        {
            "datatype": ["func", "func", "func", "func", "func"],
            "suffix": ["bold", "bold", "bold", "bold", "bold"],
            "desc": ["preproc", "preproc", "preproc", "preproc", "preproc"],
            "space": [
                "MNI152NLin6ASym",
                "MNI152NLin6ASym",
                "MNI152NLin6ASym",
                "MNI152NLin6ASym",
                "T1w",  # Should be excluded by space filter
            ],
            "sub": ["01", "01", "02", "01", "01"],
            "ses": ["baseline", "baseline", "baseline", "vis2", "baseline"],
            "task": ["rest", "nback", "rest", "rest", "rest"],
            "run": [None, None, None, None, None],
            "ext": [".nii.gz"] * 5,
            "root": ["/data"] * 5,
            "path": [
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_space-MNI_bold.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-nback_space-MNI_bold.nii.gz",
                "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_space-MNI_bold.nii.gz",
                "sub-01/ses-vis2/func/sub-01_ses-vis2_task-rest_space-MNI_bold.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_space-T1w_bold.nii.gz",
            ],
        }
    )


class TestQCArgs:
    """Tests for QCArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task and start_tr.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = QCArgs.validate_namespace(base_args)
        assert isinstance(args, QCArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for task, start_tr, regressor, and labels are preserved."""
        args = QCArgs.validate_namespace(base_args)
        assert args.task is None
        assert args.start_tr == 2
        assert args.regressor == "36-parameter"
        assert args.participant_label == []
        assert args.session_label == []

    def test_custom_start_tr(self, base_args: argparse.Namespace) -> None:
        """Custom start_tr is correctly preserved through validation."""
        base_args.start_tr = 5
        args = QCArgs.validate_namespace(base_args)
        assert args.start_tr == 5

    def test_invalid_start_tr_zero(self, base_args: argparse.Namespace) -> None:
        """start_tr of 0 raises ValueError."""
        base_args.start_tr = 0
        with pytest.raises(ValueError, match="Start TR should be greater than 0"):
            QCArgs.validate_namespace(base_args)

    def test_invalid_start_tr_negative(self, base_args: argparse.Namespace) -> None:
        """Negative start_tr raises ValueError."""
        base_args.start_tr = -1
        with pytest.raises(ValueError, match="Start TR should be greater than 0"):
            QCArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = regressor
        args = QCArgs.validate_namespace(base_args)
        assert args.regressor == regressor

    def test_task_preserved(self, base_args: argparse.Namespace) -> None:
        """Provided task label is preserved through validation."""
        base_args.task = "rest"
        args = QCArgs.validate_namespace(base_args)
        assert args.task == "rest"

    @pytest.mark.parametrize(
        "task",
        ["rest", "nback", "faces+n+back", "task123", None],
        ids=["simple", "alphanumeric", "plus_separator", "with_digits", "none"],
    )
    def test_valid_task_labels(
        self, base_args: argparse.Namespace, task: str | None
    ) -> None:
        """Valid task labels pass validation."""
        base_args.task = task
        args = QCArgs.validate_namespace(base_args)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels(
        self, base_args: argparse.Namespace, task: str
    ) -> None:
        """Invalid task labels raise ValueError."""
        base_args.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            QCArgs.validate_namespace(base_args)


class TestQCFiltering:
    """Tests for subject/session/task/space filtering in QC main()."""

    @pytest.mark.parametrize(
        ("participant", "session", "task", "expected_count"),
        [
            ([], [], None, 4),  # All MNI bold runs
            (["01"], [], None, 3),  # All sub-01 bold runs
            ([], ["baseline"], None, 3),  # All ses-baseline bold runs
            (["01"], ["baseline"], None, 2),  # sub-01_ses-baseline only
            (["01"], ["baseline"], "rest", 1),  # sub-01_ses-baseline_task-rest only
            (["01", "02"], ["baseline"], None, 3),  # sub-01 & sub-02 ses-baseline
            (["01"], ["baseline", "vis2"], None, 3),  # sub-01, both sessions
            (["99"], [], None, 0),  # No matches
        ],
        ids=[
            "all_mni_bold",
            "filter_by_participant",
            "filter_by_session",
            "filter_by_participant_and_session",
            "filter_by_all",
            "filter_by_multi_participant",
            "filter_by_multi_session",
            "no_matches",
        ],
    )
    def test_filtering(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
        participant: list[str],
        session: list[str],
        task: str | None,
        expected_count: int,
    ) -> None:
        """QC is invoked exactly once per matching bold run after filtering."""
        base_args.participant_label = participant
        base_args.session_label = session
        base_args.task = task
        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, participant, session, task)

        with _patch_qc(filtered_df, sample_dataframe) as (mock_qc, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = qc.main(args)
            assert result == 0
            assert mock_qc.call_count == expected_count

    def test_space_filter_excludes_non_mni(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Non-MNI bold rows are excluded regardless of other filters."""
        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, [], [], None)

        with _patch_qc(filtered_df, sample_dataframe) as (mock_qc, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            qc.main(args)
            assert mock_qc.call_count == 4


class TestQCOutputs:
    """Tests for QC output logging and export behaviour."""

    def test_passed_status_logged(
        self,
        mock_setup: Mock,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """PASSED is logged when QC outputs indicate the run passed."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_qc(filtered_df, sample_dataframe, qc_passed=True) as (
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            qc.main(args)
            log_calls = [
                str(c) for c in mock_setup.return_value.logger.info.call_args_list
            ]
            assert any("PASSED" in c for c in log_calls)

    def test_failed_status_logged(
        self,
        mock_setup: Mock,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """FAILED is logged when QC outputs indicate the run failed."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_qc(filtered_df, sample_dataframe, qc_passed=False) as (
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            qc.main(args)
            log_calls = [
                str(c) for c in mock_setup.return_value.logger.info.call_args_list
            ]
            assert any("FAILED" in c for c in log_calls)

    def test_qc_export_called(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """pipe_ctx.export is called once per QC run with correct suffix and desc."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_qc(filtered_df, sample_dataframe) as (_, mock_ctx_cls):
            mock_pipe_ctx = Mock(sub="01", ses="baseline")
            mock_ctx_cls.return_value = mock_pipe_ctx
            qc.main(args)
            assert mock_pipe_ctx.export.call_count == 1
            export_kwargs = mock_pipe_ctx.export.call_args[1]
            assert export_kwargs["suffix"] == "quality"
            assert export_kwargs["desc"] == "xcp"


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Runner environment variables are set to the expected defaults."""
        from rbc.core import CPAC_ANTS_SEED

        args = QCArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, [], [], None)

        with (
            patch("rbc.cli.qc.setup_runner") as mock_setup,
            _patch_qc(filtered_df, sample_dataframe) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            qc.main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }


class TestQCRegistration:
    """Test that QC command is registered and discoverable."""

    def test_qc_command_registered(self) -> None:
        """Test QC workflow is available in parser."""
        result = cli(["/input", "/output", "qc", "--help"])
        assert result == 0

    def test_qc_parser_has_task(self) -> None:
        """Test QC subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc", "--task", "rest"])
        assert args.task == "rest"

    def test_qc_parser_has_start_tr(self) -> None:
        """Test QC subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_qc_parser_has_regressor(self) -> None:
        """Test QC subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.regressor == "36-parameter"

    def test_qc_parser_task_default_none(self) -> None:
        """Test QC subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.task is None

    def test_qc_parser_start_tr_default(self) -> None:
        """Test QC subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.start_tr == 2
