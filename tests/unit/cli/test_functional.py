"""Unit tests for Functional CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli import functional


def _make_groups(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
    task: str | None = None,
) -> tuple[pl.DataFrame, list[tuple]]:
    """Filter sample dataframe and build iter_session_files groups."""
    filtered_df = sample_dataframe.filter(
        pl.col("suffix") == "bold",
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        *([pl.col("task") == task] if task else []),
    )
    groups = [
        (
            filtered_df.filter(
                pl.col("sub") == row["sub"],
                pl.col("ses") == row["ses"],
                pl.col("task") == row["task"],
            ),
            pl.DataFrame(),  # anat_df placeholder
        )
        for row in filtered_df.unique(["sub", "ses", "task"]).iter_rows(named=True)
    ]
    return filtered_df, groups


def _mock_functional_outputs() -> Mock:
    """Create a mock FunctionalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.sbref = fake / "sbref.nii.gz"
    outputs.motion_corrected_bold = fake / "bold_preproc.nii.gz"
    outputs.motion_params = fake / "motion.1D"
    outputs.rms_rel = fake / "rms_rel.rms"
    outputs.rms_abs = fake / "rms_abs.rms"
    outputs.bold_mask = fake / "bold_mask.nii.gz"
    outputs.bold_to_anat_matrix = fake / "bold_to_anat.mat"
    outputs.regressor_file = fake / "regressors.1D"
    return outputs


@contextmanager
def _patch_functional(
    filtered_df: pl.DataFrame, groups: list[tuple]
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for functional tests."""
    # Group by (sub, ses) to match how main() calls iter_session_files
    sub_ses_groups: dict[tuple, list] = {}
    for func_df, anat_df in groups:
        if func_df.is_empty():
            continue
        row = func_df.row(0, named=True)
        key = (row["sub"], row["ses"])
        sub_ses_groups.setdefault(key, [])
        sub_ses_groups[key].append((func_df, anat_df))

    with (
        patch("rbc.cli.functional.load_table", return_value=filtered_df),
        patch("rbc.cli.functional.load_session", return_value=Mock()),
        patch(
            "rbc.cli.functional.iter_session_files",
            side_effect=list(sub_ses_groups.values()),
        ),
        patch(
            "rbc.cli.functional.get_file_path",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.cli.functional.single_session_preprocess",
            return_value=_mock_functional_outputs(),
        ) as mock_preprocess,
        patch("rbc.cli.functional.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_preprocess, mock_ctx_cls


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.functional.setup_runner") as mock:
        ctx = Mock()
        ctx.runner = Mock()
        ctx.logger = Mock()
        ctx.verbose = False
        mock.return_value = ctx
        yield mock


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base argument namespace."""
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
        regressor="36-parameter",
        task=None,
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe for testing."""
    return pl.DataFrame(
        {
            "datatype": ["func", "func", "func", "func", "anat"],
            "suffix": ["bold", "bold", "bold", "bold", "T1w"],
            "ext": [".nii.gz", ".nii.gz", ".nii.gz", ".nii.gz", ".nii.gz"],
            "sub": ["01", "01", "02", "01", "01"],
            "ses": ["baseline", "baseline", "baseline", "vis2", "baseline"],
            "task": ["rest", "nback", "rest", "rest", None],
            "run": [None, None, None, None, None],
            "root": ["/data"] * 5,
            "path": [
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-nback_bold.nii.gz",
                "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-vis2/func/sub-01_ses-vis2_task-rest_bold.nii.gz",
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
            ],
        }
    )


class TestFunctionalArgs:
    """Tests for FunctionalArgs validation."""

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""
        args = functional.FunctionalArgs.validate_namespace(base_args)
        assert isinstance(args, functional.FunctionalArgs)

    @pytest.mark.parametrize(
        "task",
        ["rest", "nback", "faces+n+back", "task123", None],
        ids=["simple", "alphanumeric", "plus_separator", "with_digits", "none"],
    )
    def test_valid_task_labels(
        self, base_args: argparse.Namespace, task: str | None
    ) -> None:
        """Tests valid task labels pass validation."""
        base_args.task = task
        args = functional.FunctionalArgs.validate_namespace(base_args)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels(
        self, base_args: argparse.Namespace, task: str
    ) -> None:
        """Tests invalid task labels raise ValueError."""
        base_args.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            functional.FunctionalArgs.validate_namespace(base_args)


class TestFunctional:
    """Testing suite for functional processing."""

    @pytest.mark.parametrize(
        ("participant", "session", "task", "expected_count"),
        [
            ([], [], None, 4),  # All bold files
            (["01"], [], None, 3),  # All sub-01 bold
            ([], ["baseline"], None, 3),  # All ses-baseline bold
            (["01"], ["baseline"], None, 2),  # sub-01_ses-baseline
            (["01"], ["baseline"], "rest", 1),  # sub-01_ses-baseline_task-rest
            (["01", "02"], ["baseline"], None, 3),  # sub-01 & sub-02 ses-baseline
            (["01"], ["baseline", "vis2"], None, 3),  # sub-01, both sessions
            (["99"], [], None, 0),  # No matches
        ],
        ids=[
            "filter_by_type",
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
        """Test various filtering scenarios using parametrization."""
        base_args.participant_label = participant
        base_args.session_label = session
        base_args.task = task
        args = functional.FunctionalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, participant, session, task)

        with _patch_functional(filtered_df, groups) as (mock_preprocess, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = functional.main(args)
            assert result == 0
            assert mock_preprocess.call_count == expected_count

    def test_filtering_validate_files_processed(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that correct files are processed after filtering."""
        participant, session, task = ["01"], ["baseline"], "rest"
        base_args.participant_label = participant
        base_args.session_label = session
        base_args.task = task
        args = functional.FunctionalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, participant, session, task)

        with _patch_functional(filtered_df, groups) as (mock_preprocess, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            functional.main(args)
            assert mock_preprocess.call_count == 1
            processed_path = mock_preprocess.call_args[1]["in_bold"]
            assert "sub-01" in str(processed_path)
            assert "baseline" in str(processed_path)
            assert "rest" in str(processed_path)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test runner environment variables are configured correctly."""
        from rbc.core import CPAC_ANTS_SEED

        args = functional.FunctionalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, [], [], None)

        with (
            patch("rbc.cli.functional.setup_runner") as mock_setup,
            _patch_functional(filtered_df, groups) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            functional.main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }
