"""Unit tests for Longitudinal CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli.longitudinal import LongitudinalArgs, _require_file, main


def _make_anat_groups(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
) -> tuple[pl.DataFrame, list[tuple]]:
    """Filter sample dataframe and build iter_session_files groups."""
    filtered_df = sample_dataframe.filter(
        pl.col("suffix") == "T1w",
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
    )
    groups = [
        (
            pl.DataFrame(),  # func_df placeholder
            filtered_df.filter(
                pl.col("sub") == row["sub"],
                pl.col("ses") == row["ses"],
            ),
        )
        for row in filtered_df.unique(["sub", "ses"]).iter_rows(named=True)
    ]
    return filtered_df, groups


def _mock_anatomical_outputs(all_present: bool = True) -> Mock:  # noqa: FBT001, FBT002
    """Create a mock LongitudinalAnatomicalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.brain = fake / "brain.nii.gz"
    outputs.brain_mask = (fake / "brain_mask.nii.gz") if all_present else None
    outputs.csf_mask = (fake / "csf_mask.nii.gz") if all_present else None
    outputs.gm_mask = (fake / "gm_mask.nii.gz") if all_present else None
    outputs.wm_mask = (fake / "wm_mask.nii.gz") if all_present else None
    return outputs


@contextmanager
def _patch_longitudinal_anat(
    filtered_df: pl.DataFrame,
    groups: list[tuple],
    all_outputs_present: bool = True,  # noqa: FBT001, FBT002
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for longitudinal anatomical tests."""
    sub_ses_groups: dict[tuple, list] = {}
    for func_df, anat_df in groups:
        if anat_df.is_empty():
            continue
        row = anat_df.row(0, named=True)
        key = (row["sub"], row["ses"])
        sub_ses_groups.setdefault(key, [])
        sub_ses_groups[key].append((func_df, anat_df))

    with (
        patch("rbc.cli.longitudinal.load_table", return_value=filtered_df),
        patch("rbc.cli.longitudinal.load_session", return_value=Mock()),
        patch(
            "rbc.cli.longitudinal.iter_session_files",
            side_effect=list(sub_ses_groups.values()),
        ),
        patch(
            "rbc.cli.longitudinal.get_file_path",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.cli.longitudinal.anatomical_longitudinal",
            return_value=_mock_anatomical_outputs(all_outputs_present),
        ) as mock_process,
        patch("rbc.cli.longitudinal.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_process, mock_ctx_cls


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.longitudinal.setup_runner") as mock:
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
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        participant_label=[],
        session_label=[],
        anatomical=True,
        functional=False,
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe for testing."""
    return pl.DataFrame(
        {
            "datatype": ["anat", "anat", "anat", "anat"],
            "suffix": ["T1w", "T1w", "T1w", "T1w"],
            "ext": [".nii.gz", ".nii.gz", ".nii.gz", ".nii.gz"],
            "sub": ["01", "01", "02", "01"],
            "ses": ["baseline", "vis2", "baseline", "baseline"],
            "run": [None, None, None, None],
            "root": ["/data"] * 4,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-vis2/anat/sub-01_ses-vis2_T1w.nii.gz",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii.gz",
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_run-2_T1w.nii.gz",
            ],
        }
    )


class TestRequireFile:
    """Tests for _require_file helper."""

    def test_returns_path_when_present(self) -> None:
        """Test _require_file returns path when not None."""
        path = Path("some/file.nii.gz")
        assert _require_file(path, "brain") == path

    def test_raises_when_none(self) -> None:
        """Test _require_file raises ValueError when path is None."""
        with pytest.raises(ValueError, match="'brain_mask'"):
            _require_file(None, "brain_mask")


class TestLongitudinalArgs:
    """Tests for LongitudinalArgs validation."""

    def test_validate_anatomical(self, base_args: argparse.Namespace) -> None:
        """Test LongitudinalArgs validates successfully with anatomical flag."""
        args = LongitudinalArgs.validate_namespace(base_args)
        assert isinstance(args, LongitudinalArgs)
        assert args.anatomical is True
        assert args.functional is False

    def test_functional_raises_not_implemented(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test that functional flag raises NotImplementedError."""
        base_args.functional = True
        with pytest.raises(
            NotImplementedError, match="Functional longitudinal pipeline"
        ):
            LongitudinalArgs.validate_namespace(base_args)

    def test_neither_flag_raises_value_error(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test that neither flag raises ValueError."""
        base_args.anatomical = False
        base_args.functional = False
        with pytest.raises(
            ValueError, match="At least one of '--anatomical' or '--functional'"
        ):
            LongitudinalArgs.validate_namespace(base_args)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Test default values are preserved."""
        args = LongitudinalArgs.validate_namespace(base_args)
        assert args.participant_label == []
        assert args.session_label == []


class TestLongitudinalAnat:
    """Testing suite for longitudinal anatomical processing."""

    @pytest.mark.parametrize(
        ("participant", "session", "expected_count"),
        [
            ([], [], 3),
            (["01"], [], 2),
            ([], ["baseline"], 2),
            (["01"], ["baseline"], 1),
            (["99"], [], 0),
        ],
        ids=[
            "all",
            "filter_by_participant",
            "filter_by_session",
            "filter_by_participant_and_session",
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
        expected_count: int,
    ) -> None:
        """Test various filtering scenarios using parametrization."""
        base_args.participant_label = participant
        base_args.session_label = session
        args = LongitudinalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_anat_groups(sample_dataframe, participant, session)

        with _patch_longitudinal_anat(filtered_df, groups) as (
            mock_process,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = main(args)
            assert result == 0
            assert mock_process.call_count == expected_count

    def test_missing_optional_outputs_raise(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that None optional outputs raise ValueError via _require_file."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_anat_groups(sample_dataframe, ["01"], ["baseline"])

        with _patch_longitudinal_anat(
            filtered_df, groups, all_outputs_present=False
        ) as (
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            with pytest.raises(ValueError, match="is missing"):
                main(args)

    def test_get_anat_file_returns_none_on_missing(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that _get_anat_file returns None on FileNotFoundError."""
        args = LongitudinalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_anat_groups(sample_dataframe, ["01"], ["baseline"])

        with (
            _patch_longitudinal_anat(filtered_df, groups) as (_, mock_ctx_cls),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=FileNotFoundError,
            ),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            with pytest.raises(ValueError, match="'brain'"):
                main(args)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test runner environment variables are configured correctly."""
        from rbc.core import CPAC_ANTS_SEED

        args = LongitudinalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_anat_groups(sample_dataframe, [], [])

        with (
            patch("rbc.cli.longitudinal.setup_runner") as mock_setup,
            _patch_longitudinal_anat(filtered_df, groups) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }

    def test_experimental_warning_emitted(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test that the experimental workflow warning is logged."""
        args = LongitudinalArgs.validate_namespace(base_args)
        filtered_df, groups = _make_anat_groups(sample_dataframe, [], [])

        with (
            patch("rbc.cli.longitudinal.setup_runner") as mock_setup,
            _patch_longitudinal_anat(filtered_df, groups) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            main(args)
            ctx.logger.warning.assert_called_once()
            assert "experimental" in ctx.logger.warning.call_args[0][0].lower()
