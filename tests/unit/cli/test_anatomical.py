"""Unit tests for Anatomical CLI module."""

import argparse
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest

from rbc.cli import anatomical


@pytest.fixture
def mock_setup() -> Generator[MagicMock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.anatomical.setup_runner") as mock:
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
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe for testing."""
    return pl.DataFrame(
        {
            "datatype": ["anat", "anat", "func", "anat", "anat", "anat"],
            "suffix": ["T1w", "T2w", "bold", "T1w", "T1w", "T1w"],
            "ext": [".nii.gz", ".nii.gz", ".nii.gz", ".nii", ".json", ".nii.gz"],
            "sub": ["01", "01", "01", "02", "01", "01"],
            "ses": ["baseline", "baseline", "baseline", "baseline", "baseline", "vis2"],
            "run": [None, None, None, None, None, None],
            "root": ["/data"] * 6,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T2w.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_bold.nii.gz",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii",
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.json",
                "sub-01/ses-vis2/anat/sub-01_ses-vis2_T1w.nii.gz",
            ],
        }
    )


def _mock_anatomical_outputs() -> Mock:
    """Create a mock AnatomicalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.brain = fake / "brain.nii.gz"
    outputs.brain_mask = fake / "brain_mask.nii.gz"
    outputs.csf_mask = fake / "csf_mask.nii.gz"
    outputs.gm_mask = fake / "gm_mask.nii.gz"
    outputs.wm_mask = fake / "wm_mask.nii.gz"
    outputs.wm_bbr_mask = fake / "wm_bbr_mask.nii.gz"
    outputs.forward_xfm = fake / "forward_xfm.nii.gz"
    outputs.inverse_xfm = fake / "inverse_xfm.nii.gz"
    return outputs


class TestAnatomical:
    """Testing suite for anatomical processing."""

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""
        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        assert isinstance(args, anatomical.AnatomicalArgs)

    def test_namespace_validation(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""

    @pytest.mark.parametrize(
        ("participant", "session", "expected_count"),
        [
            ([], [], 3),  # All T1w files
            (["01"], [], 2),  # All sub-01 T1w
            ([], ["baseline"], 2),  # All ses-baseline
            (["01"], ["baseline"], 1),  # sub-01_ses-baseline only
            (["01", "02"], ["baseline"], 2),  # sub-01 & sub-02 with ses-baseline
            (["01"], ["baseline", "vis2"], 2),  # sub-01, ses-baseline & ses-vis2
            (["99"], [], 0),  # No matches
        ],
        ids=[
            "filter_by_type",
            "filter_by_participant",
            "filter_by_session",
            "filter_by_both",
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
        expected_count: int,
    ) -> None:
        """Test various filtering scenarios using parametrization."""
        base_args.participant_label = participant
        base_args.session_label = session
        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        with (
            patch("rbc.cli.anatomical.load_table", return_value=sample_dataframe),
            patch(
                "rbc.cli.anatomical.single_session_preprocess",
                return_value=_mock_anatomical_outputs(),
            ) as mock_preprocess,
            patch("rbc.cli.anatomical.PipelineContext") as mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock()
            result = anatomical.main(args)
            assert result == 0
            assert mock_preprocess.call_count == expected_count

    def test_filtering_validate_files_processed(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that correct files are processed after filtering."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        with (
            patch("rbc.cli.anatomical.load_table", return_value=sample_dataframe),
            patch(
                "rbc.cli.anatomical.single_session_preprocess",
                return_value=_mock_anatomical_outputs(),
            ) as mock_preprocess,
            patch("rbc.cli.anatomical.PipelineContext") as mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock()
            anatomical.main(args)
            assert mock_preprocess.call_count == 1
            processed_path = mock_preprocess.call_args[1]["in_t1w"]
            assert "sub-01" in str(processed_path)
            assert "baseline" in str(processed_path)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test runner environment variables are configured correctly."""
        from rbc.core import CPAC_ANTS_SEED

        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        with (
            patch("rbc.cli.anatomical.setup_runner") as mock_setup,
            patch("rbc.cli.anatomical.load_table", return_value=sample_dataframe),
            patch(
                "rbc.cli.anatomical.single_session_preprocess",
                return_value=_mock_anatomical_outputs(),
            ),
            patch("rbc.cli.anatomical.PipelineContext") as mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock()
            ctx = Mock()
            ctx.runner = Mock()
            ctx.runner.environ = {}
            ctx.logger = Mock()
            ctx.verbose = False
            mock_setup.return_value = ctx

            anatomical.main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }
