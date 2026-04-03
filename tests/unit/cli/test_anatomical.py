"""Unit tests for Anatomical CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli import anatomical


def _make_groups(
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
            {"run": row["run"]},
            filtered_df.filter(
                pl.col("sub") == row["sub"],
                pl.col("ses") == row["ses"],
            ),
        )
        for row in filtered_df.unique(["sub", "ses"]).iter_rows(named=True)
    ]
    return filtered_df, groups


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


@contextmanager
def _patch_anatomical(
    filtered_df: pl.DataFrame,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for anatomical tests."""
    from rbc.bids.session import SessionTables

    mock_anat_df = pl.DataFrame(
        {
            "suffix": ["T1w"],
            "ext": [".nii.gz"],
            "run": [None],
            "acq": [None],
            "part": [None],
            "echo": [None],
            "ce": [None],
            "rec": [None],
            "inv": [None],
            "space": [None],
            "desc": [None],
            "root": ["/data"],
            "path": ["sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"],
        }
    )
    mock_session = SessionTables(anat=mock_anat_df, func=None)
    with (
        patch("rbc.orchestration.anatomical.load_table", return_value=filtered_df),
        patch("rbc.orchestration.anatomical.load_session", return_value=mock_session),
        patch(
            "rbc.orchestration.anatomical.single_session_preprocess",
            return_value=_mock_anatomical_outputs(),
        ) as mock_preprocess,
        patch("rbc.orchestration.anatomical.RunContext") as mock_ctx_cls,
    ):
        yield mock_preprocess, mock_ctx_cls


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking init_runner so no real runner is created."""
    with patch("rbc.orchestration.anatomical.init_runner") as mock:
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
        tmp_dir=None,
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
            "run": [None] * 6,
            "space": [None] * 6,
            "desc": [None] * 6,
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
            ([], [], 3),
            (["01"], [], 2),
            ([], ["baseline"], 2),
            (["01"], ["baseline"], 1),
            (["01", "02"], ["baseline"], 2),
            (["01"], ["baseline", "vis2"], 2),
            (["99"], [], 0),
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
        filtered_df, _ = _make_groups(sample_dataframe, participant, session)

        with _patch_anatomical(filtered_df) as (mock_preprocess, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
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
        participant, session = ["01"], ["baseline"]
        base_args.participant_label = participant
        base_args.session_label = session
        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        filtered_df, _ = _make_groups(sample_dataframe, participant, session)

        with _patch_anatomical(filtered_df) as (mock_preprocess, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            anatomical.main(args)
            assert mock_preprocess.call_count == 1
            processed_path = mock_preprocess.call_args[1]["in_t1w"]
            assert "sub-01" in str(processed_path)
            assert "baseline" in str(processed_path)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_init_runner_called_with_config(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test init_runner is called with the correct RunnerConfig."""
        from rbc.orchestration import RunnerConfig

        args = anatomical.AnatomicalArgs.validate_namespace(base_args)
        filtered_df, _ = _make_groups(sample_dataframe, [], [])

        with (
            patch("rbc.orchestration.anatomical.init_runner") as mock_init,
            _patch_anatomical(filtered_df) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            anatomical.main(args)
            mock_init.assert_called_once()
            config = mock_init.call_args[0][0]
            assert isinstance(config, RunnerConfig)
