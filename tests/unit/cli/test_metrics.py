"""Unit tests for Metrics CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli import metrics
from rbc.cli.main import cli, create_parser
from rbc.cli.metrics import MetricsArgs


def _make_filtered_df(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
    task: str | None = None,
) -> pl.DataFrame:
    """Filter sample dataframe to match metrics main() filtering logic."""
    return sample_dataframe.filter(
        pl.col("datatype") == "func",
        pl.col("suffix") == "bold",
        pl.col("desc") == "preproc",
        pl.col("space") == "MNI152NLin6Asym",
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        *([pl.col("task") == task] if task else []),
    )


def _mock_metrics_outputs() -> Mock:
    """Create a mock MetricsOutputs with fake output paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.alff = fake / "alff.nii.gz"
    outputs.falff = fake / "falff.nii.gz"
    outputs.alff_smooth = fake / "alff_smooth.nii.gz"
    outputs.falff_smooth = fake / "falff_smooth.nii.gz"
    outputs.alff_zscored = fake / "alff_zscored.nii.gz"
    outputs.falff_zscored = fake / "falff_zscored.nii.gz"
    outputs.reho = fake / "reho.nii.gz"
    outputs.reho_smooth = fake / "reho_smooth.nii.gz"
    outputs.reho_zscored = fake / "reho_zscored.nii.gz"
    outputs.timeseries = fake / "timeseries.tsv"
    outputs.correlation_matrix = fake / "correlations.tsv"
    return outputs


@contextmanager
def _patch_metrics(
    filtered_df: pl.DataFrame,
    deriv_df: pl.DataFrame,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for metrics tests.

    Patches load_table to return filtered_df on first call (primary BOLD
    filtering) and deriv_df on subsequent calls (derivative lookups per
    sub/ses group). Also patches get_file_path, single_session_metrics,
    and PipelineContext.
    """
    with (
        patch(
            "rbc.cli.metrics.load_table", side_effect=[filtered_df, *([deriv_df] * 100)]
        ),
        patch(
            "rbc.cli.metrics.get_file_path",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.cli.metrics.single_session_metrics",
            return_value=_mock_metrics_outputs(),
        ) as mock_metrics,
        patch("rbc.cli.metrics.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_metrics, mock_ctx_cls


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.metrics.setup_runner") as mock:
        ctx = Mock()
        ctx.runner = Mock()
        ctx.logger = Mock()
        ctx.verbose = False
        mock.return_value = ctx
        yield mock


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base metrics argument namespace with sensible defaults."""
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
        atlas="schaefer_200",
        fwhm=6.0,
        regressor="36-parameter",
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Preprocessed derivatives dataframe simulating metrics input.

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
                "MNI152NLin6Asym",
                "MNI152NLin6Asym",
                "MNI152NLin6Asym",
                "MNI152NLin6Asym",
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


# --- Tests ---


class TestMetricsArgs:
    """Tests for MetricsArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task, fwhm, and atlas.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = MetricsArgs.validate_namespace(base_args)
        assert isinstance(args, MetricsArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for all fields are preserved through validation."""
        args = MetricsArgs.validate_namespace(base_args)
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.task is None
        assert args.regressor == "36-parameter"
        assert args.participant_label == []
        assert args.session_label == []

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = regressor
        args = MetricsArgs.validate_namespace(base_args)
        assert args.regressor == regressor

    @pytest.mark.parametrize(
        "atlas",
        ["schaefer_200", "schaefer_300", "schaefer_400", "schaefer_1000", "aal"],
    )
    def test_valid_atlases(self, base_args: argparse.Namespace, atlas: str) -> None:
        """All supported atlas options pass validation."""
        base_args.atlas = atlas
        args = MetricsArgs.validate_namespace(base_args)
        assert args.atlas == atlas

    def test_invalid_atlas_raises(self, base_args: argparse.Namespace) -> None:
        """Unsupported atlas name raises ValueError."""
        base_args.atlas = "invalid_atlas"
        with pytest.raises(ValueError, match="atlas"):
            MetricsArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("fwhm", [0.1, 1.0, 6.0, 10.0])
    def test_valid_fwhm(self, base_args: argparse.Namespace, fwhm: float) -> None:
        """Positive FWHM values pass validation."""
        base_args.fwhm = fwhm
        args = MetricsArgs.validate_namespace(base_args)
        assert args.fwhm == fwhm

    @pytest.mark.parametrize("fwhm", [0.0, -1.0, -6.0])
    def test_invalid_fwhm_raises(
        self, base_args: argparse.Namespace, fwhm: float
    ) -> None:
        """Zero or negative FWHM raises ValueError."""
        base_args.fwhm = fwhm
        with pytest.raises(ValueError, match="FWHM"):
            MetricsArgs.validate_namespace(base_args)

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
        args = MetricsArgs.validate_namespace(base_args)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels_raise(
        self, base_args: argparse.Namespace, task: str
    ) -> None:
        """Invalid task labels raise ValueError."""
        base_args.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            MetricsArgs.validate_namespace(base_args)


class TestMetricsFiltering:
    """Tests for subject/session/task/space filtering in metrics main()."""

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
        """Metrics is invoked exactly once per matching bold run after filtering."""
        base_args.participant_label = participant
        base_args.session_label = session
        base_args.task = task
        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, participant, session, task)

        with _patch_metrics(filtered_df, sample_dataframe) as (
            mock_metrics,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = metrics.main(args)
            assert result == 0
            assert mock_metrics.call_count == expected_count

    def test_space_filter_excludes_non_mni(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Non-MNI bold rows are excluded regardless of other filters."""
        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, [], [], None)

        with _patch_metrics(filtered_df, sample_dataframe) as (
            mock_metrics,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            metrics.main(args)
            assert mock_metrics.call_count == 4


class TestMetricsExports:
    """Tests for metrics output export behaviour."""

    def test_all_outputs_exported(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """All 11 metric outputs are exported per run."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_metrics(filtered_df, sample_dataframe) as (_, mock_ctx_cls):
            mock_pipe_ctx = Mock(sub="01", ses="baseline")
            mock_ctx_cls.return_value = mock_pipe_ctx
            metrics.main(args)
            assert mock_pipe_ctx.export.call_count == 11

    def test_export_uses_correct_space(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """All exports use MNI152NLin6Asym space."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_metrics(filtered_df, sample_dataframe) as (_, mock_ctx_cls):
            mock_pipe_ctx = Mock(sub="01", ses="baseline")
            mock_ctx_cls.return_value = mock_pipe_ctx
            metrics.main(args)
            for call in mock_pipe_ctx.export.call_args_list:
                assert call[1]["space"] == "MNI152NLin6Asym"

    def test_export_includes_regressor_extra(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """All exports include the regressor in the extra field."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, ["01"], ["baseline"], "rest")

        with _patch_metrics(filtered_df, sample_dataframe) as (_, mock_ctx_cls):
            mock_pipe_ctx = Mock(sub="01", ses="baseline")
            mock_ctx_cls.return_value = mock_pipe_ctx
            metrics.main(args)
            for call in mock_pipe_ctx.export.call_args_list:
                assert call[1]["extra"] == {"reg": "36-parameter"}


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Runner environment variables are set to the expected defaults."""
        from rbc.core import CPAC_ANTS_SEED

        args = MetricsArgs.validate_namespace(base_args)
        filtered_df = _make_filtered_df(sample_dataframe, [], [], None)

        with (
            patch("rbc.cli.metrics.setup_runner") as mock_setup,
            _patch_metrics(filtered_df, sample_dataframe) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            metrics.main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }


class TestMetricsRegistration:
    """Test that metrics command is registered and discoverable."""

    def test_metrics_command_registered(self) -> None:
        """Test metrics workflow is available in parser."""
        result = cli(["/input", "/output", "metrics", "--help"])
        assert result == 0

    def test_metrics_parser_has_atlas(self) -> None:
        """Test metrics subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.atlas == "schaefer_200"

    def test_metrics_parser_atlas_choices(self) -> None:
        """Test metrics subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--atlas", "aal"])
        assert args.atlas == "aal"

    def test_metrics_parser_has_fwhm(self) -> None:
        """Test metrics subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--fwhm", "8.0"])
        assert args.fwhm == 8.0

    def test_metrics_parser_has_task(self) -> None:
        """Test metrics subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--task", "rest"])
        assert args.task == "rest"

    def test_metrics_parser_has_regressor(self) -> None:
        """Test metrics subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.regressor == "36-parameter"

    def test_metrics_parser_task_default_none(self) -> None:
        """Test metrics subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.task is None

    def test_metrics_parser_fwhm_default(self) -> None:
        """Test metrics subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.fwhm == 6.0
