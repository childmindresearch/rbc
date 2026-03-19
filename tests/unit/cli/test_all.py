"""Unit tests for All (combined pipeline) CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli import all as all_cli
from rbc.cli.all import AllArgs
from rbc.cli.main import cli, create_parser


def _make_filtered_df(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
    task: str | None = None,
) -> pl.DataFrame:
    """Filter sample dataframe to match all main() filtering logic."""
    return sample_dataframe.filter(
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        *([pl.col("task") == task] if task else []),
    )


def _mock_anat_outputs() -> Mock:
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


def _mock_func_outputs() -> Mock:
    """Create a mock FunctionalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.sbref = fake / "sbref.nii.gz"
    outputs.motion_corrected_bold = fake / "bold_preproc.nii.gz"
    outputs.motion_params = fake / "motion.1D"
    outputs.rms_rel = fake / "rms_rel.rms"
    outputs.rms_abs = fake / "rms_abs.rms"
    outputs.bold_mask = fake / "bold_mask.nii.gz"
    outputs.bold_to_anat_matrix = fake / "bold_to_anat.txt"
    outputs.regressor_file = fake / "regressors.1D"
    outputs.template_bold = fake / "template_bold.nii.gz"
    outputs.cleaned_bold = fake / "cleaned_bold.nii.gz"
    outputs.template_brain_mask = fake / "template_brain_mask.nii.gz"
    return outputs


def _mock_metrics_outputs() -> Mock:
    """Create a mock MetricsOutputs with fake paths."""
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


def _mock_qc_outputs(*, passed: bool = True) -> Mock:
    """Create a mock QCOutputs with a fake qc_file path and pass/fail status."""
    outputs = Mock()
    outputs.qc_file = Path("fake_workdir") / "qc.tsv"
    outputs.passed = passed
    return outputs


@contextmanager
def _patch_all(
    filtered_df: pl.DataFrame,
    groups: list[list[str]],
    *,
    qc_passed: bool = True,
) -> Generator[tuple[Mock, Mock, Mock, Mock, Mock], None, None]:
    """Common context manager patches for all-pipeline tests.

    Yields (mock_anat, mock_func, mock_metrics, mock_qc, mock_ctx_cls).
    """
    from rbc.cli.query import SessionTables

    mock_anat_df = pl.DataFrame(
        {
            "suffix": ["T1w"],
            "ext": [".nii.gz"],
            "run": [None],
            "root": ["/data"],
            "path": ["sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"],
        }
    )
    mock_session = SessionTables(anat=mock_anat_df, func=None)

    with (
        patch("rbc.cli.all.load_table", return_value=filtered_df),
        patch("rbc.cli.all.load_session", return_value=mock_session),
        patch("rbc.cli.all.iter_session_files", side_effect=groups),
        patch(
            "rbc.cli.all.anatomical_preprocess", return_value=_mock_anat_outputs()
        ) as mock_anat,
        patch(
            "rbc.cli.all.functional_preprocess", return_value=_mock_func_outputs()
        ) as mock_func,
        patch(
            "rbc.cli.all.metrics_pipeline", return_value=_mock_metrics_outputs()
        ) as mock_metrics,
        patch(
            "rbc.cli.all.qc_pipeline", return_value=_mock_qc_outputs(passed=qc_passed)
        ) as mock_qc,
        patch("rbc.cli.all.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_anat, mock_func, mock_metrics, mock_qc, mock_ctx_cls


def _make_groups(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
    task: str | None = None,
) -> tuple[pl.DataFrame, list[list[str]]]:
    """Filter sample dataframe and build iter_session_files groups."""
    filtered_df = sample_dataframe.filter(
        pl.col("suffix") == "bold",
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        *([pl.col("task") == task] if task else []),
    )
    # Build sub_ses_groups matching how main() calls iter_session_files
    sub_ses_groups: dict[tuple, list] = {}
    for row in filtered_df.unique(["sub", "ses", "task"]).iter_rows(named=True):
        func_group = filtered_df.filter(
            pl.col("sub") == row["sub"],
            pl.col("ses") == row["ses"],
            pl.col("task") == row["task"],
        )
        key = (row["sub"], row["ses"])
        sub_ses_groups.setdefault(key, [])
        sub_ses_groups[key].append((func_group, pl.DataFrame()))

    full_df = _make_filtered_df(sample_dataframe, participant, session, task)
    return full_df, list(sub_ses_groups.values())


@pytest.fixture
def mock_setup() -> Generator[Mock, None, None]:
    """Fixture for mocking setup_runner with consistent return value."""
    with patch("rbc.cli.all.setup_runner") as mock:
        ctx = Mock()
        ctx.runner = Mock()
        ctx.logger = Mock()
        ctx.verbose = False
        mock.return_value = ctx
        yield mock


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base all-pipeline argument namespace with sensible defaults."""
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
        atlas="schaefer_200",
        fwhm=6.0,
        start_tr=2,
        tmp_dir=None,
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """BIDS-like dataframe spanning multiple subjects, sessions, and tasks.

    Contains:
    - sub-01/ses-baseline: anat (T1w) + func (task-rest, task-nback)
    - sub-02/ses-baseline: anat (T1w) + func (task-rest)
    - sub-01/ses-vis2: anat (T1w) + func (task-rest)
    """
    return pl.DataFrame(
        {
            "datatype": ["anat", "func", "func", "anat", "func", "anat", "func"],
            "suffix": ["T1w", "bold", "bold", "T1w", "bold", "T1w", "bold"],
            "ext": [".nii.gz"] * 7,
            "sub": ["01", "01", "01", "02", "02", "01", "01"],
            "ses": [
                "baseline",
                "baseline",
                "baseline",
                "baseline",
                "baseline",
                "vis2",
                "vis2",
            ],
            "task": [None, "rest", "nback", None, "rest", None, "rest"],
            "run": [None] * 7,
            "acq": [None] * 7,
            "dir": [None] * 7,
            "echo": [None] * 7,
            "part": [None] * 7,
            "rec": [None] * 7,
            "root": ["/data"] * 7,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-nback_bold.nii.gz",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii.gz",
                "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-vis2/anat/sub-01_ses-vis2_T1w.nii.gz",
                "sub-01/ses-vis2/func/sub-01_ses-vis2_task-rest_bold.nii.gz",
            ],
        }
    )


class TestAllArgs:
    """Tests for AllArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task, fwhm,
    atlas, and start_tr.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = AllArgs.validate_namespace(base_args)
        assert isinstance(args, AllArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for all fields are preserved through validation."""
        args = AllArgs.validate_namespace(base_args)
        assert args.regressor == "36-parameter"
        assert args.task is None
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.start_tr == 2
        assert args.participant_label == []
        assert args.session_label == []

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = regressor
        args = AllArgs.validate_namespace(base_args)
        assert args.regressor == regressor

    @pytest.mark.parametrize(
        "atlas",
        ["schaefer_200", "schaefer_300", "schaefer_400", "schaefer_1000", "aal"],
    )
    def test_valid_atlases(self, base_args: argparse.Namespace, atlas: str) -> None:
        """All supported atlas options pass validation."""
        base_args.atlas = atlas
        args = AllArgs.validate_namespace(base_args)
        assert args.atlas == atlas

    def test_invalid_atlas_raises(self, base_args: argparse.Namespace) -> None:
        """Unsupported atlas name raises ValueError."""
        base_args.atlas = "invalid_atlas"
        with pytest.raises(ValueError, match="atlas"):
            AllArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("fwhm", [0.1, 1.0, 6.0, 10.0])
    def test_valid_fwhm(self, base_args: argparse.Namespace, fwhm: float) -> None:
        """Positive FWHM values pass validation."""
        base_args.fwhm = fwhm
        args = AllArgs.validate_namespace(base_args)
        assert args.fwhm == fwhm

    @pytest.mark.parametrize("fwhm", [0.0, -1.0, -6.0])
    def test_invalid_fwhm_raises(
        self, base_args: argparse.Namespace, fwhm: float
    ) -> None:
        """Zero or negative FWHM raises ValueError."""
        base_args.fwhm = fwhm
        with pytest.raises(ValueError, match="FWHM"):
            AllArgs.validate_namespace(base_args)

    def test_custom_start_tr(self, base_args: argparse.Namespace) -> None:
        """Custom start_tr is correctly preserved through validation."""
        base_args.start_tr = 5
        args = AllArgs.validate_namespace(base_args)
        assert args.start_tr == 5

    @pytest.mark.parametrize("start_tr", [0, -1, -5])
    def test_invalid_start_tr_raises(
        self, base_args: argparse.Namespace, start_tr: int
    ) -> None:
        """Zero or negative start_tr raises ValueError."""
        base_args.start_tr = start_tr
        with pytest.raises(ValueError, match="Start TR"):
            AllArgs.validate_namespace(base_args)

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
        args = AllArgs.validate_namespace(base_args)
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
            AllArgs.validate_namespace(base_args)


class TestAllPipeline:
    """Tests for the combined pipeline execution in all main()."""

    @pytest.mark.parametrize(
        ("participant", "session", "task", "expected_func_count"),
        [
            ([], [], None, 4),  # All bold runs
            (["01"], [], None, 3),  # All sub-01 bold runs
            ([], ["baseline"], None, 3),  # All ses-baseline bold runs
            (["01"], ["baseline"], None, 2),  # sub-01_ses-baseline only
            (["01"], ["baseline"], "rest", 1),  # sub-01_ses-baseline_task-rest only
            (["01", "02"], ["baseline"], None, 3),  # sub-01 & sub-02 ses-baseline
            (["01"], ["baseline", "vis2"], None, 3),  # sub-01, both sessions
            (["99"], [], None, 0),  # No matches
        ],
        ids=[
            "all_bold",
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
        expected_func_count: int,
    ) -> None:
        """Functional, metrics, and QC are each invoked once per matching bold run."""
        base_args.participant_label = participant
        base_args.session_label = session
        base_args.task = task
        args = AllArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, participant, session, task)

        with _patch_all(filtered_df, groups) as (
            _,
            mock_func,
            mock_metrics,
            mock_qc,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = all_cli.main(args)
            assert result == 0
            assert mock_func.call_count == expected_func_count
            assert mock_metrics.call_count == expected_func_count
            assert mock_qc.call_count == expected_func_count

    def test_anatomical_runs_once_per_session(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Anatomical preprocessing runs exactly once per sub/ses group."""
        args = AllArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, [], [], None)

        with _patch_all(filtered_df, groups) as (mock_anat, _, _, _, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            all_cli.main(args)
            # 3 unique sub/ses combinations in sample_dataframe
            assert mock_anat.call_count == 3

    def test_qc_passed_logged(
        self,
        mock_setup: Mock,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """PASSED is logged when QC outputs indicate the run passed."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = AllArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(
            sample_dataframe, ["01"], ["baseline"], "rest"
        )

        with _patch_all(filtered_df, groups, qc_passed=True) as (
            _,
            _,
            _,
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            all_cli.main(args)
            log_calls = [
                str(c) for c in mock_setup.return_value.logger.info.call_args_list
            ]
            assert any("PASSED" in c for c in log_calls)

    def test_qc_failed_logged(
        self,
        mock_setup: Mock,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """FAILED is logged when QC outputs indicate the run failed."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        base_args.task = "rest"
        args = AllArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(
            sample_dataframe, ["01"], ["baseline"], "rest"
        )

        with _patch_all(filtered_df, groups, qc_passed=False) as (
            _,
            _,
            _,
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            all_cli.main(args)
            log_calls = [
                str(c) for c in mock_setup.return_value.logger.info.call_args_list
            ]
            assert any("FAILED" in c for c in log_calls)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
    ) -> None:
        """Runner environment variables are set to the expected defaults."""
        from rbc.core import CPAC_ANTS_SEED

        args = AllArgs.validate_namespace(base_args)
        filtered_df, groups = _make_groups(sample_dataframe, [], [], None)

        with (
            patch("rbc.cli.all.setup_runner") as mock_setup,
            _patch_all(filtered_df, groups) as (_, _, _, _, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            all_cli.main(args)
            assert ctx.runner.environ == {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
                "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
            }


class TestAllRegistration:
    """Test that all command is registered and discoverable."""

    def test_all_command_registered(self) -> None:
        """Test all workflow is available in parser."""
        result = cli(["/input", "/output", "all", "--help"])
        assert result == 0

    def test_all_parser_has_regressor(self) -> None:
        """Test all subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.regressor == "36-parameter"

    def test_all_parser_has_task(self) -> None:
        """Test all subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--task", "rest"])
        assert args.task == "rest"

    def test_all_parser_has_atlas(self) -> None:
        """Test all subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.atlas == "schaefer_200"

    def test_all_parser_atlas_choices(self) -> None:
        """Test all subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--atlas", "aal"])
        assert args.atlas == "aal"

    def test_all_parser_has_fwhm(self) -> None:
        """Test all subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--fwhm", "8.0"])
        assert args.fwhm == 8.0

    def test_all_parser_has_start_tr(self) -> None:
        """Test all subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_all_parser_task_default_none(self) -> None:
        """Test all subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.task is None

    def test_all_parser_fwhm_default(self) -> None:
        """Test all subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.fwhm == 6.0

    def test_all_parser_start_tr_default(self) -> None:
        """Test all subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.start_tr == 2
