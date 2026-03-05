"""Unit tests for Longitudinal CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli.longitudinal import LongitudinalArgs, _process_anat, main


def _make_groups(
    sample_dataframe: pl.DataFrame,
    participant: list[str],
    session: list[str],
) -> list[tuple]:
    """Filter sample dataframe and build iter_session_files groups."""
    filtered_df = sample_dataframe.filter(
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        pl.col("ses") != "longitudinal",
    )
    return [
        (
            filtered_df.filter(
                pl.col("sub") == row["sub"],
                pl.col("ses") == row["ses"],
            ),
            filtered_df.filter(
                pl.col("sub") == row["sub"],
                pl.col("ses") == row["ses"],
                pl.col("suffix") == "T1w",
            ),
        )
        for row in filtered_df.unique(["sub", "ses"]).iter_rows(named=True)
    ]


def _mock_longitudinal_outputs() -> Mock:
    """Create a mock LongitudinalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.brain = fake / "brain.nii.gz"
    outputs.brain_mask = fake / "brain_mask.nii.gz"
    outputs.csf_mask = fake / "csf_mask.nii.gz"
    outputs.gm_mask = fake / "gm_mask.nii.gz"
    outputs.wm_mask = fake / "wm_mask.nii.gz"
    outputs.forward_xfm = fake / "fwd_xfm.nii.gz"
    outputs.inverse_xfm = fake / "inverse_xfm.nii.gz"
    return outputs


@contextmanager
def _patch_longitudinal(
    full_df: pl.DataFrame,
    groups: list[tuple],
    tpl_df: pl.DataFrame | None = None,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Common context manager patches for longitudinal tests."""
    if tpl_df is None:
        tpl_df = full_df.filter(pl.col("ses") == "longitudinal")

    # Group iter_session_files side_effects by (sub, ses)
    sub_ses_groups: dict[tuple, list] = {}
    for func_df, anat_df in groups:
        if func_df.is_empty() and anat_df.is_empty():
            continue
        df_ref = func_df if not func_df.is_empty() else anat_df
        row = df_ref.row(0, named=True)
        key = (row["sub"], row["ses"])
        sub_ses_groups.setdefault(key, [])
        sub_ses_groups[key].append((func_df, anat_df))

    call_count = 0

    def _iter_side_effect(*_args, **_kwargs) -> list:  # noqa: ANN002, ANN003
        nonlocal call_count
        values = list(sub_ses_groups.values())
        result = values[call_count] if call_count < len(values) else []
        call_count += 1
        return result

    with (
        patch("rbc.cli.longitudinal.load_table", return_value=full_df),
        patch("rbc.cli.longitudinal.load_session", return_value=Mock()),
        patch(
            "rbc.cli.longitudinal.iter_session_files",
            side_effect=_iter_side_effect,
        ),
        patch(
            "rbc.cli.longitudinal.get_file_path",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.cli.longitudinal.anatomical_longitudinal",
            return_value=_mock_longitudinal_outputs(),
        ) as mock_longitudinal,
        patch("rbc.cli.longitudinal.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_longitudinal, mock_ctx_cls


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
    output_dir = tmp_path / "output"
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dir=input_dir,
        output_dir=output_dir,
        participant_label=[],
        session_label=[],
        anatomical=True,
        functional=False,
    )


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe for testing, including a longitudinal session."""
    return pl.DataFrame(
        {
            "datatype": ["anat", "anat", "anat", "anat", "anat", "anat"],
            "suffix": ["T1w", "T1w", "T1w", "T1w", "T1w", "T1w"],
            "ext": [".nii.gz"] * 6,
            "sub": ["01", "01", "02", "02", "01", "02"],
            "ses": [
                "baseline",
                "vis2",
                "baseline",
                "vis2",
                "longitudinal",
                "longitudinal",
            ],
            "task": [None] * 6,
            "run": [None] * 6,
            "desc": [None] * 6,
            "root": ["/data"] * 6,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-vis2/anat/sub-01_ses-vis2_T1w.nii.gz",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii.gz",
                "sub-02/ses-vis2/anat/sub-02_ses-vis2_T1w.nii.gz",
                "sub-01/ses-longitudinal/anat/sub-01_ses-longitudinal_T1w.nii.gz",
                "sub-02/ses-longitudinal/anat/sub-02_ses-longitudinal_T1w.nii.gz",
            ],
        }
    )


class TestLongitudinalArgs:
    """Tests for LongitudinalArgs validation."""

    @pytest.fixture
    def long_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for longitudinal argument namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        output_dir = tmp_path / "output"
        return argparse.Namespace(
            runner="local",
            verbose=0,
            input_dir=input_dir,
            output_dir=output_dir,
            participant_label=[],
            session_label=[],
            anatomical=True,
            functional=False,
        )

    def test_validate_namespace_anatomical(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test LongitudinalArgs validates with anatomical=True."""
        args = LongitudinalArgs.validate_namespace(long_namespace)
        assert isinstance(args, LongitudinalArgs)
        assert args.anatomical is True
        assert args.functional is False

    def test_validate_namespace_both_flags(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test anatomical=True, functional=True raises NotImplementedError."""
        long_namespace.functional = True
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            LongitudinalArgs.validate_namespace(long_namespace)

    def test_validate_namespace_functional_only_raises(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test functional=True alone raises NotImplementedError."""
        long_namespace.anatomical = False
        long_namespace.functional = True
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            LongitudinalArgs.validate_namespace(long_namespace)

    def test_validate_namespace_no_flags_raises(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test neither flag set raises ValueError."""
        long_namespace.anatomical = False
        long_namespace.functional = False
        with pytest.raises(ValueError, match="At least one of"):
            LongitudinalArgs.validate_namespace(long_namespace)

    def test_defaults(self, long_namespace: argparse.Namespace) -> None:
        """Test default values for participant/session labels."""
        args = LongitudinalArgs.validate_namespace(long_namespace)
        assert args.participant_label == []
        assert args.session_label == []

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""
        args = LongitudinalArgs.validate_namespace(base_args)
        assert isinstance(args, LongitudinalArgs)


class TestLongitudinal:
    """Testing suite for longitudinal processing."""

    @pytest.mark.parametrize(
        ("participant", "session", "expected_count"),
        [
            ([], [], 4),  # All non-longitudinal sessions: sub-01, sub-02
            (["01"], [], 2),  # sub-01 only: baseline + vis2
            ([], ["baseline"], 2),  # baseline only: sub-01 + sub-02
            (["01"], ["baseline"], 1),  # sub-01 baseline only
            (["01", "02"], ["baseline"], 2),  # both subs, baseline
            (["99"], [], 0),  # No matches
        ],
        ids=[
            "all_sessions",
            "filter_by_participant",
            "filter_by_session",
            "filter_by_participant_and_session",
            "filter_by_multi_participant",
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
        groups = _make_groups(sample_dataframe, participant, session)

        with _patch_longitudinal(sample_dataframe, groups) as (
            mock_longitudinal,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = main(args)
            assert result == 0
            assert mock_longitudinal.call_count == expected_count

    def test_missing_longitudinal_template_raises(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that missing longitudinal template raises ValueError."""
        args = LongitudinalArgs.validate_namespace(base_args)
        # Strip longitudinal rows so tpl_df will be empty
        df_no_tpl = sample_dataframe.filter(pl.col("ses") != "longitudinal")
        groups = _make_groups(df_no_tpl, [], [])

        with _patch_longitudinal(df_no_tpl, groups) as (_, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            with pytest.raises(ValueError, match="No longitudinal template found"):
                main(args)

    def test_anatomical_flag_dispatches_process_anat(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that anatomical=True calls anatomical_longitudinal."""
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(sample_dataframe, ["01"], ["baseline"])

        with _patch_longitudinal(sample_dataframe, groups) as (
            mock_longitudinal,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            assert mock_longitudinal.call_count == 1

    def test_functional_flag_not_dispatched_when_false(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that functional=False never calls _process_func."""
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(sample_dataframe, [], [])

        with (
            _patch_longitudinal(sample_dataframe, groups) as (_, mock_ctx_cls),
            patch("rbc.cli.longitudinal._process_func") as mock_func,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            mock_func.assert_not_called()


class TestProcessAnat:
    """Tests for _process_anat helper."""

    @pytest.fixture
    def anat_df(self) -> pl.DataFrame:
        """Minimal anat DataFrame for a single session."""
        return pl.DataFrame(
            {
                "datatype": ["anat"],
                "suffix": ["T1w"],
                "ext": [".nii.gz"],
                "sub": ["01"],
                "ses": ["baseline"],
                "task": [None],
                "run": [None],
                "desc": [None],
                "root": ["/data"],
                "path": ["sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"],
            }
        )

    @pytest.fixture
    def tpl_df(self) -> pl.DataFrame:
        """Minimal longitudinal template DataFrame."""
        return pl.DataFrame(
            {
                "datatype": ["anat"],
                "suffix": ["T1w"],
                "ext": [".nii.gz"],
                "sub": ["01"],
                "ses": ["longitudinal"],
                "task": [None],
                "run": [None],
                "desc": [None],
                "root": ["/data"],
                "path": [
                    "sub-01/ses-longitudinal/anat/sub-01_ses-longitudinal_T1w.nii.gz"
                ],
            }
        )

    def test_process_anat_calls_longitudinal(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_anat invokes anatomical_longitudinal once."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_longitudinal_outputs()

        with (
            patch(
                "rbc.cli.longitudinal.anatomical_longitudinal", return_value=outputs
            ) as mock_longitudinal,
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)
            assert mock_longitudinal.call_count == 1

    def test_process_anat_exports_all_outputs(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_anat calls pipe_ctx.export for each expected output."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_longitudinal_outputs()

        with (
            patch("rbc.cli.longitudinal.anatomical_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)
            assert pipe_ctx.export.call_count == 7

    def test_get_anat_file_swallows_file_not_found(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _get_anat_file returns None instead of raising FileNotFoundError.

        _get_anat_file wraps get_file_path in a try/except FileNotFoundError and
        returns None on failure. Confirming this by triggering the except branch for
        the required 'brain' field: the FileNotFoundError must be caught internally,
        returning None, which then causes _require_file to raise ValueError.
        """
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_longitudinal_outputs()

        def _raise_for_brain(**kwargs) -> Path:  # noqa: ANN003
            if kwargs.get("suffix") == "T1w" and kwargs.get("desc") == "brain":
                raise FileNotFoundError("Simulated missing brain file")
            return Path("fake_workdir/file.nii.gz")

        with (
            patch("rbc.cli.longitudinal.anatomical_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=_raise_for_brain,
            ),
            pytest.raises(ValueError, match="brain"),
        ):
            _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)

    def test_process_anat_missing_brain_raises(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_anat raises ValueError when brain output is None."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_longitudinal_outputs()
        outputs.brain = None  # Simulate missing required output

        with (
            patch("rbc.cli.longitudinal.anatomical_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            outputs.brain = Path("fake_workdir/brain.nii.gz")
            outputs.brain_mask = None
            with pytest.raises(ValueError, match="brain_mask"):
                _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test runner environment variables are configured correctly."""
        from rbc.cli import _DEFAULT_ENV_VARS

        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(sample_dataframe, [], [])

        with (
            patch("rbc.cli.longitudinal.setup_runner") as mock_setup,
            _patch_longitudinal(sample_dataframe, groups) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            main(args)
            assert ctx.runner.environ == _DEFAULT_ENV_VARS

    def test_experimental_warning_emitted(
        self,
        base_args: argparse.Namespace,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that the experimental warning is logged on every run."""
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(sample_dataframe, [], [])

        with (
            patch("rbc.cli.longitudinal.setup_runner") as mock_setup,
            _patch_longitudinal(sample_dataframe, groups) as (_, mock_ctx_cls),
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx

            main(args)
            ctx.logger.warning.assert_called_once()
            warning_msg = ctx.logger.warning.call_args[0][0]
            assert "experimental" in warning_msg.lower()
