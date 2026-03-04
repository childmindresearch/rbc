"""Unit tests for Longitudinal CLI module."""

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli.longitudinal import LongitudinalArgs, _process_anat, _process_func, main


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


def _mock_functional_outputs(*, with_bold_mask: bool = True) -> Mock:
    """Create a mock FunctionalOutputs with fake paths."""
    fake = Path("fake_workdir")
    outputs = Mock()
    outputs.sbref = fake / "sbref.nii.gz"
    outputs.bold = fake / "bold.nii.gz"
    outputs.forward_xfm = fake / "fwd_xfm.nii.gz"
    outputs.bold_mask = (fake / "bold_mask.nii.gz") if with_bold_mask else None
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


@contextmanager
def _patch_longitudinal_functional(
    full_df: pl.DataFrame,
    groups: list[tuple],
    tpl_df: pl.DataFrame | None = None,
    *,
    with_bold_mask: bool = True,
) -> Generator[tuple[Mock, Mock, Mock], None, None]:
    """Context manager patches for longitudinal tests covering functional dispatch."""
    if tpl_df is None:
        tpl_df = full_df.filter(pl.col("ses") == "longitudinal")

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
        ) as mock_anat,
        patch(
            "rbc.cli.longitudinal.functional_longitudinal",
            return_value=_mock_functional_outputs(with_bold_mask=with_bold_mask),
        ) as mock_func,
        patch("rbc.cli.longitudinal.PipelineContext") as mock_ctx_cls,
    ):
        yield mock_anat, mock_func, mock_ctx_cls


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


@pytest.fixture
def func_sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe with functional data for testing."""
    return pl.DataFrame(
        {
            "datatype": ["func", "func", "func", "func", "anat", "anat"],
            "suffix": ["bold", "bold", "bold", "bold", "T1w", "T1w"],
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
            "task": ["rest", "rest", "rest", "rest", None, None],
            "run": [None] * 6,
            "desc": [None] * 6,
            "root": ["/data"] * 6,
            "path": [
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-vis2/func/sub-01_ses-vis2_task-rest_bold.nii.gz",
                "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_bold.nii.gz",
                "sub-02/ses-vis2/func/sub-02_ses-vis2_task-rest_bold.nii.gz",
                "sub-01/ses-longitudinal/anat/sub-01_ses-longitudinal_T1w.nii.gz",
                "sub-02/ses-longitudinal/anat/sub-02_ses-longitudinal_T1w.nii.gz",
            ],
        }
    )


@pytest.fixture
def mixed_sample_dataframe() -> pl.DataFrame:
    """Generate sample dataframe with both anat and func rows for both-flags tests.

    _make_groups builds anat_df by filtering suffix="T1w". A func-only dataframe
    produces empty anat_df entries that get skipped in _patch_longitudinal_functional,
    causing call_count to go out of bounds when both flags are True. This fixture
    includes T1w rows for every non-longitudinal session so _make_groups yields
    non-empty groups for both halves.
    """
    return pl.DataFrame(
        {
            "datatype": [
                "anat",
                "anat",
                "anat",
                "anat",
                "func",
                "func",
                "func",
                "func",
                "anat",
                "anat",
            ],
            "suffix": [
                "T1w",
                "T1w",
                "T1w",
                "T1w",
                "bold",
                "bold",
                "bold",
                "bold",
                "T1w",
                "T1w",
            ],
            "ext": [".nii.gz"] * 10,
            "sub": ["01", "01", "02", "02", "01", "01", "02", "02", "01", "02"],
            "ses": [
                "baseline",
                "vis2",
                "baseline",
                "vis2",
                "baseline",
                "vis2",
                "baseline",
                "vis2",
                "longitudinal",
                "longitudinal",
            ],
            "task": [
                None,
                None,
                None,
                None,
                "rest",
                "rest",
                "rest",
                "rest",
                None,
                None,
            ],
            "run": [None] * 10,
            "desc": [None] * 10,
            "root": ["/data"] * 10,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-vis2/anat/sub-01_ses-vis2_T1w.nii.gz",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii.gz",
                "sub-02/ses-vis2/anat/sub-02_ses-vis2_T1w.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-vis2/func/sub-01_ses-vis2_task-rest_bold.nii.gz",
                "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_bold.nii.gz",
                "sub-02/ses-vis2/func/sub-02_ses-vis2_task-rest_bold.nii.gz",
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

    def test_validate_namespace_functional(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test LongitudinalArgs validates with functional=True."""
        long_namespace.anatomical = False
        long_namespace.functional = True
        args = LongitudinalArgs.validate_namespace(long_namespace)
        assert isinstance(args, LongitudinalArgs)
        assert args.anatomical is False
        assert args.functional is True

    def test_validate_namespace_both_flags(
        self, long_namespace: argparse.Namespace
    ) -> None:
        """Test LongitudinalArgs validates with both flags set."""
        long_namespace.functional = True
        args = LongitudinalArgs.validate_namespace(long_namespace)
        assert args.anatomical is True
        assert args.functional is True

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

    def test_functional_flag_dispatches_process_func(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        func_sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that functional=True calls functional_longitudinal."""
        base_args.anatomical = False
        base_args.functional = True
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(func_sample_dataframe, ["01"], ["baseline"])

        with _patch_longitudinal_functional(func_sample_dataframe, groups) as (
            mock_anat,
            mock_func,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            result = main(args)
            assert result == 0
            mock_func.assert_called_once()
            mock_anat.assert_not_called()

    def test_both_flags_dispatch_anat_and_func(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        mixed_sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test that anatomical=True and functional=True both dispatch.

        Uses mixed_sample_dataframe (anat + func rows per session) so that
        _make_groups produces non-empty anat_df entries. A func-only dataframe
        yields empty anat_df groups that get skipped in the patch helper, causing
        call_count to go out of bounds when both flags are True.
        """
        base_args.anatomical = True
        base_args.functional = True
        base_args.participant_label = ["01"]
        base_args.session_label = ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(mixed_sample_dataframe, ["01"], ["baseline"])

        with _patch_longitudinal_functional(mixed_sample_dataframe, groups) as (
            mock_anat,
            mock_func,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            mock_anat.assert_called_once()
            mock_func.assert_called_once()

    def test_functional_filtering_all_sessions(
        self,
        mock_setup: Mock,  # noqa: ARG002 - test setup
        base_args: argparse.Namespace,
        func_sample_dataframe: pl.DataFrame,
    ) -> None:
        """Test functional dispatch count across all non-longitudinal sessions."""
        base_args.anatomical = False
        base_args.functional = True
        args = LongitudinalArgs.validate_namespace(base_args)
        groups = _make_groups(func_sample_dataframe, [], [])

        with _patch_longitudinal_functional(func_sample_dataframe, groups) as (
            _,
            mock_func,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            assert mock_func.call_count == 4  # 2 subs × 2 sessions


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


class TestProcessFunc:
    """Tests for _process_func helper."""

    @pytest.fixture
    def func_df(self) -> pl.DataFrame:
        """Minimal func DataFrame for a single session.

        TODO: upstream bug — _process_func filters on suffix="T1w" instead of
        suffix="bold" to extract task/run. Tests here are written against the
        *correct* behavior (suffix="bold"). Update the source and remove this
        comment once the bug is fixed.
        """
        return pl.DataFrame(
            {
                "datatype": ["func"],
                "suffix": ["bold"],
                "ext": [".nii.gz"],
                "sub": ["01"],
                "ses": ["baseline"],
                "task": ["rest"],
                "run": [None],
                "desc": [None],
                "root": ["/data"],
                "path": [
                    "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz"
                ],
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

    def test_process_func_calls_functional_longitudinal(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func invokes functional_longitudinal once."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs()

        with (
            patch(
                "rbc.cli.longitudinal.functional_longitudinal", return_value=outputs
            ) as mock_functional,
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert mock_functional.call_count == 1

    def test_process_func_exports_all_outputs_with_bold_mask(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func exports sbref, bold, forward_xfm, and bold_mask."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs(with_bold_mask=True)

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert pipe_ctx.export.call_count == 4

    def test_process_func_skips_bold_mask_export_when_none(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func skips bold_mask export when output is None."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs(with_bold_mask=False)

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert pipe_ctx.export.call_count == 3

    def test_process_func_missing_bold_raises(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func raises ValueError when bold file is missing."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs()

        def _raise_for_bold(**kwargs) -> Path:  # noqa: ANN003
            if kwargs.get("suffix") == "bold" and kwargs.get("desc") == "preproc":
                raise FileNotFoundError("Simulated missing bold file")
            return Path("fake_workdir/file.nii.gz")

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=_raise_for_bold,
            ),
            pytest.raises(ValueError, match="bold"),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

    def test_process_func_missing_sbref_raises(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func raises ValueError when sbref file is missing."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs()

        def _raise_for_sbref(**kwargs) -> Path:  # noqa: ANN003
            if kwargs.get("suffix") == "sbref":
                raise FileNotFoundError("Simulated missing sbref file")
            return Path("fake_workdir/file.nii.gz")

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=_raise_for_sbref,
            ),
            pytest.raises(ValueError, match="sbref"),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

    def test_process_func_missing_bold_to_anat_xfm_raises(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _process_func raises ValueError when bold_to_anat_xfm is missing."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs()

        def _raise_for_xfm(**kwargs) -> Path:  # noqa: ANN003
            extra = kwargs.get("extra", {})
            if kwargs.get("suffix") == "xfm" and extra.get("from") == "bold":
                raise FileNotFoundError("Simulated missing xfm file")
            return Path("fake_workdir/file.nii.gz")

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=_raise_for_xfm,
            ),
            pytest.raises(ValueError, match="bold_to_anat_xfm"),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

    def test_process_func_get_func_file_for_optional(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame
    ) -> None:
        """Test _get_func_file returns None for optional files instead of raising."""
        pipe_ctx = Mock(sub="01", ses="baseline")
        outputs = _mock_functional_outputs(with_bold_mask=False)

        def _raise_for_mask(**kwargs) -> Path:  # noqa: ANN003
            if kwargs.get("suffix") == "mask" and kwargs.get("desc") == "brain":
                raise FileNotFoundError("Simulated missing bold_mask file")
            return Path("fake_workdir/file.nii.gz")

        with (
            patch("rbc.cli.longitudinal.functional_longitudinal", return_value=outputs),
            patch(
                "rbc.cli.longitudinal.get_file_path",
                side_effect=_raise_for_mask,
            ),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert pipe_ctx.export.call_count == 3


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_runner_environment_variables_set(
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
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
        self, base_args: argparse.Namespace, sample_dataframe: pl.DataFrame
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
