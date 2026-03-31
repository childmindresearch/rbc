"""Unit tests for Longitudinal CLI module."""

import argparse
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from rbc.cli.longitudinal import LongitudinalArgs, _process_anat, _process_func, main
from rbc.context import PipelineContext

_SCHEMA = [
    "datatype",
    "suffix",
    "ext",
    "sub",
    "ses",
    "space",
    "task",
    "run",
    "desc",
    "root",
    "path",
]


def _df(*rows: tuple) -> pl.DataFrame:
    """Build a DataFrame from row tuples using the shared BIDS schema."""
    return pl.DataFrame(dict(zip(_SCHEMA, zip(*rows, strict=True), strict=True)))


def _anat_row(sub: str, ses: str, *, suffix: str = "T1w") -> tuple:
    path = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_{suffix}.nii.gz"
    return ("anat", suffix, ".nii.gz", sub, ses, None, None, None, None, "/data", path)


def _func_row(sub: str, ses: str, task: str = "rest") -> tuple:
    path = f"sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_bold.nii.gz"
    return ("func", "bold", ".nii.gz", sub, ses, None, task, None, None, "/data", path)


def _mock_anat_outputs() -> Mock:
    fake = Path("fake_workdir")
    m = Mock()
    m.brain = fake / "brain.nii.gz"
    m.brain_mask = fake / "brain_mask.nii.gz"
    m.csf_mask = fake / "csf_mask.nii.gz"
    m.gm_mask = fake / "gm_mask.nii.gz"
    m.wm_mask = fake / "wm_mask.nii.gz"
    m.forward_xfm = fake / "fwd_xfm.nii.gz"
    m.inverse_xfm = fake / "inverse_xfm.nii.gz"
    return m


def _mock_func_outputs(*, with_bold_mask: bool = True) -> Mock:
    fake = Path("fake_workdir")
    m = Mock()
    m.sbref = fake / "sbref.nii.gz"
    m.bold = fake / "bold.nii.gz"
    m.forward_xfm = fake / "fwd_xfm.nii.gz"
    m.bold_mask = (fake / "bold_mask.nii.gz") if with_bold_mask else None
    return m


def _none_for(**match: str) -> Callable[..., Path | None]:
    """Return side-effect that returns None for matched kwargs.

    Checks both top-level kwargs and the ``entities`` dict.
    """

    def _side_effect(*_args: object, **kwargs: object) -> Path | None:
        merged = {**kwargs, **(kwargs.get("entities") or {})}  # type: ignore[dict-item]
        if all(merged.get(k) == v for k, v in match.items()):
            return None
        return Path("fake_workdir/file.nii.gz")

    return _side_effect


def _make_groups(
    df: pl.DataFrame, participant: list[str], session: list[str]
) -> list[tuple]:
    filtered = df.filter(
        *([pl.col("sub").is_in(participant)] if participant else []),
        *([pl.col("ses").is_in(session)] if session else []),
        pl.col("ses") != "longitudinal",
    )
    return [
        (
            filtered.filter(pl.col("sub") == r["sub"], pl.col("ses") == r["ses"]),
            filtered.filter(
                pl.col("sub") == r["sub"],
                pl.col("ses") == r["ses"],
                pl.col("suffix") == "T1w",
            ),
        )
        for r in filtered.unique(["sub", "ses"]).iter_rows(named=True)
    ]


def _build_iter_side_effect(groups: list[tuple]) -> Callable[..., list]:
    sub_ses_groups: dict[tuple, list] = {}
    for func_df, anat_df in groups:
        if func_df.is_empty() and anat_df.is_empty():
            continue
        ref = func_df if not func_df.is_empty() else anat_df
        key = (ref["sub"][0], ref["ses"][0])
        sub_ses_groups.setdefault(key, []).append((func_df, anat_df))

    call_count = 0

    def _side_effect(*_args, **_kwargs) -> list:  # noqa: ANN002, ANN003
        nonlocal call_count
        values = list(sub_ses_groups.values())
        result = values[call_count] if call_count < len(values) else []
        call_count += 1
        return result

    return _side_effect


@contextmanager
def _patch_main(
    full_df: pl.DataFrame,
    groups: list[tuple],
    *,
    with_bold_mask: bool = True,
) -> Generator[tuple[Mock, Mock, Mock], None, None]:
    """Patch all external calls made by main()."""
    from rbc.cli.query import SessionTables

    mock_anat_df = pl.DataFrame(
        {
            "suffix": ["T1w"],
            "ext": [".nii.gz"],
            "run": [None],
            "acq": [None],
            "space": [None],
            "desc": [None],
            "root": ["/data"],
            "path": ["sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"],
        }
    )
    mock_session = SessionTables(anat=mock_anat_df, func=None)
    with (
        patch("rbc.cli.longitudinal.load_table", return_value=full_df),
        patch("rbc.cli.longitudinal.load_session", return_value=mock_session),
        patch(
            "rbc.cli.longitudinal.iter_session_files",
            side_effect=_build_iter_side_effect(groups),
        ),
        patch(
            "rbc.core.bids2table.find_file",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.cli.longitudinal.anatomical_longitudinal",
            return_value=_mock_anat_outputs(),
        ) as mock_anat,
        patch(
            "rbc.cli.longitudinal.functional_longitudinal",
            return_value=_mock_func_outputs(with_bold_mask=with_bold_mask),
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
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        participant_label=[],
        session_label=[],
        anatomical=True,
        functional=False,
        tmp_dir=None,
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
            "space": [None] * 6,
            "run": [None] * 6,
            "acq": [None] * 6,
            "dir": [None] * 6,
            "echo": [None] * 6,
            "part": [None] * 6,
            "rec": [None] * 6,
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
def anat_df() -> pl.DataFrame:
    """Fixture for anatomical dataframe."""
    return _df(_anat_row("01", "baseline"))


@pytest.fixture
def func_df() -> pl.DataFrame:
    """Fixture for functional dataframe."""
    return _df(_func_row("01", "baseline"))


@pytest.fixture
def tpl_df() -> pl.DataFrame:
    """Fixture for template dataframe."""
    return _df(_anat_row("01", "longitudinal"))


@pytest.fixture
def anat_df_full() -> pl.DataFrame:
    """Anat-only dataframe."""
    return _df(
        _anat_row("01", "baseline"),
        _anat_row("01", "vis2"),
        _anat_row("02", "baseline"),
        _anat_row("02", "vis2"),
        _anat_row("01", "longitudinal"),
        _anat_row("02", "longitudinal"),
    )


@pytest.fixture
def func_df_full() -> pl.DataFrame:
    """Func-only dataframe."""
    return _df(
        _func_row("01", "baseline"),
        _func_row("01", "vis2"),
        _func_row("02", "baseline"),
        _func_row("02", "vis2"),
        _anat_row("01", "longitudinal"),
        _anat_row("02", "longitudinal"),
    )


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """Anat + func dataframe for both-flags tests.

    _make_groups builds anat_df by filtering suffix="T1w". A func-only dataframe
    produces empty anat groups that get skipped, causing call_count to go out of
    bounds when both flags are True. Including T1w rows for every non-longitudinal
    session avoids this.
    """
    return _df(
        _anat_row("01", "baseline"),
        _anat_row("01", "vis2"),
        _anat_row("02", "baseline"),
        _anat_row("02", "vis2"),
        _func_row("01", "baseline"),
        _func_row("01", "vis2"),
        _func_row("02", "baseline"),
        _func_row("02", "vis2"),
        _anat_row("01", "longitudinal"),
        _anat_row("02", "longitudinal"),
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
            tmp_dir=None,
        )

    @pytest.mark.parametrize(
        ("anat", "func"), [(True, False), (False, True), (True, True)]
    )
    def test_valid_flag_combinations(
        self,
        base_args: argparse.Namespace,
        anat: bool,  # noqa: FBT001
        func: bool,  # noqa: FBT001
    ) -> None:
        """Test different combination of valid longitudinal flags."""
        base_args.anatomical, base_args.functional = anat, func
        args = LongitudinalArgs.validate_namespace(base_args)
        assert args.anatomical is anat
        assert args.functional is func

    def test_no_flags_raises(self, base_args: argparse.Namespace) -> None:
        """Test error raised if no processing selected."""
        base_args.anatomical = base_args.functional = False
        with pytest.raises(ValueError, match="At least one of"):
            LongitudinalArgs.validate_namespace(base_args)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Test defaults."""
        args = LongitudinalArgs.validate_namespace(base_args)
        assert args.participant_label == []
        assert args.session_label == []


class TestLongitudinalMain:
    """Integration tests for main() dispatch and filtering."""

    @pytest.mark.parametrize(
        ("participant", "session", "expected"),
        [
            ([], [], 4),
            (["01"], [], 2),
            ([], ["baseline"], 2),
            (["01"], ["baseline"], 1),
            (["01", "02"], ["baseline"], 2),
            (["99"], [], 0),
        ],
        ids=["all", "sub_filter", "ses_filter", "sub_and_ses", "multi_sub", "no_match"],
    )
    def test_anat_filtering(
        self,
        mock_setup: Mock,  # noqa: ARG002
        base_args: argparse.Namespace,
        anat_df_full: pl.DataFrame,
        participant: list[str],
        session: list[str],
        expected: int,
    ) -> None:
        """Test filtering anat df."""
        base_args.participant_label, base_args.session_label = participant, session
        args = LongitudinalArgs.validate_namespace(base_args)
        with _patch_main(
            anat_df_full, _make_groups(anat_df_full, participant, session)
        ) as (
            mock_anat,
            _,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            assert main(args) == 0
            assert mock_anat.call_count == expected

    def test_missing_template_raises(
        self,
        mock_setup: Mock,  # noqa: ARG002
        base_args: argparse.Namespace,
        anat_df_full: pl.DataFrame,
    ) -> None:
        """Test missing template raises error."""
        df = anat_df_full.filter(pl.col("ses") != "longitudinal")
        args = LongitudinalArgs.validate_namespace(base_args)
        with _patch_main(df, _make_groups(df, [], [])) as (_, __, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            with pytest.raises(ValueError, match="No longitudinal template found"):
                main(args)

    def test_functional_false_never_calls_process_func(
        self,
        mock_setup: Mock,  # noqa: ARG002
        base_args: argparse.Namespace,
        anat_df_full: pl.DataFrame,
    ) -> None:
        """Test functional not called if only anat selected."""
        args = LongitudinalArgs.validate_namespace(base_args)
        with (
            _patch_main(anat_df_full, _make_groups(anat_df_full, [], [])) as (
                _,
                __,
                mock_ctx_cls,
            ),
            patch("rbc.cli.longitudinal._process_func") as mock_func,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            mock_func.assert_not_called()

    def test_functional_true_dispatches_process_func(
        self,
        mock_setup: Mock,  # noqa: ARG002
        base_args: argparse.Namespace,
        func_df_full: pl.DataFrame,
    ) -> None:
        """Test functional processing works without anatomical."""
        base_args.anatomical, base_args.functional = False, True
        base_args.participant_label, base_args.session_label = ["01"], ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        with _patch_main(
            func_df_full, _make_groups(func_df_full, ["01"], ["baseline"])
        ) as (mock_anat, mock_func, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            assert main(args) == 0
            mock_func.assert_called_once()
            mock_anat.assert_not_called()

    def test_both_flags_dispatch_anat_and_func(
        self,
        mock_setup: Mock,  # noqa: ARG002
        base_args: argparse.Namespace,
        mixed_df: pl.DataFrame,
    ) -> None:
        """Test both longitudinal processing works in single call."""
        base_args.anatomical, base_args.functional = True, True
        base_args.participant_label, base_args.session_label = ["01"], ["baseline"]
        args = LongitudinalArgs.validate_namespace(base_args)
        with _patch_main(mixed_df, _make_groups(mixed_df, ["01"], ["baseline"])) as (
            mock_anat,
            mock_func,
            mock_ctx_cls,
        ):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            mock_anat.assert_called_once()
            mock_func.assert_called_once()


class TestRunnerSetup:
    """Test runner configuration and environment setup."""

    def test_experimental_warning_emitted(
        self, base_args: argparse.Namespace, anat_df_full: pl.DataFrame
    ) -> None:
        """Test experimental warning message is logged."""
        args = LongitudinalArgs.validate_namespace(base_args)
        with (
            patch("rbc.cli.longitudinal.setup_runner") as mock_setup,
            _patch_main(anat_df_full, _make_groups(anat_df_full, [], [])) as (
                _,
                __,
                mock_ctx_cls,
            ),
        ):
            ctx = Mock(runner=Mock(environ={}), logger=Mock(), verbose=False)
            mock_setup.return_value = ctx
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            main(args)
            ctx.logger.warning.assert_called_once()
            assert "experimental" in ctx.logger.warning.call_args[0][0].lower()


class TestProcessAnat:
    """Tests for _process_anat helper."""

    def test_calls_anatomical_longitudinal(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test anatomical longitudinal is called."""
        pipe_ctx = PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.cli.longitudinal.anatomical_longitudinal",
                return_value=_mock_anat_outputs(),
            ) as mock_long,
            patch(
                "rbc.core.bids2table.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
            patch("rbc.core.bids.shutil.copy2"),
        ):
            _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)
            assert mock_long.call_count == 1

    @pytest.mark.parametrize(
        ("null_field", "side_effect", "expected_error"),
        [
            ("brain", _none_for(suffix="T1w", desc="brain"), FileNotFoundError),
            ("brain_mask", None, ValueError),
        ],
        ids=["missing_brain_file", "missing_brain_mask_output"],
    )
    def test_missing_required_output_raises(
        self,
        anat_df: pl.DataFrame,
        tpl_df: pl.DataFrame,
        null_field: str,
        side_effect,  # noqa: ANN001
        expected_error: type,
        tmp_path: Path,
    ) -> None:
        """Test error raised if required anatomical outputs missing."""
        pipe_ctx = PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)
        outputs = _mock_anat_outputs()
        if side_effect is None:
            setattr(outputs, null_field, None)
            get_patch = patch(
                "rbc.core.bids2table.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            )
        else:
            get_patch = patch("rbc.core.bids2table.find_file", side_effect=side_effect)

        with (
            patch("rbc.cli.longitudinal.anatomical_longitudinal", return_value=outputs),
            get_patch,
            patch("rbc.core.bids.shutil.copy2"),
            pytest.raises(expected_error, match=null_field),
        ):
            _process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)


class TestProcessFunc:
    """Tests for _process_func helper."""

    def test_calls_functional_longitudinal(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test functional longitudinal is called."""
        pipe_ctx = PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.cli.longitudinal.functional_longitudinal",
                return_value=_mock_func_outputs(),
            ) as mock_func,
            patch(
                "rbc.core.bids2table.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
            patch("rbc.core.bids.shutil.copy2"),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert mock_func.call_count == 1

    @pytest.mark.parametrize(
        ("match_field", "match_kwargs"),
        [
            ("bold", {"suffix": "bold", "desc": "preproc"}),
            ("sbref", {"suffix": "sbref"}),
            (
                "bold_to_anat_xfm",
                {
                    "suffix": "xfm",
                    "desc": "linear",
                    "extension": ".txt",
                    "extra": {"from": "bold", "to": "T1w", "mode": "image"},
                },
            ),
        ],
    )
    def test_missing_required_file_raises(
        self,
        func_df: pl.DataFrame,
        tpl_df: pl.DataFrame,
        match_field: str,  # noqa: ARG002
        match_kwargs: dict,
        tmp_path: Path,
    ) -> None:
        """Test missing required functional outputs raises error."""
        pipe_ctx = PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.cli.longitudinal.functional_longitudinal",
                return_value=_mock_func_outputs(),
            ),
            patch(
                "rbc.core.bids2table.find_file",
                side_effect=_none_for(**match_kwargs),
            ),
            pytest.raises(FileNotFoundError),
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

    def test_optional_bold_mask_file_not_found(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Optional bold_mask not found is caught; 3 exports emitted."""
        pipe_ctx = PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.cli.longitudinal.functional_longitudinal",
                return_value=_mock_func_outputs(with_bold_mask=False),
            ),
            patch(
                "rbc.core.bids2table.find_file",
                side_effect=_none_for(suffix="mask", desc="brain"),
            ),
            patch("rbc.core.bids.shutil.copy2") as mock_copy,
        ):
            _process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert mock_copy.call_count == 3
