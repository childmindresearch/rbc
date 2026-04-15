"""Unit tests for longitudinal orchestration helpers and dispatch."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

import polars as pl
import pytest

from rbc.context import RunContext
from rbc.orchestration import Filters
from rbc.orchestration.longitudinal import process_anat, process_func

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
    m.long_to_template_xfm = fake / "long_to_template_xfm.nii.gz"
    m.template_to_long_xfm = fake / "template_to_long_xfm.nii.gz"
    return m


def _mock_func_outputs(*, with_bold_mask: bool = True) -> Mock:
    fake = Path("fake_workdir")
    m = Mock()
    m.sbref = fake / "sbref.nii.gz"
    m.bold = fake / "bold.nii.gz"
    m.bold_to_long_xfm = fake / "bold_to_long_xfm.nii.gz"
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
    """Anat-only dataframe with longitudinal templates."""
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
    """Func dataframe with longitudinal templates."""
    return _df(
        _func_row("01", "baseline"),
        _func_row("01", "vis2"),
        _func_row("02", "baseline"),
        _func_row("02", "vis2"),
        _anat_row("01", "longitudinal"),
        _anat_row("02", "longitudinal"),
    )


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
def _patch_anat_run(
    full_df: pl.DataFrame,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Patch external calls made by orchestration.longitudinal.anatomical.run()."""
    from rbc.bids.session import SessionTables

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
        patch("rbc.orchestration.longitudinal.anatomical.init_runner"),
        patch("rbc.orchestration.longitudinal._iter.load_table", return_value=full_df),
        patch(
            "rbc.orchestration.longitudinal._iter.load_session",
            return_value=mock_session,
        ),
        patch(
            "rbc.bids.query.find_file",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.orchestration.longitudinal.anatomical.anatomical_longitudinal",
            return_value=_mock_anat_outputs(),
        ) as mock_anat,
        patch("rbc.orchestration.longitudinal._iter.RunContext") as mock_ctx_cls,
    ):
        yield mock_anat, mock_ctx_cls


@contextmanager
def _patch_func_run(
    full_df: pl.DataFrame,
    groups: list[tuple],
    *,
    with_bold_mask: bool = True,
) -> Generator[tuple[Mock, Mock], None, None]:
    """Patch external calls made by orchestration.longitudinal.functional.run()."""
    from rbc.bids.session import SessionTables

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
        patch("rbc.orchestration.longitudinal.functional.init_runner"),
        patch("rbc.orchestration.longitudinal._iter.load_table", return_value=full_df),
        patch(
            "rbc.orchestration.longitudinal._iter.load_session",
            return_value=mock_session,
        ),
        patch(
            "rbc.orchestration.longitudinal.functional.iter_session_files",
            side_effect=_build_iter_side_effect(groups),
        ),
        patch(
            "rbc.bids.query.find_file",
            return_value=Path("fake_workdir/file.nii.gz"),
        ),
        patch(
            "rbc.orchestration.longitudinal.functional.functional_longitudinal",
            return_value=_mock_func_outputs(with_bold_mask=with_bold_mask),
        ) as mock_func,
        patch("rbc.orchestration.longitudinal._iter.RunContext") as mock_ctx_cls,
    ):
        yield mock_func, mock_ctx_cls


class TestProcessAnat:
    """Tests for process_anat helper."""

    def test_calls_anatomical_longitudinal(
        self, anat_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test anatomical longitudinal is called."""
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.orchestration.longitudinal.anatomical.anatomical_longitudinal",
                return_value=_mock_anat_outputs(),
            ) as mock_long,
            patch(
                "rbc.bids.query.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
            patch("rbc.bids.builder.shutil.copy2"),
        ):
            process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)
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
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        outputs = _mock_anat_outputs()
        if side_effect is None:
            setattr(outputs, null_field, None)
            get_patch = patch(
                "rbc.bids.query.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            )
        else:
            get_patch = patch("rbc.bids.query.find_file", side_effect=side_effect)

        with (
            patch(
                "rbc.orchestration.longitudinal.anatomical.anatomical_longitudinal",
                return_value=outputs,
            ),
            get_patch,
            patch("rbc.bids.builder.shutil.copy2"),
            pytest.raises(expected_error, match=null_field),
        ):
            process_anat(pipe_ctx=pipe_ctx, anat_df=anat_df, tpl_df=tpl_df)


class TestProcessFunc:
    """Tests for process_func helper."""

    def test_calls_functional_longitudinal(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test functional longitudinal is called."""
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.orchestration.longitudinal.functional.functional_longitudinal",
                return_value=_mock_func_outputs(),
            ) as mock_func,
            patch(
                "rbc.bids.query.find_file",
                return_value=Path("fake_workdir/file.nii.gz"),
            ),
            patch("rbc.bids.builder.shutil.copy2"),
        ):
            process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
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
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.orchestration.longitudinal.functional.functional_longitudinal",
                return_value=_mock_func_outputs(),
            ),
            patch(
                "rbc.bids.query.find_file",
                side_effect=_none_for(**match_kwargs),
            ),
            pytest.raises(FileNotFoundError),
        ):
            process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)

    def test_optional_bold_mask_file_not_found(
        self, func_df: pl.DataFrame, tpl_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Optional bold_mask not found is caught; 3 exports emitted."""
        pipe_ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        with (
            patch(
                "rbc.orchestration.longitudinal.functional.functional_longitudinal",
                return_value=_mock_func_outputs(with_bold_mask=False),
            ),
            patch(
                "rbc.bids.query.find_file",
                side_effect=_none_for(suffix="mask", desc="brain"),
            ),
            patch("rbc.bids.builder.shutil.copy2") as mock_copy,
        ):
            process_func(pipe_ctx=pipe_ctx, func_df=func_df, tpl_df=tpl_df)
            assert mock_copy.call_count == 3


class TestLongitudinalAnatomicalRun:
    """Tests for the longitudinal anatomical orchestration entrypoint."""

    def test_missing_template_raises(
        self,
        anat_df_full: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Missing longitudinal template raises ValueError."""
        from rbc.orchestration.longitudinal.anatomical import run

        df_no_tpl = anat_df_full.filter(pl.col("ses") != "longitudinal")
        with _patch_anat_run(df_no_tpl) as (_, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            with pytest.raises(ValueError, match="No longitudinal template found"):
                run(
                    input_dirs=[tmp_path],
                    output_dir=tmp_path,
                    filters=Filters(),
                )

    def test_dispatches_anat_processing(
        self,
        anat_df_full: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Anatomical processing dispatches for each matching session."""
        from rbc.orchestration.longitudinal.anatomical import run

        with _patch_anat_run(anat_df_full) as (mock_anat, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            run(
                input_dirs=[tmp_path],
                output_dir=tmp_path,
                filters=Filters(participant_label=["01"], session_label=["baseline"]),
            )
            mock_anat.assert_called_once()


class TestLongitudinalFunctionalRun:
    """Tests for the longitudinal functional orchestration entrypoint."""

    def test_dispatches_func_processing(
        self,
        func_df_full: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Functional processing dispatches for each matching BOLD run."""
        from rbc.orchestration.longitudinal.functional import run

        groups = _make_groups(func_df_full, ["01"], ["baseline"])
        with _patch_func_run(func_df_full, groups) as (mock_func, mock_ctx_cls):
            mock_ctx_cls.return_value = Mock(sub="01", ses="baseline")
            run(
                input_dirs=[tmp_path],
                output_dir=tmp_path,
                filters=Filters(participant_label=["01"], session_label=["baseline"]),
            )
            mock_func.assert_called_once()

    def test_experimental_warning_emitted(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Experimental warning is logged when run() starts."""
        from rbc.orchestration.longitudinal.functional import run

        empty_df = pl.DataFrame(
            {c: [] for c in ["sub", "ses", "datatype", "suffix", "space", "task"]}
        )
        with (
            caplog.at_level(logging.WARNING),
            patch("rbc.orchestration.longitudinal.functional.init_runner"),
            patch(
                "rbc.orchestration.longitudinal._iter.load_table", return_value=empty_df
            ),
        ):
            run(
                input_dirs=[tmp_path],
                output_dir=tmp_path,
                filters=Filters(),
            )
            assert any("experimental" in msg.lower() for msg in caplog.messages)
