"""Unit tests for ``rbc.bids.longitudinal.functional``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from rbc.bids.longitudinal.functional import (
    export_longitudinal_func,
    resolve_longitudinal_func,
)
from rbc.context import RunContext
from rbc.workflows.longitudinal.functional import FunctionalLongOutputs

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _func_row(
    *,
    sub: str,
    ses: str,
    suffix: str,
    task: str,
    desc: str | None = None,
    ext: str = ".nii.gz",
    space: str | None = None,
    extra: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build a single BIDS-like row for functional derivatives."""
    desc_part = f"_desc-{desc}" if desc else ""
    space_part = f"_space-{space}" if space else ""
    path = (
        f"sub-{sub}/ses-{ses}/func/"
        f"sub-{sub}_ses-{ses}{space_part}_task-{task}{desc_part}_{suffix}{ext}"
    )
    return {
        "datatype": "func",
        "suffix": suffix,
        "task": task,
        "ext": ext,
        "sub": sub,
        "ses": ses,
        "space": space,
        "desc": desc,
        "root": "/data",
        "path": path,
        "extra_entities": extra or [],
    }


def _anat_row(
    *,
    sub: str,
    ses: str,
    suffix: str,
    res: str | None = None,
    desc: str | None = None,
    ext: str = ".nii.gz",
    extra: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build a single BIDS-like row for anatomical/template derivatives."""
    desc_part = f"_desc-{desc}" if desc else ""
    path = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}{desc_part}_{suffix}{ext}"
    return {
        "datatype": "anat",
        "suffix": suffix,
        "ext": ext,
        "sub": sub,
        "ses": ses,
        "space": None,
        "res": res,
        "desc": desc,
        "root": "/data",
        "path": path,
        "extra_entities": extra or [],
    }


def _df(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def _make_long_outputs(workdir: Path) -> FunctionalLongOutputs:
    """Build a populated FunctionalLongOutputs pointing at dummy files."""

    def _dummy(name: str) -> Path:
        p = workdir / name
        p.write_bytes(b"\x00")
        return p

    return FunctionalLongOutputs(
        bold_to_long_xfm=_dummy("bold_to_long_xfm.nii.gz"),
        sbref=_dummy("sbref.nii.gz"),
        bold=_dummy("bold.nii.gz"),
        bold_mask=_dummy("bold_mask.nii.gz"),
        regressed_bold={
            "36-parameter": _dummy("regressed_36p.nii.gz"),
            "aCompCor": _dummy("regressed_acompcor.nii.gz"),
        },
        cleaned_bold={
            "36-parameter": _dummy("cleaned_36p.nii.gz"),
            "aCompCor": _dummy("cleaned_acompcor.nii.gz"),
        },
        cleaned_bold_smooth={
            "36-parameter": _dummy("cleaned_36p_smooth.nii.gz"),
            "aCompCor": _dummy("cleaned_acompcor_smooth.nii.gz"),
        },
    )


# ---------------------------------------------------------------------------
# resolve_longitudinal_func
# ---------------------------------------------------------------------------


class TestResolveLongitudinalFunc:
    """Tests for :func:`resolve_longitudinal_func`."""

    def test_resolves_single_regressor(self, tmp_path: Path) -> None:
        """Single regressor resolves raw regressor file from derivatives."""
        func_df = _df(
            _func_row(sub="01", ses="baseline", task="rest", suffix="sbref"),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="bold", desc="preproc"
            ),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="mask", desc="brain"
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="xfm",
                desc="linearITK",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "bold"},
                    {"key": "to", "value": "T1w"},
                    {"key": "mode", "value": "image"},
                ],
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="regressors",
                desc="36parameter",
                ext=".1D",
            ),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", res="rest", suffix="T1w"),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        func_q = ctx.bids(datatype="func")
        tpl_q = ctx.bids(datatype="anat").derive(ses="longitudinal")

        resolved = resolve_longitudinal_func(
            func_q,
            tpl_q,
            func_df,
            tpl_df,
            ses="baseline",
            task="rest",
            regressors=["36-parameter"],
        )

        assert set(resolved) == {
            "template",
            "anat_to_template_xfm",
            "bold_to_anat_itk",
            "sbref",
            "bold",
            "bold_mask",
            "regressor_files",
        }
        assert isinstance(resolved["regressor_files"], dict)
        assert "36-parameter" in resolved["regressor_files"]
        assert str(resolved["regressor_files"]["36-parameter"]).endswith(
            "regressors.1D"
        )

    def test_resolves_multiple_regressors(self, tmp_path: Path) -> None:
        """Multiple regressors each get their own raw regressor file resolved."""
        func_df = _df(
            _func_row(sub="01", ses="baseline", task="rest", suffix="sbref"),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="bold", desc="preproc"
            ),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="mask", desc="brain"
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="xfm",
                desc="linearITK",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "bold"},
                    {"key": "to", "value": "T1w"},
                    {"key": "mode", "value": "image"},
                ],
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="regressors",
                desc="36parameter",
                ext=".1D",
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="regressors",
                desc="aCompCor",
                ext=".1D",
            ),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", res="rest", suffix="T1w"),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        func_q = ctx.bids(datatype="func")
        tpl_q = ctx.bids(datatype="anat").derive(ses="longitudinal")

        resolved = resolve_longitudinal_func(
            func_q,
            tpl_q,
            func_df,
            tpl_df,
            ses="baseline",
            task="rest",
            regressors=["36-parameter", "aCompCor"],
        )

        reg_files = resolved["regressor_files"]
        assert isinstance(reg_files, dict)
        assert set(reg_files) == {"36-parameter", "aCompCor"}

    def test_missing_regressor_raises(self, tmp_path: Path) -> None:
        """Requesting a regressor not present in derivatives raises."""
        func_df = _df(
            _func_row(sub="01", ses="baseline", task="rest", suffix="sbref"),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="bold", desc="preproc"
            ),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="mask", desc="brain"
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="xfm",
                desc="linearITK",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "bold"},
                    {"key": "to", "value": "T1w"},
                    {"key": "mode", "value": "image"},
                ],
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="regressors",
                desc="36parameter",
                ext=".1D",
            ),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", res="rest", suffix="T1w"),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        func_q = ctx.bids(datatype="func")
        tpl_q = ctx.bids(datatype="anat").derive(ses="longitudinal")

        with pytest.raises(FileNotFoundError):
            resolve_longitudinal_func(
                func_q,
                tpl_q,
                func_df,
                tpl_df,
                ses="baseline",
                task="rest",
                regressors=["aCompCor"],
            )

    def test_bold_mask_mandatory(self, tmp_path: Path) -> None:
        """bold_mask is now resolved with expect(), so missing raises."""
        func_df = _df(
            _func_row(sub="01", ses="baseline", task="rest", suffix="sbref"),
            _func_row(
                sub="01", ses="baseline", task="rest", suffix="bold", desc="preproc"
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="xfm",
                desc="linearITK",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "bold"},
                    {"key": "to", "value": "T1w"},
                    {"key": "mode", "value": "image"},
                ],
            ),
            _func_row(
                sub="01",
                ses="baseline",
                task="rest",
                suffix="regressors",
                desc="36parameter",
                ext=".1D",
            ),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", res="rest", suffix="T1w"),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        func_q = ctx.bids(datatype="func")
        tpl_q = ctx.bids(datatype="anat").derive(ses="longitudinal")

        with pytest.raises(FileNotFoundError):
            resolve_longitudinal_func(
                func_q,
                tpl_q,
                func_df,
                tpl_df,
                ses="baseline",
                task="rest",
                regressors=["36-parameter"],
            )


# ---------------------------------------------------------------------------
# export_longitudinal_func
# ---------------------------------------------------------------------------


class TestExportLongitudinalFunc:
    """Tests for :func:`export_longitudinal_func`."""

    def test_writes_expected_files(self, tmp_path: Path) -> None:
        """Exports BOLD, sbref, mask, xfm, plus per-regressor regressed/cleaned."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        out_dir = tmp_path / "out"

        ctx = RunContext(sub="01", ses="baseline", output_dir=out_dir)
        fex = ctx.bids(datatype="func").derive(space="longitudinal")

        outputs = _make_long_outputs(workdir)
        export_longitudinal_func(fex, outputs, regressors=["36-parameter", "aCompCor"])

        saved = sorted(p.name for p in out_dir.rglob("*.*"))
        # 4 fixed (sbref, bold, xfm, mask)
        # + 2 regressors x 2 (regressed + cleaned) = 4 per-regressor
        # = 8 total
        assert len(saved) == 8

    def test_regressor_entity_in_filenames(self, tmp_path: Path) -> None:
        """Per-regressor outputs carry the reg-<strategy> entity."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        out_dir = tmp_path / "out"

        ctx = RunContext(sub="01", ses="baseline", output_dir=out_dir)
        fex = ctx.bids(datatype="func").derive(space="longitudinal")

        outputs = _make_long_outputs(workdir)
        export_longitudinal_func(fex, outputs, regressors=["36-parameter", "aCompCor"])

        names = [p.name for p in out_dir.rglob("*.*")]
        reg_files = [n for n in names if "reg-" in n]
        assert len(reg_files) == 4

        assert any("reg-36parameter" in n for n in reg_files)
        assert any("reg-aCompCor" in n for n in reg_files)

    def test_single_regressor_file_count(self, tmp_path: Path) -> None:
        """Single regressor produces 6 files: 4 fixed + 2 per-regressor."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        out_dir = tmp_path / "out"

        ctx = RunContext(sub="01", ses="baseline", output_dir=out_dir)
        fex = ctx.bids(datatype="func").derive(space="longitudinal")

        outputs = _make_long_outputs(workdir)
        export_longitudinal_func(fex, outputs, regressors=["36-parameter"])

        saved = list(out_dir.rglob("*.*"))
        assert len(saved) == 6
