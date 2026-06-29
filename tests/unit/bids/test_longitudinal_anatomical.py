"""Unit tests for ``rbc.bids.longitudinal.anatomical``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from rbc.bids.longitudinal.anatomical import (
    export_longitudinal_anat,
    resolve_longitudinal_anat,
)
from rbc.context import RunContext
from rbc.workflows.longitudinal.anatomical import AnatomicalLongOutputs

if TYPE_CHECKING:
    from pathlib import Path


def _make_long_outputs(workdir: Path) -> AnatomicalLongOutputs:
    """Build a populated AnatomicalLongOutputs pointing at dummy files."""

    def _dummy(name: str) -> Path:
        p = workdir / name
        p.write_bytes(b"\x00")
        return p

    return AnatomicalLongOutputs(
        brain=_dummy("brain.nii.gz"),
        brain_mask=_dummy("brain_mask.nii.gz"),
        long_to_template_xfm=_dummy("long_to_tpl.nii.gz"),
        template_to_long_xfm=_dummy("tpl_to_long.nii.gz"),
    )


class TestExportLongitudinalAnat:
    """Tests for :func:`export_longitudinal_anat`."""

    def test_writes_expected_files(self, tmp_path: Path) -> None:
        """Exports brain, brain_mask, and the two xfms under space-longitudinal."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        out_dir = tmp_path / "out"

        ctx = RunContext(sub="01", ses="baseline", output_dir=out_dir)
        aex = ctx.bids(datatype="anat").derive(space="longitudinal")

        export_longitudinal_anat(aex, _make_long_outputs(workdir))

        saved = sorted(p.name for p in out_dir.rglob("*.*"))
        # Xfm filenames currently carry both ``space-longitudinal`` and the
        # ``from-…_to-…`` pair, which is redundant (Jason handover #18 flags
        # the ``from-`` workaround more broadly). We assert the current shape
        # so regressions are caught, but if that cleanup lands the expected
        # xfm names here should drop ``space-longitudinal_``.
        assert saved == [
            "sub-01_ses-baseline_space-longitudinal_desc-T1w_mask.nii.gz",
            "sub-01_ses-baseline_space-longitudinal_desc-brain_T1w.nii.gz",
            "sub-01_ses-baseline_space-longitudinal_from-MNI152NLin6Asym"
            "_to-longitudinal_mode-image_xfm.nii.gz",
            "sub-01_ses-baseline_space-longitudinal_from-longitudinal"
            "_to-MNI152NLin6Asym_mode-image_xfm.nii.gz",
        ]

    def test_tissue_masks_not_produced(self, tmp_path: Path) -> None:
        """No csf/gm/wm masks are written under space-longitudinal (#5)."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        out_dir = tmp_path / "out"

        ctx = RunContext(sub="01", ses="baseline", output_dir=out_dir)
        aex = ctx.bids(datatype="anat").derive(space="longitudinal")

        export_longitudinal_anat(aex, _make_long_outputs(workdir))

        names = [p.name for p in out_dir.rglob("*.*")]
        for tissue in ("csf", "gm", "wm"):
            assert not any(f"desc-{tissue}_mask" in n for n in names), (
                f"Expected no space-longitudinal {tissue}_mask; got {names}"
            )


def _anat_row(
    *,
    sub: str,
    ses: str,
    suffix: str,
    desc: str | None,
    ext: str = ".nii.gz",
    extra: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build a single BIDS-like row, including an ``extra_entities`` list."""
    desc_part = f"_desc-{desc}" if desc else ""
    path = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}{desc_part}_{suffix}{ext}"
    return {
        "datatype": "anat",
        "suffix": suffix,
        "ext": ext,
        "sub": sub,
        "ses": ses,
        "desc": desc,
        "res": None,
        "root": "/data",
        "path": path,
        "extra_entities": extra or [],
    }


def _df(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


class TestResolveLongitudinalAnat:
    """Tests for :func:`resolve_longitudinal_anat`."""

    def test_resolves_all_inputs(self, tmp_path: Path) -> None:
        """Happy path: all four keys resolve to the expected paths."""
        anat_df = _df(
            _anat_row(sub="01", ses="baseline", suffix="T1w", desc="brain"),
            _anat_row(sub="01", ses="baseline", suffix="mask", desc="T1w"),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", suffix="T1w", desc=None),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                desc=None,
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        anat_q = ctx.bids(datatype="anat")
        tpl_q = anat_q.derive(ses="longitudinal")

        resolved = resolve_longitudinal_anat(
            anat_q, tpl_q, anat_df, tpl_df, ses="baseline"
        )

        assert set(resolved) == {
            "template",
            "subj_to_template_xfm",
            "brain",
            "brain_mask",
        }
        assert str(resolved["template"]).endswith("sub-01_ses-longitudinal_T1w.nii.gz")
        assert str(resolved["subj_to_template_xfm"]).endswith(
            "sub-01_ses-longitudinal_xfm.txt"
        )
        assert str(resolved["brain"]).endswith(
            "sub-01_ses-baseline_desc-brain_T1w.nii.gz"
        )
        assert str(resolved["brain_mask"]).endswith(
            "sub-01_ses-baseline_desc-T1w_mask.nii.gz"
        )

    def test_missing_brain_mask_is_none(self, tmp_path: Path) -> None:
        """brain_mask is optional (``.find``) — returns None when absent."""
        anat_df = _df(
            _anat_row(sub="01", ses="baseline", suffix="T1w", desc="brain"),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", suffix="T1w", desc=None),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                desc=None,
                ext=".txt",
                extra=[
                    {"key": "from", "value": "baseline"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="baseline", output_dir=tmp_path)
        anat_q = ctx.bids(datatype="anat")
        tpl_q = anat_q.derive(ses="longitudinal")

        resolved = resolve_longitudinal_anat(
            anat_q, tpl_q, anat_df, tpl_df, ses="baseline"
        )

        assert resolved["brain_mask"] is None
        assert resolved["brain"] is not None

    def test_xfm_query_uses_bids_safe_label(self, tmp_path: Path) -> None:
        """Session labels with non-alphanumerics resolve through ``bids_safe_label``.

        ``pre-op`` is stored as ``from-preop`` in the xfm entity; the resolver
        must sanitize the ses arg before querying or the xfm won't be found.
        """
        anat_df = _df(
            _anat_row(sub="01", ses="pre-op", suffix="T1w", desc="brain"),
        )
        tpl_df = _df(
            _anat_row(sub="01", ses="longitudinal", suffix="T1w", desc=None),
            _anat_row(
                sub="01",
                ses="longitudinal",
                suffix="xfm",
                desc=None,
                ext=".txt",
                extra=[
                    {"key": "from", "value": "preop"},
                    {"key": "to", "value": "longitudinal"},
                ],
            ),
        )

        ctx = RunContext(sub="01", ses="pre-op", output_dir=tmp_path)
        anat_q = ctx.bids(datatype="anat")
        tpl_q = anat_q.derive(ses="longitudinal")

        resolved = resolve_longitudinal_anat(
            anat_q, tpl_q, anat_df, tpl_df, ses="pre-op"
        )

        assert resolved["subj_to_template_xfm"] is not None
        assert str(resolved["subj_to_template_xfm"]).endswith(
            "sub-01_ses-longitudinal_xfm.txt"
        )
