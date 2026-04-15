"""Unit tests for ``rbc.bids.longitudinal.anatomical``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids.longitudinal.anatomical import export_longitudinal_anat
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
