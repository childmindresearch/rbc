"""Unit tests for BIDS export/resolve functions.

Tests use real Bids instances (not mocks) so that BIDS entity validation
actually runs. This catches bugs like unsanitized atlas or regressor names.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import pytest

from rbc.bids.anatomical import export_anatomical
from rbc.bids.functional import export_functional
from rbc.bids.metrics import export_metrics
from rbc.bids.qc import export_qc
from rbc.context import RunContext
from rbc.workflows.anatomical import AnatomicalOutputs
from rbc.workflows.functional import FunctionalOutputs
from rbc.workflows.metrics import MetricsOutputs

if TYPE_CHECKING:
    from pathlib import Path

    from rbc.bids import Bids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy(workdir: Path, name: str) -> Path:
    """Create a dummy file and return its path."""
    p = workdir / name
    p.write_bytes(b"\x00")
    return p


def _make_anat_outputs(w: Path) -> AnatomicalOutputs:
    return AnatomicalOutputs(
        brain=_dummy(w, "brain.nii.gz"),
        brain_mask=_dummy(w, "brain_mask.nii.gz"),
        brain_tpl=_dummy(w, "brain_tpl.nii.gz"),
        csf_mask=_dummy(w, "csf_mask.nii.gz"),
        gm_mask=_dummy(w, "gm_mask.nii.gz"),
        wm_mask=_dummy(w, "wm_mask.nii.gz"),
        wm_bbr_mask=_dummy(w, "wm_bbr_mask.nii.gz"),
        forward_xfm=_dummy(w, "forward_xfm.nii.gz"),
        inverse_xfm=_dummy(w, "inverse_xfm.nii.gz"),
    )


def _make_func_outputs(w: Path, regressors: list[str]) -> FunctionalOutputs:
    return FunctionalOutputs(
        reoriented_bold=_dummy(w, "reoriented.nii.gz"),
        truncated_bold=_dummy(w, "truncated.nii.gz"),
        despiked_bold=_dummy(w, "despiked.nii.gz"),
        sbref=_dummy(w, "sbref.nii.gz"),
        distortion_corrected_ref=None,
        distortion_warp=None,
        stc_bold=_dummy(w, "stc.nii.gz"),
        preproc_bold=_dummy(w, "preproc.nii.gz"),
        motion_params=_dummy(w, "motion.1D"),
        rms_rel=_dummy(w, "rms_rel.rms"),
        rms_abs=_dummy(w, "rms_abs.rms"),
        mat_dir=w / "mat",
        bold_mask=_dummy(w, "mask.nii.gz"),
        skull_stripped_bold=_dummy(w, "skull_stripped.nii.gz"),
        bold_to_anat_matrix=_dummy(w, "bold2anat.txt"),
        bold_to_anat_itk=_dummy(w, "bold2anat_itk.txt"),
        template_bold=_dummy(w, "template_bold.nii.gz"),
        regressed_bold={r: _dummy(w, f"regressed_{r}.nii.gz") for r in regressors},
        cleaned_bold={r: _dummy(w, f"cleaned_{r}.nii.gz") for r in regressors},
        regressor_file={r: _dummy(w, f"regressors_{r}.1D") for r in regressors},
        template_brain_mask=_dummy(w, "template_mask.nii.gz"),
    )


def _make_metrics_outputs(w: Path, atlases: list[str]) -> MetricsOutputs:
    return MetricsOutputs(
        alff=_dummy(w, "alff.nii.gz"),
        falff=_dummy(w, "falff.nii.gz"),
        alff_smooth=_dummy(w, "alff_smooth.nii.gz"),
        falff_smooth=_dummy(w, "falff_smooth.nii.gz"),
        alff_zscored=_dummy(w, "alff_z.nii.gz"),
        falff_zscored=_dummy(w, "falff_z.nii.gz"),
        reho=_dummy(w, "reho.nii.gz"),
        reho_smooth=_dummy(w, "reho_smooth.nii.gz"),
        reho_zscored=_dummy(w, "reho_z.nii.gz"),
        timeseries={a: _dummy(w, f"ts_{a}.tsv") for a in atlases},
        correlation_matrix={a: _dummy(w, f"corr_{a}.tsv") for a in atlases},
    )


@pytest.fixture
def pipe_ctx(tmp_path: Path) -> RunContext:
    """RunContext with a temp output directory."""
    return RunContext(sub="01", ses="baseline", output_dir=tmp_path / "output")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Working directory for dummy source files."""
    w = tmp_path / "work"
    w.mkdir()
    return w


@pytest.fixture
def anat_bids(pipe_ctx: RunContext) -> Bids:
    """Bids builder for anat datatype."""
    return pipe_ctx.bids(datatype="anat")


@pytest.fixture
def func_bids(pipe_ctx: RunContext) -> Bids:
    """Bids builder for func datatype."""
    return pipe_ctx.bids(datatype="func", entities={"task": "rest", "run": 1})


# ---------------------------------------------------------------------------
# Anatomical exports
# ---------------------------------------------------------------------------


class TestExportAnatomical:
    """Tests for export_anatomical."""

    def test_creates_8_files(
        self, anat_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """All 8 anatomical outputs are saved."""
        outputs = _make_anat_outputs(workdir)
        export_anatomical(anat_bids, outputs)
        saved = list(pipe_ctx.output_dir.rglob("*.*"))
        assert len(saved) == 9

    def test_filenames_contain_expected_entities(
        self, anat_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Output filenames contain sub and ses entities."""
        outputs = _make_anat_outputs(workdir)
        export_anatomical(anat_bids, outputs)
        for p in pipe_ctx.output_dir.rglob("*.*"):
            assert "sub-01" in p.name
            assert "ses-baseline" in p.name


# ---------------------------------------------------------------------------
# Functional exports
# ---------------------------------------------------------------------------


class TestExportFunctional:
    """Tests for export_functional."""

    def test_returns_mni_builder(self, func_bids: Bids, workdir: Path) -> None:
        """export_functional returns a Bids builder with MNI space."""
        outputs = _make_func_outputs(workdir, ["36-parameter"])
        mni = export_functional(func_bids, outputs, regressors=["36-parameter"])
        path = mni.path(suffix="bold")
        assert "space-MNI152NLin6Asym" in path.name

    def test_sanitizes_regressor_labels(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Regressor names with hyphens are sanitized in filenames."""
        outputs = _make_func_outputs(workdir, ["36-parameter"])
        export_functional(func_bids, outputs, regressors=["36-parameter"])
        all_names = [p.name for p in pipe_ctx.output_dir.rglob("*.*")]
        reg_files = [n for n in all_names if "reg-" in n]
        for name in reg_files:
            assert "36parameter" in name
            assert "36-parameter" not in name

    def test_file_count_single_regressor(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Correct file count with one regressor.

        8 native-space fixed + 1 regressor file + 2 per-regressor MNI
        + 2 fixed MNI = 13.
        """
        outputs = _make_func_outputs(workdir, ["36-parameter"])
        export_functional(func_bids, outputs, regressors=["36-parameter"])
        saved = list(pipe_ctx.output_dir.rglob("*.*"))
        assert len(saved) == 13

    def test_file_count_two_regressors(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Correct file count with two regressors.

        8 fixed + 2 regressor files + 4 per-regressor MNI + 2 fixed MNI = 16.
        """
        regs = ["36-parameter", "aCompCor"]
        outputs = _make_func_outputs(workdir, regs)
        export_functional(func_bids, outputs, regressors=regs)
        saved = list(pipe_ctx.output_dir.rglob("*.*"))
        assert len(saved) == 16


# ---------------------------------------------------------------------------
# Metrics exports
# ---------------------------------------------------------------------------


class TestExportMetrics:
    """Tests for export_metrics."""

    def test_sanitizes_atlas_labels(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Atlas names with underscores are sanitized in filenames."""
        mni = func_bids.derive(space="MNI152NLin6Asym")
        outputs = _make_metrics_outputs(workdir, ["schaefer_200"])
        export_metrics(mni, outputs, regressor="36-parameter", atlases=["schaefer_200"])
        atlas_files = [
            p.name for p in pipe_ctx.output_dir.rglob("*.*") if "atlas-" in p.name
        ]
        assert len(atlas_files) > 0
        for name in atlas_files:
            assert "atlas-schaefer200" in name
            assert "schaefer_200" not in name

    def test_sanitizes_regressor_labels(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Regressor names with hyphens are sanitized in filenames."""
        mni = func_bids.derive(space="MNI152NLin6Asym")
        outputs = _make_metrics_outputs(workdir, ["aal"])
        export_metrics(mni, outputs, regressor="36-parameter", atlases=["aal"])
        all_names = [p.name for p in pipe_ctx.output_dir.rglob("*.*")]
        reg_files = [n for n in all_names if "reg-" in n]
        assert len(reg_files) > 0
        for name in reg_files:
            assert "reg-36parameter" in name

    def test_file_count_single_atlas(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """9 scalar maps + 2 atlas files = 11."""
        mni = func_bids.derive(space="MNI152NLin6Asym")
        outputs = _make_metrics_outputs(workdir, ["schaefer_200"])
        export_metrics(mni, outputs, regressor="aCompCor", atlases=["schaefer_200"])
        saved = list(pipe_ctx.output_dir.rglob("*.*"))
        assert len(saved) == 11

    def test_file_count_multiple_atlases(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """9 scalar maps + 2 * 2 atlas files = 13."""
        atlases = ["schaefer_200", "aal"]
        mni = func_bids.derive(space="MNI152NLin6Asym")
        outputs = _make_metrics_outputs(workdir, atlases)
        export_metrics(mni, outputs, regressor="aCompCor", atlases=atlases)
        saved = list(pipe_ctx.output_dir.rglob("*.*"))
        assert len(saved) == 13


# ---------------------------------------------------------------------------
# QC exports
# ---------------------------------------------------------------------------


class TestExportQC:
    """Tests for export_qc."""

    def test_sanitizes_regressor_labels(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """Regressor names with hyphens are sanitized in QC filenames."""
        from dataclasses import dataclass

        @dataclass
        class _FakeQC:
            qc_file: dict[str, Path] = field(default_factory=dict)

        mni = func_bids.derive(space="MNI152NLin6Asym")
        qc = _FakeQC(qc_file={"36-parameter": _dummy(workdir, "qc.tsv")})
        export_qc(mni, qc, regressors=["36-parameter"])  # type: ignore[arg-type]
        saved = list(pipe_ctx.output_dir.rglob("*.tsv"))
        assert len(saved) == 1
        assert "reg-36parameter" in saved[0].name

    def test_file_count_two_regressors(
        self, func_bids: Bids, workdir: Path, pipe_ctx: RunContext
    ) -> None:
        """One QC file per regressor."""
        from dataclasses import dataclass

        @dataclass
        class _FakeQC:
            qc_file: dict[str, Path] = field(default_factory=dict)

        regs = ["36-parameter", "aCompCor"]
        mni = func_bids.derive(space="MNI152NLin6Asym")
        qc = _FakeQC(qc_file={r: _dummy(workdir, f"qc_{r}.tsv") for r in regs})
        export_qc(mni, qc, regressors=regs)  # type: ignore[arg-type]
        saved = list(pipe_ctx.output_dir.rglob("*.tsv"))
        assert len(saved) == 2
