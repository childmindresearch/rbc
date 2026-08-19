"""Unit tests for rbc.core.qc.report."""

from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from rbc.core.qc.dvars import DVARSQCMetrics
from rbc.core.qc.motion import MotionQCMetrics
from rbc.core.qc.registration import RegistrationQCMetrics
from rbc.core.qc.report import (
    ReportSection,
    _warp_mask,
    figure_to_svg,
    generate_qc_report,
    metric_rows,
    render_carpet,
    render_lightbox,
)
from rbc.core.qc.xcp import XCPQCMetrics, generate_xcp_qc

if TYPE_CHECKING:
    from pathlib import Path

_SHAPE3 = (16, 16, 16)
_N_VOLS = 20
_SVG_PREFIX = b"<?xml"


def _write_nifti(
    path: Path,
    data: np.ndarray,
    spacing: float = 2.0,
    affine: np.ndarray | None = None,
) -> Path:
    """Write *data* to *path* as a float32 NIfTI."""
    if affine is None:
        affine = np.diag([spacing, spacing, spacing, 1.0])
    nib.Nifti1Image(data.astype(np.float32), affine).to_filename(str(path))
    return path


def _sphere() -> np.ndarray:
    """Return a spherical mask on the test grid."""
    n = _SHAPE3[0]
    xx, yy, zz = np.mgrid[0:n, 0:n, 0:n]
    d2 = (xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2
    return (d2 < (n / 2) ** 2).astype(np.float32)


@pytest.fixture
def qc_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create the synthetic NIfTI/text inputs consumed by the report."""
    rng = np.random.default_rng(42)
    sphere = _sphere()
    dataset: dict[str, Path] = {
        "brain_mask": _write_nifti(tmp_path / "brain_mask.nii.gz", sphere),
        "bold_mask": _write_nifti(tmp_path / "bold_mask.nii.gz", sphere, spacing=2.5),
        "template_brain_mask": _write_nifti(
            tmp_path / "template_brain_mask.nii.gz", sphere
        ),
        "func_template": _write_nifti(
            tmp_path / "func_template.nii.gz", rng.random(_SHAPE3) * 100
        ),
        "cleaned_36": _write_nifti(
            tmp_path / "cleaned_36.nii.gz", rng.random((*_SHAPE3, _N_VOLS)) * 10
        ),
        "cleaned_acomp": _write_nifti(
            tmp_path / "cleaned_acomp.nii.gz", rng.random((*_SHAPE3, _N_VOLS)) * 10
        ),
    }
    xfm = tmp_path / "xfm.txt"
    np.savetxt(xfm, np.eye(4))
    dataset["xfm"] = xfm
    motion_params = tmp_path / "motion.1D"
    np.savetxt(motion_params, rng.random((_N_VOLS, 6)) * 0.1)
    dataset["motion_params"] = motion_params
    rms_rel = tmp_path / "rel.rms"
    np.savetxt(rms_rel, rng.random(_N_VOLS - 1) * 0.1)
    dataset["rms_rel"] = rms_rel
    return dataset


def _metrics(regressor: str) -> XCPQCMetrics:
    """Build a passing-quality XCP metrics row for *regressor*."""
    return generate_xcp_qc(
        sub="01",
        ses="baseline",
        task="rest",
        run=1,
        desc="RBC",
        regressors=regressor,
        space="MNI152NLin2009CAsym",
        motion=MotionQCMetrics(0.05, 0.03, 0.1, 1),
        dvars_init=DVARSQCMetrics(1.2, 0.2),
        dvars_final=DVARSQCMetrics(1.0, 0.1),
        n_vols_removed=2,
        coreg=RegistrationQCMetrics(0.90, 0.82, 0.88, 0.95),
        norm=RegistrationQCMetrics(0.85, 0.75, 0.85, 0.90),
    )


def _generate(
    qc_dataset: dict[str, Path],
    tmp_path: Path,
    *,
    sub: str = "01",
    passed: bool = True,
    n_sections: int = 2,
) -> Path:
    """Render a report from the synthetic dataset and return the HTML path."""
    sections = [
        ReportSection(
            regressor="36-parameter",
            metrics=_metrics("36-parameter"),
            passed=passed,
            cleaned_bold=qc_dataset["cleaned_36"],
        )
    ]
    if n_sections == 2:
        sections.append(
            ReportSection(
                regressor="aCompCor",
                metrics=_metrics("aCompCor"),
                passed=not passed,
                cleaned_bold=qc_dataset["cleaned_acomp"],
            )
        )
    out = tmp_path / "out" / "report.html"
    generate_qc_report(
        sub=sub,
        ses="baseline",
        task="rest",
        run=1,
        sections=sections,
        template_brain_mask=qc_dataset["template_brain_mask"],
        bold_mask=qc_dataset["bold_mask"],
        brain_mask=qc_dataset["brain_mask"],
        bold_to_anat_matrix=qc_dataset["xfm"],
        motion_params=qc_dataset["motion_params"],
        rms_rel=qc_dataset["rms_rel"],
        out_path=out,
        mni_brain_mask=qc_dataset["template_brain_mask"],
        func_template=qc_dataset["func_template"],
    )
    return out


class TestGenerateQcReport:
    """End-to-end rendering of the HTML report on synthetic data."""

    def test_report_written_and_complete(
        self, qc_dataset: dict[str, Path], tmp_path: Path
    ) -> None:
        """HTML exists with all sections, thresholds, and valid SVGs."""
        out = _generate(qc_dataset, tmp_path)
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        for needle in (
            "QC Summary",
            "Coregistration",
            "Normalization",
            "Motion traces",
            "36-parameter",
            "aCompCor",
            "0.2",
            "0.8",
            "Dice",
            "Jaccard",
            "Cross Corr",
            "Coverage",
            "Final mean DVARS",
            "QC FAILED",
        ):
            assert needle in html, needle
        svgs = re.findall(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)", html)
        # 3 shared panels + one carpet per regressor
        assert len(svgs) == 5
        assert base64.b64decode(svgs[0])[:5] == _SVG_PREFIX
        assert "http://" not in html
        assert "https://" not in html
        # Only the report file is written to the output directory.
        assert {p.name for p in out.parent.iterdir()} == {"report.html"}

    def test_single_regressor_report(
        self, qc_dataset: dict[str, Path], tmp_path: Path
    ) -> None:
        """A single-regressor report has exactly one carpet section."""
        html = _generate(qc_dataset, tmp_path, n_sections=1).read_text(encoding="utf-8")
        assert len(re.findall(r"data:image/svg\+xml;base64,", html)) == 4
        assert html.count('<section id="reg-') == 1

    def test_passed_banner(self, qc_dataset: dict[str, Path], tmp_path: Path) -> None:
        """All-pass sections produce the PASSED banner."""
        html = _generate(qc_dataset, tmp_path, passed=True, n_sections=1).read_text(
            encoding="utf-8"
        )
        assert "QC PASSED" in html

    def test_failed_banner(self, qc_dataset: dict[str, Path], tmp_path: Path) -> None:
        """Any failed section produces the FAILED banner."""
        # Two sections with one failing (section 2 is the inverse of section 1).
        html = _generate(qc_dataset, tmp_path, passed=True).read_text(encoding="utf-8")
        assert "QC FAILED" in html

    def test_labels_are_html_escaped(
        self, qc_dataset: dict[str, Path], tmp_path: Path
    ) -> None:
        """Subject labels cannot inject raw HTML into the report."""
        html = _generate(
            qc_dataset, tmp_path, sub="01<script>alert(1)</script>"
        ).read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestSharedUtilities:
    """Direct checks on the public rendering primitives."""

    def test_figure_to_svg(self) -> None:
        """figure_to_svg returns a decodable base64 SVG data URI."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1).plot([1, 2], [3, 4])
        uri = figure_to_svg(fig)
        assert uri.startswith("data:image/svg+xml;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:5] == _SVG_PREFIX

    def test_metric_rows(self) -> None:
        """metric_rows returns the ten numeric run-level cells."""
        rows = metric_rows(_metrics("36-parameter"))
        assert len(rows) == 10
        assert all(re.fullmatch(r"[\d.]+", v) for v in rows)

    def test_render_lightbox_with_overlay(self) -> None:
        """render_lightbox draws a background mosaic plus a mask overlay."""
        fig = plt.figure()
        data = _sphere()
        render_lightbox(
            fig.add_subplot(1, 1, 1),
            data,
            overlays=[(data, "#ffb74d", 0.9)],
            title="overlay",
        )
        uri = figure_to_svg(fig)
        assert base64.b64decode(uri.split(",", 1)[1])[:5] == _SVG_PREFIX

    def test_warp_mask_fsl_internal_units(self, tmp_path: Path) -> None:
        """A FSL .mat (internal voxel units) lands the blob in T1w space.

        FSL matrices map in->ref voxels in FSL's spacing-scaled, radiological
        internal coordinates -- they are NOT RAS world affines. This catches
        the regression of treating the .mat as a world-mm matrix, which would
        place the blob at the wrong location (or out of volume).
        """
        bold_affine = np.array(
            [[-2, 0, 0, 40], [0, 2, 0, -40], [0, 0, 2, -40], [0, 0, 0, 1]],
            dtype=float,
        )
        t1w_affine = np.array(
            [[-1, 0, 0, 50], [0, 1, 0, -50], [0, 0, 1, -50], [0, 0, 0, 1]],
            dtype=float,
        )
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[8, 8, 8] = 1.0
        bold_mask = _write_nifti(
            tmp_path / "bold_mask.nii.gz", mask, affine=bold_affine
        )
        brain_mask = _write_nifti(
            tmp_path / "brain_mask.nii.gz", np.zeros((48, 48, 48)), affine=t1w_affine
        )
        # World transform BOLD->T1w is a +10 mm x-shift; the equivalent FSL
        # internal matrix (in->ref, spacing-scaled coordinates) is:
        xfm = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 10], [0, 0, 1, 10], [0, 0, 0, 1]], dtype=float
        )
        xfm_path = tmp_path / "xfm.txt"
        np.savetxt(xfm_path, xfm)

        warped = _warp_mask(bold_mask, brain_mask, xfm_path)
        # Blob world (24, -24, -24) + 10 mm x -> (34, -24, -24) -> index (16, 26, 26).
        assert warped[16, 26, 26] > 0.5
        # Misreading the .mat as a RAS world affine lands at (26, 36, 36).
        assert warped[26, 36, 36] == 0

    def test_render_carpet(self) -> None:
        """render_carpet handles a 4-D volume without error."""
        rng = np.random.default_rng(0)
        data4d = rng.random((*_SHAPE3, _N_VOLS)).astype(np.float32)
        fig = plt.figure()
        render_carpet(fig, data4d, _sphere())
        uri = figure_to_svg(fig)
        assert base64.b64decode(uri.split(",", 1)[1])[:5] == _SVG_PREFIX
