"""Longitudinal QC HTML report generation.

Renders a self-contained HTML quality-control report for one subject,
summarizing all sessions' alignment to the within-subject longitudinal
template: a per-run pass/fail summary, per-session registration overlays
on the template, a cross-session BOLD coverage map, and per-session
motion traces.

Builds on the cross-sectional report primitives
(``rbc.core.qc.report``) for the shared dark theme, lightbox rendering,
and SVG embedding, so the result is a single self-contained HTML
document that opens offline from ``file://``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from rbc.core.qc.motion import framewise_displacement_jenkinson
from rbc.core.qc.registration import DICE_THRESHOLD
from rbc.core.qc.report import (
    BG_COLOR,
    BOLD_MASK_COLOR,
    SUMMARY_CSS,
    TEMPLATE_MASK_COLOR,
    TEXT_COLOR,
    figure_to_svg,
    render_lightbox,
    section_header,
    style_axes,
    style_legend,
)
from rbc.core.qc.xcp import FD_THRESHOLD_MM

matplotlib.use("Agg", force=False)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from rbc.core.qc.registration import RegistrationQCMetrics

# One distinct color per session for the coverage panel (BOLD orange first).
SESSION_COLORS = ("#ffb74d", "#fdd835", "#66bb6a", "#ce93d8", "#4fc3f7")


@dataclass(frozen=True)
class ReportSection:
    """Input data for one session's report section.

    Attributes:
        ses: Session label.
        run: Run number.
        metrics: Registration overlap metrics for this run.
        passed: Whether this run passes the Dice threshold.
        anat_mask: Anatomical brain mask in longitudinal space.
        bold_mask: BOLD brain mask in longitudinal space.
        template: Longitudinal template (BOLD grid) for the background.
        rms_rel: MCFLIRT relative RMS ``.rms`` file for this run.
    """

    ses: str
    run: int | str
    metrics: RegistrationQCMetrics
    passed: bool
    anat_mask: Path
    bold_mask: Path
    template: Path
    rms_rel: Path


def _to_grid(img: nib.Nifti1Image, ref: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return *img* on the reference image's grid, resampled if needed.

    ``render_lightbox`` draws overlays with the background's slice
    indices, so every overlay must share the background's shape.
    """
    if img.shape[:3] != ref.shape[:3]:
        return resample_from_to(img, ref, order=0)
    return img


def _section_labels(sections: Sequence[ReportSection]) -> list[str]:
    """Return unique display labels, one per section, in order.

    Labels are the BIDS session name (``ses-<label>``); a repeated
    session is disambiguated by appending its run number.
    """
    labels: list[str] = []
    for section in sections:
        label = f"ses-{section.ses}"
        if label in labels:
            label = f"ses-{section.ses}_run{section.run}"
        labels.append(label)
    return labels


def render_motion_by_session(
    ax: Axes,
    *,
    fd_by_session: Mapping[str, np.ndarray],
) -> None:
    """Plot one framewise-displacement trace per session on a shared axis.

    Args:
        ax: Matplotlib axes to render into.
        fd_by_session: Mapping of session label to per-volume FD (mm).
    """
    for i, (label, fd) in enumerate(fd_by_session.items()):
        color = SESSION_COLORS[i % len(SESSION_COLORS)]
        ax.plot(np.arange(len(fd)), fd, color=color, linewidth=0.8, label=label)
    ax.axhline(FD_THRESHOLD_MM, color=TEXT_COLOR, ls="--", alpha=0.5)
    ax.set_xlabel("Volume", fontsize=8)
    ax.set_ylabel("FD (mm)", fontsize=8)
    ax.set_title(
        f"Framewise displacement by session (threshold: {FD_THRESHOLD_MM} mm)",
        fontsize=10,
        fontweight="bold",
    )
    style_axes(ax)
    style_legend(ax)


def generate_qc_report(
    *,
    sub: str,
    sessions: Sequence[ReportSection],
    out_path: Path,
) -> Path:
    """Render a self-contained HTML longitudinal QC report for one subject.

    Args:
        sub: Subject label (without ``sub-``).
        sessions: One section per processed (session, run), in display
            order.
        out_path: Destination HTML file (parent dirs created if needed).

    Returns:
        The report path (same as *out_path*).
    """
    esc = html.escape
    subject = f"sub-{esc(sub)}"
    labels = _section_labels(sessions)

    overall_passed = all(section.passed for section in sessions)
    banner_class = "passed" if overall_passed else "failed"
    banner_text = "QC PASSED" if overall_passed else "QC FAILED"
    criteria = f"Dice &ge; {DICE_THRESHOLD}"

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{subject} longitudinal QC report</title>",
        f"<style>{SUMMARY_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{subject} — longitudinal QC</h1>",
        f'<div class="banner {banner_class}">{banner_text}</div>',
        f'<p class="thresholds">Pass criteria: {criteria}</p>',
        "</header>",
        '<section id="summary">',
        section_header("Registration QC Summary"),
        "<table>",
        "<thead>",
        "<tr><th>Session</th><th>Run</th><th>Dice</th><th>Jaccard</th>"
        "<th>Coverage</th><th>Cross Corr</th><th>Status</th></tr>",
        "</thead>",
        "<tbody>",
    ]
    parts.extend(
        f"<tr><td>ses-{esc(section.ses)}</td><td>{esc(str(section.run))}</td>"
        f"<td>{section.metrics.dice:.3f}</td>"
        f"<td>{section.metrics.jaccard:.3f}</td>"
        f"<td>{section.metrics.coverage:.3f}</td>"
        f"<td>{section.metrics.cross_corr:.3f}</td>"
        f"<td>{'PASS' if section.passed else 'FAIL'}</td></tr>"
        for section in sessions
    )
    parts += ["</tbody>", "</table>", "</section>"]

    # -- Registration: each session's masks on the longitudinal template --
    parts += [
        '<section id="registration">',
        section_header("Registration: sessions on the longitudinal template"),
    ]
    for label, section in zip(labels, sessions, strict=True):
        tmpl_img = nib.nifti1.load(section.template)
        tmpl_bg = tmpl_img.get_fdata()
        anat_data = _to_grid(nib.nifti1.load(section.anat_mask), tmpl_img).get_fdata()
        bold_data = _to_grid(nib.nifti1.load(section.bold_mask), tmpl_img).get_fdata()
        fig = plt.figure(figsize=(14, 5))
        fig.set_facecolor(BG_COLOR)
        render_lightbox(
            fig.add_subplot(1, 1, 1),
            tmpl_bg,
            overlays=[
                (anat_data, TEMPLATE_MASK_COLOR, 0.5),
                (bold_data, BOLD_MASK_COLOR, 0.9),
            ],
            title=f"{label} on longitudinal template",
        )
        svg = figure_to_svg(fig)
        parts += [
            "<figure>",
            f'<img alt="{esc(label)} registration overlay" src="{svg}">',
            f"<figcaption>{esc(label)}: anatomical brain mask (blue) and "
            "BOLD brain mask (orange) in longitudinal template space.</figcaption>",
            "</figure>",
        ]
    parts.append("</section>")

    # -- Coverage: all sessions' BOLD masks on one template --
    bg_img = nib.nifti1.load(sessions[0].template)
    bg = bg_img.get_fdata()
    overlays: list[tuple[np.ndarray, str, float]] = []
    for i, section in enumerate(sessions):
        color = SESSION_COLORS[i % len(SESSION_COLORS)]
        bold_data = _to_grid(nib.nifti1.load(section.bold_mask), bg_img).get_fdata()
        overlays.append((bold_data, color, 0.9))
    fig = plt.figure(figsize=(14, 5))
    fig.set_facecolor(BG_COLOR)
    render_lightbox(
        fig.add_subplot(1, 1, 1),
        bg,
        overlays=overlays,
        title="BOLD coverage across sessions",
    )
    coverage_svg = figure_to_svg(fig)
    chips = " ".join(
        f'<span style="color:{SESSION_COLORS[i % len(SESSION_COLORS)]}">'
        "&#9632;</span> " + esc(labels[i])
        for i in range(len(sessions))
    )
    parts += [
        '<section id="coverage">',
        section_header("Coverage: BOLD across sessions"),
        "<figure>",
        f'<img alt="BOLD coverage across sessions" src="{coverage_svg}">',
        f"<figcaption>Longitudinal-space BOLD brain masks, one color per "
        f"session: {chips}.</figcaption>",
        "</figure>",
        "</section>",
    ]

    # -- Motion: one FD trace per session --
    fd_by_session: dict[str, np.ndarray] = {}
    medians: list[tuple[str, float, float]] = []
    for label, section in zip(labels, sessions, strict=True):
        rms = np.loadtxt(section.rms_rel)
        fd = framewise_displacement_jenkinson(rms)
        fd_by_session[label] = fd
        medians.append((label, float(np.median(fd)), float(np.median(rms))))
    fig = plt.figure(figsize=(14, 5))
    fig.set_facecolor(BG_COLOR)
    render_motion_by_session(fig.add_subplot(1, 1, 1), fd_by_session=fd_by_session)
    motion_svg = figure_to_svg(fig)
    parts += [
        '<section id="motion">',
        section_header("Motion by session"),
        "<table>",
        "<thead>",
        "<tr><th>Session</th><th>Median FD (mm)</th><th>Median rel. RMS (mm)</th></tr>",
        "</thead>",
        "<tbody>",
    ]
    for label, fd_med, rms_med in medians:
        parts.append(
            f"<tr><td>{esc(label)}</td><td>{fd_med:.3f}</td><td>{rms_med:.3f}</td></tr>"
        )
    parts += [
        "</tbody>",
        "</table>",
        "<figure>",
        f'<img alt="Framewise displacement by session" src="{motion_svg}">',
        "<figcaption>Framewise displacement per session with the shared "
        f"{FD_THRESHOLD_MM} mm reference line.</figcaption>",
        "</figure>",
        "</section>",
    ]

    parts += [
        "<footer>",
        "<p>Generated by RBC. "
        f"Session(s): {esc(', '.join(dict.fromkeys(labels)))}.</p>",
        "</footer>",
        "</body>",
        "</html>",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
