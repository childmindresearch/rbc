"""HTML QC report generation.

Renders a self-contained HTML quality-control report from preprocessing
outputs: a pass/fail summary with numeric thresholds, BOLD-to-T1w
coregistration and template-normalization overlays, motion traces
(FD, DVARS, RMS over time), and carpet plots of the cleaned BOLD.

All figures are rendered with the matplotlib Agg backend and
base64-embedded as PNG data URIs, so the result is a single
self-contained HTML document that opens offline from ``file://``.

The rendering helpers (:func:`render_lightbox`,
:func:`render_motion_parameters`, :func:`render_displacement_traces`,
:func:`render_carpet`, :func:`figure_to_png`, :func:`section_header`,
:func:`metric_rows`) are intentionally public so that future report
types (e.g. longitudinal QC) can reuse the same layout primitives.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nibabel.processing import resample_from_to

from rbc.core.qc.dvars import dvars
from rbc.core.qc.motion import framewise_displacement_jenkinson
from rbc_resources import REGISTRATION_TEMPLATES

matplotlib.use("Agg", force=False)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from rbc.core.qc.xcp import XCPQCMetrics

    Overlay = tuple[np.ndarray, str, float]

# -- Dark theme (shared with scripts/visualize_pipeline.py) --
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#4fc3f7"
GRID_COLOR = "#444444"
SPINE_COLOR = "#666666"

# -- Panel colors --
BOLD_MASK_COLOR = "#ffb74d"
TEMPLATE_MASK_COLOR = "#4fc3f7"
FD_COLOR = "#ef5350"
DVARS_COLORS = ("#4fc3f7", "#fdd835", "#66bb6a", "#ce93d8", "#ffb74d")

# -- QC pass/fail thresholds (mirror passes_rbc_qc in rbc.core.qc.xcp) --
FD_THRESHOLD_MM = 0.2
NORM_CROSS_CORR_THRESHOLD = 0.8

# -- Rendering limits --
MAX_BG_VOLUMES = 32
MAX_CARPET_ELEMENTS = 20_000_000
MOSAIC_WIDTH = 1600
N_SLICES = 7
PANEL_ALPHA = 0.35

SUMMARY_CSS: str = """
:root { color-scheme: dark; }
body { margin: 0 auto; max-width: 1100px; padding: 24px; background: #1a1a2e;
       color: #e0e0e0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.5rem; margin: 0 0 8px; }
h2 { border-bottom: 1px solid #4fc3f7; padding-bottom: 4px; font-size: 1.1rem;
     color: #4fc3f7; margin-top: 0; }
.banner { display: inline-block; padding: 6px 14px; border-radius: 6px;
          font-weight: 700; }
.banner.passed { background: #1b3a1b; color: #66bb6a; border: 1px solid #66bb6a; }
.banner.failed { background: #3a1b1b; color: #ef5350; border: 1px solid #ef5350; }
.thresholds { color: #9e9e9e; font-size: 0.9rem; }
section { margin-top: 28px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 6px 10px; border: 1px solid #444444; text-align: left;
         font-size: 0.9rem; }
thead th { background: #333355; color: #4fc3f7; }
tbody tr { background: #262640; }
td.pass { color: #66bb6a; font-weight: 700; }
td.fail { color: #ef5350; font-weight: 700; }
figure { margin: 0 0 16px; background: #0d0d1a; border: 1px solid #333333;
         padding: 8px; }
img { max-width: 100%; height: auto; display: block; }
figcaption { color: #9e9e9e; font-size: 0.85rem; margin-top: 6px; }
footer { margin-top: 32px; color: #666666; font-size: 0.8rem; }
"""


@dataclass(frozen=True)
class ReportSection:
    """Input data for one regressor's report section.

    Attributes:
        regressor: Regressor label (e.g. ``"36-parameter"``).
        metrics: XCP-style QC metrics row for this regressor.
        passed: Whether this run passes the RBC QC thresholds.
        cleaned_bold: Path to the post-denoising BOLD in template space.
    """

    regressor: str
    metrics: XCPQCMetrics
    passed: bool
    cleaned_bold: Path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a ``#rrggbb`` string to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _robust_vmax(data: np.ndarray) -> float:
    """Return the 98th percentile of non-zero voxels (or 1.0 if empty)."""
    values = data[data > 0]
    return float(np.percentile(values, 98)) if values.size else 1.0


def _axial_slices(data: np.ndarray, n: int) -> list[int]:
    """Return *n* evenly spaced axial slice indices, skipping empty edges."""
    nz = data.shape[2]
    sums = np.sum(np.abs(data), axis=(0, 1))
    nonzero = np.nonzero(sums > 0)[0]
    if not nonzero.size:
        return list(np.linspace(0, nz - 1, n, dtype=int))
    lo, hi = int(nonzero[0]), int(nonzero[-1])
    margin = max(1, int((hi - lo) * 0.05))
    lo = min(lo + margin, hi)
    hi = max(hi - margin, lo)
    return list(np.linspace(lo, hi, n, dtype=int))


def _resample_mosaic(mosaic: np.ndarray, target_w: int = MOSAIC_WIDTH) -> np.ndarray:
    """Resample a mosaic to a fixed pixel width, preserving aspect ratio."""
    h, w = mosaic.shape
    target_h = max(1, int(target_w * h / w))
    row_idx = np.clip((np.arange(target_h) * h / target_h).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(target_w) * w / target_w).astype(int), 0, w - 1)
    return mosaic[np.ix_(row_idx, col_idx)]


def _build_mosaic(
    data: np.ndarray,
    n: int,
    slices: Sequence[int] | None = None,
) -> np.ndarray:
    """Build an axial-slice mosaic resampled to standard dimensions."""
    if slices is None:
        slices = _axial_slices(data, n)
    panels = [data[:, :, z].T for z in slices]
    return _resample_mosaic(np.concatenate(panels, axis=1))


def _mask_contour(binary: np.ndarray) -> np.ndarray:
    """Return a 1-px edge mask from a 2-D binary array via gradients."""
    dx = np.diff(binary, axis=1, prepend=0)
    dy = np.diff(binary, axis=0, prepend=0)
    return (np.abs(dx) + np.abs(dy) > 0).astype(np.float32)


def _load_data(path: Path) -> np.ndarray:
    """Load a NIfTI file as a float32 array (memory-mapped if possible)."""
    img = nib.nifti1.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def _warp_mask(mask_path: Path, reference_path: Path, xfm_path: Path) -> np.ndarray:
    """Warp a mask into a reference image's space with an affine matrix.

    Args:
        mask_path: Source mask NIfTI.
        reference_path: NIfTI defining the target grid.
        xfm_path: 4x4 affine (source -> reference) text matrix.

    Returns:
        The warped mask data resampled onto the reference grid.
    """
    mask_img = nib.nifti1.load(mask_path)
    ref_img = nib.nifti1.load(reference_path)
    xfm = np.loadtxt(xfm_path)
    warped = nib.Nifti1Image(mask_img.get_fdata(), ref_img.affine @ xfm)
    warped = resample_from_to(warped, ref_img, order=0)
    return warped.get_fdata()


def _style_axes(ax: Axes) -> None:
    """Apply the shared dark theme to a line-plot axis."""
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.grid(visible=True, alpha=0.2, color=GRID_COLOR)


def _id_label(label: str) -> str:
    """Return an HTML-id-safe rendering of a regressor label."""
    return "".join(ch if ch.isalnum() else "-" for ch in label)


# ---------------------------------------------------------------------------
# Public rendering primitives (reused by future report types)
# ---------------------------------------------------------------------------


def figure_to_png(fig: plt.Figure, *, dpi: int = 100) -> str:
    """Render a matplotlib figure to a base64 PNG data URI.

    Args:
        fig: Figure to render (closed after rendering).
        dpi: Output resolution.

    Returns:
        A ``data:image/png;base64,...`` URI suitable for inline HTML.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def section_header(title: str) -> str:
    """Return an HTML ``<h2>`` section header with an escaped title.

    Args:
        title: Section title text.

    Returns:
        A complete HTML heading element.
    """
    return f"<h2>{html.escape(title)}</h2>"


def metric_rows(metrics: XCPQCMetrics, *, passed: bool) -> list[str]:
    """Return the formatted summary-table cells for one regressor.

    Args:
        metrics: XCP-style QC metrics row.
        passed: Whether this run passes the RBC QC thresholds.

    Returns:
        Cell values in column order: mean FD (mm), censored volumes,
        final mean DVARS, normalization cross-correlation, status.
    """
    return [
        f"{metrics.meanFD:.4f}",
        str(metrics.nVolCensored),
        f"{metrics.meanDVFinal:.4f}",
        f"{metrics.normCrossCorr:.4f}",
        "PASS" if passed else "FAIL",
    ]


def render_lightbox(
    ax: Axes,
    data3d: np.ndarray,
    *,
    overlays: Sequence[Overlay] = (),
    title: str = "",
) -> None:
    """Draw a mosaic of axial slices with optional flat mask overlays.

    Args:
        ax: Matplotlib axes to render into.
        data3d: 3-D background volume array.
        overlays: Tuples of ``(mask_data, hex_color, outline_alpha)``;
            the mask is drawn as a flat translucent overlay and, when
            *outline_alpha* is greater than zero, its edges are stroked
            at that opacity.
        title: Panel title text.
    """
    slices = _axial_slices(data3d, N_SLICES)
    bg_mosaic = _build_mosaic(data3d, N_SLICES, slices=slices)
    vmax = max(_robust_vmax(data3d), 1e-8)

    ax.imshow(
        bg_mosaic,
        cmap="gray",
        vmin=0,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )

    for mask_data, hex_color, outline in overlays:
        mask_mosaic = _build_mosaic(mask_data, N_SLICES, slices=slices)
        binary = (mask_mosaic > 0.5).astype(np.float32)
        r, g, b = _hex_to_rgb(hex_color)
        cmap_solid = ListedColormap([(0, 0, 0, 0), (r, g, b, PANEL_ALPHA)])
        ax.imshow(
            binary,
            cmap=cmap_solid,
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        if outline > 0:
            edge = _mask_contour(binary)
            cmap_edge = ListedColormap([(0, 0, 0, 0), (r, g, b, outline)])
            ax.imshow(
                edge,
                cmap=cmap_edge,
                vmin=0,
                vmax=1,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            )

    ax.set_facecolor("black")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_COLOR, fontweight="bold", pad=4)


def render_motion_parameters(ax: Axes, motion_params: np.ndarray) -> None:
    """Plot the six MCFLIRT motion-parameter traces (vs. first volume).

    Args:
        ax: Matplotlib axes to render into.
        motion_params: ``(T, 6)`` array with columns
            ``[roll, pitch, yaw, dS, dL, dP]`` (degrees, mm).
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)
    for i, label in enumerate(("roll", "pitch", "yaw")):
        ax.plot(np.degrees(motion_params[:, i]), label=label, linewidth=0.8)
    for i, label in enumerate(("dS", "dL", "dP"), start=3):
        ax.plot(motion_params[:, i], label=label, linewidth=0.8)

    ax.set_xlabel("Volume", fontsize=8)
    ax.set_ylabel("Rotation (deg) / translation (mm)", fontsize=8)
    ax.set_title("Motion parameters", fontsize=10, fontweight="bold")
    _style_axes(ax)
    ax.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#333333",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )


def render_displacement_traces(
    ax: Axes,
    *,
    fd: np.ndarray,
    rms: np.ndarray,
    dvars_curves: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Plot framewise displacement, relative RMS, and DVARS over time.

    Args:
        ax: Matplotlib axes to render into.
        fd: Framewise displacement per volume (length T).
        rms: MCFLIRT relative RMS per frame pair (length T-1).
        dvars_curves: Optional mapping of regressor label to per-volume
            DVARS (length T); one color-coded curve per entry.
    """
    fd = np.asarray(fd, dtype=np.float64).ravel()
    rms = np.asarray(rms, dtype=np.float64).ravel()

    ax.plot(np.arange(len(fd)), fd, color=FD_COLOR, linewidth=0.8, label="FD (mm)")
    ax.plot(
        np.arange(1, len(fd)),
        rms,
        color="#9e9e9e",
        linewidth=0.8,
        label="Rel. RMS (mm)",
    )
    if dvars_curves:
        for i, (label, curve) in enumerate(dvars_curves.items()):
            color = DVARS_COLORS[i % len(DVARS_COLORS)]
            ax.plot(
                np.arange(len(curve)),
                curve,
                color=color,
                linewidth=0.8,
                label=f"DVARS: {label}",
            )

    ax.axhline(FD_THRESHOLD_MM, color=TEXT_COLOR, ls="--", alpha=0.5)
    ax.set_xlabel("Volume", fontsize=8)
    ax.set_ylabel("Motion", fontsize=8)
    ax.set_title(
        f"FD / RMS / DVARS (threshold: {FD_THRESHOLD_MM} mm)",
        fontsize=10,
        fontweight="bold",
    )
    _style_axes(ax)
    ax.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#333333",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )


def render_carpet(ax: Axes, data4d: np.ndarray, mask3d: np.ndarray) -> None:
    """Plot an in-mask voxel-by-time carpet of 4-D BOLD data.

    Args:
        ax: Matplotlib axes to render into.
        data4d: 4-D array ``(X, Y, Z, T)``.
        mask3d: 3-D boolean-compatible brain mask.
    """
    voxels = np.ascontiguousarray(data4d[mask3d > 0], dtype=np.float32)  # (V, T)
    v, t = voxels.shape
    stride = 1
    while v * t > MAX_CARPET_ELEMENTS and stride * 2 <= v:
        stride *= 2
    if stride > 1:
        voxels = voxels[::stride]

    mu = voxels.mean(axis=1, keepdims=True)
    sd = voxels.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (voxels - mu) / sd

    ax.imshow(
        z.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
        vmin=-3,
        vmax=3,
    )
    ax.set_xlabel("Volume", fontsize=8)
    ax.set_ylabel("In-mask voxel (z-scored)", fontsize=8)
    ax.set_title("Cleaned BOLD carpet", fontsize=10, fontweight="bold")
    _style_axes(ax)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def generate_qc_report(
    *,
    sub: str,
    ses: str,
    task: str,
    run: int,
    sections: Sequence[ReportSection],
    template_bold: Path,
    template_brain_mask: Path,
    bold_mask: Path,
    brain_mask: Path,
    bold_to_anat_matrix: Path,
    motion_params: Path,
    rms_rel: Path,
    out_path: Path,
    mni_brain_mask: Path | None = None,
) -> Path:
    """Render a self-contained HTML QC report for one functional run.

    Args:
        sub: Subject label (without ``sub-``).
        ses: Session label (empty string for single-session data).
        task: Task label.
        run: Run number.
        sections: One section per regressor, in display order.
        template_bold: Pre-denoising BOLD in template space (4-D).
        template_brain_mask: Brain mask warped to template space.
        bold_mask: Native-space BOLD brain mask.
        brain_mask: Anatomical (T1w-space) brain mask.
        bold_to_anat_matrix: 4x4 BOLD-to-T1w affine matrix file.
        motion_params: Six-column MCFLIRT motion ``.1D`` file.
        rms_rel: MCFLIRT relative RMS ``.rms`` file.
        out_path: Destination HTML file (parent dirs created if needed).
        mni_brain_mask: MNI standard brain mask for the normalization
            panel (default: :data:`REGISTRATION_TEMPLATES.brain_mask_2mm`).

    Returns:
        The report path (same as *out_path*).
    """
    esc = html.escape
    if mni_brain_mask is None:
        mni_brain_mask = REGISTRATION_TEMPLATES.brain_mask_2mm

    # -- Shared display arrays --
    rms = np.loadtxt(rms_rel)
    fd = framewise_displacement_jenkinson(rms)
    mparams = np.loadtxt(motion_params)

    tmpl_ref = nib.nifti1.load(template_bold)
    tmpl_bg = _load_data(template_bold)
    if tmpl_bg.ndim == 4:
        tmpl_bg = tmpl_bg[..., :MAX_BG_VOLUMES].mean(axis=-1)
    tmpl_mask_data = _load_data(template_brain_mask)

    mni_mask_img = nib.nifti1.load(mni_brain_mask)
    if mni_mask_img.shape[:3] != tmpl_ref.shape[:3]:
        mni_mask_img = resample_from_to(mni_mask_img, tmpl_ref, order=0)
    mni_mask_data = mni_mask_img.get_fdata()

    # -- Subject/session identifier --
    subject = f"sub-{esc(sub)}"
    if ses:
        subject += f"_ses-{esc(ses)}"
    subject += f"_task-{esc(task)}"
    if run:
        subject += f"_run-{run}"

    overall_passed = all(section.passed for section in sections)
    banner_class = "passed" if overall_passed else "failed"
    banner_text = "QC PASSED" if overall_passed else "QC FAILED"
    criteria = (
        f"median FD &le; {FD_THRESHOLD_MM} mm AND "
        f"normCrossCorr &ge; {NORM_CROSS_CORR_THRESHOLD}"
    )

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{subject} QC report</title>",
        f"<style>{SUMMARY_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{subject}</h1>",
        f'<div class="banner {banner_class}">{banner_text}</div>',
        f'<p class="thresholds">Pass criteria: {criteria}</p>',
        "</header>",
        '<section id="summary">',
        section_header("QC Summary"),
        "<table>",
        "<thead>",
        "<tr><th>Regressor</th><th>Mean FD (mm)</th><th>Vols censored</th>"
        "<th>Mean DVARS (final)</th><th>Norm cross-corr</th>"
        "<th>Criteria</th><th>Status</th></tr>",
        "</thead>",
        "<tbody>",
    ]

    for section in sections:
        values = metric_rows(section.metrics, passed=section.passed)
        status_class = "pass" if section.passed else "fail"
        cells = "".join(f"<td>{esc(v)}</td>" for v in values[:-1])
        parts.append(
            f"<tr><td>{esc(section.regressor)}</td>{cells}"
            f"<td>{criteria}</td>"
            f'<td class="{status_class}">{esc(values[-1])}</td></tr>'
        )
    parts += ["</tbody>", "</table>", "</section>"]

    # -- Coregistration: BOLD mask warped into T1w space --
    warped_bold_mask = _warp_mask(bold_mask, brain_mask, bold_to_anat_matrix)
    brain_mask_data = _load_data(brain_mask)
    fig = plt.figure(figsize=(14, 4))
    fig.set_facecolor(BG_COLOR)
    render_lightbox(
        fig.add_subplot(1, 1, 1),
        brain_mask_data,
        overlays=[(warped_bold_mask, BOLD_MASK_COLOR, 0.9)],
        title="BOLD brain mask (warped to T1w) on anatomical brain mask",
    )
    coreg_png = figure_to_png(fig)
    parts += [
        '<section id="coreg">',
        section_header("Coregistration: BOLD to T1w"),
        "<figure>",
        f'<img alt="Coregistration overlay" src="{coreg_png}">',
        "<figcaption>BOLD brain mask warped into T1w space (orange) "
        "on the anatomical brain mask.</figcaption>",
        "</figure>",
        "</section>",
    ]

    # -- Normalization: template brain mask vs. MNI standard --
    fig = plt.figure(figsize=(14, 4))
    fig.set_facecolor(BG_COLOR)
    render_lightbox(
        fig.add_subplot(1, 1, 1),
        tmpl_bg,
        overlays=[
            (tmpl_mask_data, TEMPLATE_MASK_COLOR, 0.5),
            (mni_mask_data, BOLD_MASK_COLOR, 0.9),
        ],
        title="Template brain mask on template BOLD",
    )
    norm_png = figure_to_png(fig)
    parts += [
        '<section id="norm">',
        section_header("Normalization: template registration"),
        "<figure>",
        f'<img alt="Normalization overlay" src="{norm_png}">',
        "<figcaption>Brain mask warped to template space (blue) and the "
        "MNI152 standard brain mask (orange outline) on template BOLD.</figcaption>",
        "</figure>",
        "</section>",
    ]

    # -- Motion: 6-parameter traces + FD / RMS / DVARS --
    dvars_curves: dict[str, np.ndarray] = {}
    for section in sections:
        data = _load_data(section.cleaned_bold)
        dvars_curves[section.regressor] = dvars(data, tmpl_mask_data)
        del data

    fig = plt.figure(figsize=(16, 4))
    fig.set_facecolor(BG_COLOR)
    render_motion_parameters(fig.add_subplot(1, 2, 1), mparams)
    render_displacement_traces(
        fig.add_subplot(1, 2, 2), fd=fd, rms=rms, dvars_curves=dvars_curves
    )
    motion_png = figure_to_png(fig)
    parts += [
        '<section id="motion">',
        section_header("Motion traces"),
        "<figure>",
        f'<img alt="Motion traces" src="{motion_png}">',
        "<figcaption>Motion parameters versus first volume, and framewise "
        "displacement, relative RMS, and per-regressor DVARS over time.</figcaption>",
        "</figure>",
        "</section>",
    ]

    # -- One carpet section per regressor --
    for section in sections:
        data = _load_data(section.cleaned_bold)
        fig = plt.figure(figsize=(14, 6))
        fig.set_facecolor(BG_COLOR)
        render_carpet(fig.add_subplot(1, 1, 1), data, tmpl_mask_data)
        carpet_png = figure_to_png(fig)
        del data
        label = section.regressor
        parts += [
            f'<section id="reg-{_id_label(label)}">',
            section_header(f"reg-{label} — cleaned BOLD"),
            "<figure>",
            f'<img alt="{esc(label)} carpet plot" src="{carpet_png}">',
            f"<figcaption>Carpet plot of the cleaned BOLD for {esc(label)}: "
            "in-mask voxels versus time (z-scored, &plusmn;3).</figcaption>",
            "</figure>",
            "</section>",
        ]

    parts += [
        "<footer>",
        f"<p>Generated by RBC. Pass criteria: {criteria}. "
        f"Regressor(s): {esc(', '.join(s.regressor for s in sections))}.</p>",
        "</footer>",
        "</body>",
        "</html>",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
