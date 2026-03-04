# /// script
# dependencies = ["matplotlib>=3.9", "nibabel>=5.0", "numpy>=1.26", "pyvista>=0.44"]
# requires-python = ">=3.12"
# ///
"""Visualize outputs from a full-pipeline e2e test run.

Reads the manifest written by ``tests/full_pipeline/conftest.py`` and generates
a multi-panel PNG report of the key anatomical, functional, QC, and metrics
outputs.

Uses pure matplotlib for rendering (lightbox mosaics, alpha mask overlays,
dark theme) and pyvista for 3D brain surface renders.

Usage::

    uv run scripts/visualize_pipeline.py                          # default manifest
    uv run scripts/visualize_pipeline.py path/to/.last_run.json   # custom path
    uv run scripts/visualize_pipeline.py --output report.png      # custom output
"""
# ruff: noqa: T201, ANN401, FBT003

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# -- Dark theme constants --
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#4fc3f7"
GRID_COLOR = "#444444"
SPINE_COLOR = "#666666"
SECTION_FONTSIZE = 12
FIG_WIDTH = 20
MOSAIC_WIDTH = 2000  # target pixel width for all lightbox mosaics

# Row heights
HEADER_HEIGHT = 0.5
LIGHTBOX_ROW_HEIGHT = 3.5
PLOT_ROW_HEIGHT = 2.4
BOTTOM_ROW_HEIGHT = 5.0

# Overlay colors (R, G, B tuples for RGBA construction)
WM_COLOR = "#4fc3f7"  # blue
GM_COLOR = "#ef5350"  # red
CSF_COLOR = "#66bb6a"  # green
MASK_COLOR = "#ffb74d"  # orange

# Cold-hot diverging colormap for stat maps
_COLD_HOT_COLORS = [
    (0.0, "#2196f3"),
    (0.25, "#64b5f6"),
    (0.5, "#000000"),
    (0.75, "#ef5350"),
    (1.0, "#f44336"),
]
COLD_HOT_CMAP = LinearSegmentedColormap.from_list(
    "cold_hot",
    [(pos, color) for pos, color in _COLD_HOT_COLORS],
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _load_vol(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (3D data, pixdim).

    For 4D images, returns the mean across time.
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    pixdim = np.abs(img.header.get_zooms()[:3])  # type: ignore[attr-defined]
    if data.ndim == 4:
        data = np.mean(data, axis=3)
    return data, np.asarray(pixdim, dtype=np.float64)


def _load_vol_std(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 4D NIfTI and return temporal standard deviation."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    pixdim = np.abs(img.header.get_zooms()[:3])  # type: ignore[attr-defined]
    if data.ndim == 4:
        data = np.std(data, axis=3)
    return data, np.asarray(pixdim, dtype=np.float64)


def _robust_vmax(data: np.ndarray) -> float:
    """Return 98th percentile of non-zero voxels."""
    values = data[data > 0]
    return float(np.percentile(values, 98)) if len(values) > 0 else 1.0


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _axial_slices(data: np.ndarray, n: int = 7) -> list[int]:
    """Return n evenly-spaced axial slice indices, skipping empty edges."""
    nz = data.shape[2]
    sums = np.sum(data, axis=(0, 1))
    nonzero = np.nonzero(sums > 0)[0]
    if len(nonzero) == 0:
        return list(np.linspace(0, nz - 1, n, dtype=int))
    lo, hi = int(nonzero[0]), int(nonzero[-1])
    margin = max(1, int((hi - lo) * 0.05))
    lo = min(lo + margin, hi)
    hi = max(hi - margin, lo)
    return list(np.linspace(lo, hi, n, dtype=int))


def _resample_mosaic(
    mosaic: np.ndarray,
    target_w: int = MOSAIC_WIDTH,
) -> np.ndarray:
    """Resample mosaic to a fixed pixel width, preserving aspect ratio.

    All mosaics get the same width so lightbox panels are visually
    uniform. Height scales proportionally to preserve the original
    data aspect ratio.
    """
    h, w = mosaic.shape
    target_h = max(1, int(target_w * h / w))
    row_idx = np.clip((np.arange(target_h) * h / target_h).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(target_w) * w / target_w).astype(int), 0, w - 1)
    return mosaic[np.ix_(row_idx, col_idx)]


def _build_mosaic(
    data: np.ndarray,
    n: int = 7,
) -> np.ndarray:
    """Build an axial-slice mosaic resampled to standard dimensions."""
    slices = _axial_slices(data, n)
    panels = [data[:, :, z].T for z in slices]
    mosaic = np.concatenate(panels, axis=1)
    return _resample_mosaic(mosaic)


def _render_lightbox(
    ax: plt.Axes,
    data: np.ndarray,
    n: int = 7,
    cmap: str = "gray",
    vmin: float = 0,
    vmax: float | None = None,
    title: str = "",
) -> None:
    """Draw a mosaic of n axial slices tiled side-by-side in a single Axes."""
    if vmax is None:
        vmax = _robust_vmax(data)

    mosaic = _build_mosaic(data, n)

    ax.imshow(
        mosaic,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_facecolor("black")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_COLOR, fontweight="bold", pad=4)


def _mask_contour(binary: np.ndarray) -> np.ndarray:
    """Return a 1-px edge mask from a 2D binary array using gradient magnitude."""
    dx = np.diff(binary.astype(np.float32), axis=1, prepend=0)
    dy = np.diff(binary.astype(np.float32), axis=0, prepend=0)
    return (np.abs(dx) + np.abs(dy) > 0).astype(np.float32)


def _render_mask_overlay(
    ax: plt.Axes,
    bg_data: np.ndarray,
    masks: list[tuple[np.ndarray, str, str]],
    alpha: float = 0.35,
    n: int = 7,
    title: str = "",
    *,
    outline: float = 0.0,
) -> None:
    """Render background lightbox with flat colored mask overlays.

    Args:
        ax: Matplotlib axes to render into.
        bg_data: 3D background volume array.
        masks: List of (mask_data, hex_color, label) tuples.
        alpha: Opacity for mask overlays.
        n: Number of axial slices.
        title: Panel title text.
        outline: If > 0, draw mask contour outlines at this opacity.
    """
    vmax = _robust_vmax(bg_data)
    bg_mosaic = _build_mosaic(bg_data, n)

    ax.imshow(
        bg_mosaic,
        cmap="gray",
        vmin=0,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )

    for mask_data, hex_color, _label in masks:
        mask_mosaic = _build_mosaic(mask_data, n)
        binary = (mask_mosaic > 0.5).astype(np.float32)

        r, g, b = _hex_to_rgb(hex_color)
        cmap_solid = ListedColormap([(0, 0, 0, 0), (r, g, b, alpha)])
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

    legend_masks = [(c, lbl) for _, c, lbl in masks if lbl]
    if legend_masks:
        handles = [
            Patch(facecolor=c, alpha=alpha, label=lbl) for c, lbl in legend_masks
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=7,
            framealpha=0.6,
            facecolor="#333333",
            edgecolor=SPINE_COLOR,
            labelcolor=TEXT_COLOR,
        )


def _render_stat_overlay(
    ax: plt.Axes,
    bg_data: np.ndarray,
    stat_data: np.ndarray,
    threshold: float = 2.0,
    n: int = 7,
    title: str = "",
) -> None:
    """Lightbox with thresholded stat map overlaid on background."""
    bg_vmax = _robust_vmax(bg_data)
    bg_mosaic = _build_mosaic(bg_data, n)
    stat_mosaic = _build_mosaic(stat_data, n)

    ax.imshow(
        bg_mosaic,
        cmap="gray",
        vmin=0,
        vmax=bg_vmax,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )

    # Compute stat range
    if np.any(stat_mosaic != 0):
        pct = float(np.percentile(np.abs(stat_mosaic[stat_mosaic != 0]), 98))
        stat_vmax = max(pct, 0.1)
    else:
        stat_vmax = 5.0
    masked_stat = np.where(np.abs(stat_mosaic) >= threshold, stat_mosaic, np.nan)

    im = ax.imshow(
        masked_stat,
        cmap=COLD_HOT_CMAP,
        vmin=-stat_vmax,
        vmax=stat_vmax,
        origin="lower",
        aspect="equal",
        alpha=0.85,
        interpolation="bilinear",
    )
    ax.set_facecolor("black")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_COLOR, fontweight="bold", pad=4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label("z-score", fontsize=7, color=TEXT_COLOR, labelpad=2)
    cbar.ax.tick_params(labelsize=6, colors=TEXT_COLOR)
    cbar.outline.set_edgecolor(SPINE_COLOR)


def _extract_surface(nifti_path: str | Path) -> Any | None:
    """Extract a smoothed surface mesh from a NIfTI mask.

    Returns a pyvista PolyData or None on failure.
    Imported lazily so pyvista is only required when called.
    """
    import pyvista as pv

    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    if data.ndim == 4:
        data = np.mean(data, axis=3)

    volume = (data > 0.5).astype(np.float32)
    affine = img.affine  # type: ignore[attr-defined]
    spacing = np.abs(np.diag(affine)[:3])

    grid = pv.ImageData(
        dimensions=volume.shape,
        spacing=spacing,
        origin=affine[:3, 3],
    )
    grid.point_data["values"] = volume.flatten(order="F")

    surface = grid.contour([0.5], scalars="values", method="marching_cubes")
    if surface.n_points == 0:
        return None

    return surface.smooth(n_iter=50, relaxation_factor=0.1)


def _render_3d_surface(
    surfaces: list[tuple[str | Path, str]],
    window_size: tuple[int, int] = (1200, 500),
    *,
    clip: str = "",
) -> np.ndarray | None:
    """Render one or more 3D surfaces using pyvista (offscreen).

    Args:
        surfaces: List of (nifti_path, hex_color) tuples.  Each NIfTI is
            treated as a mask (binarized at 0.5) and extracted via marching
            cubes.
        window_size: Pixel (width, height) for the offscreen render.
        clip: Clip axis name ("x", "y", or "z") for a midline clip plane
            that reveals interior structures.  Empty string disables.

    Returns:
        RGB array or ``None`` if rendering fails / pyvista unavailable.
    """
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True

        meshes: list[tuple[pv.PolyData, str]] = []
        for nifti_path, hex_color in surfaces:
            surface = _extract_surface(nifti_path)
            if surface is not None:
                meshes.append((surface, hex_color))

        if not meshes:
            return None

        # Compute center / extent from the union of all surfaces
        all_bounds = np.array([m.bounds for m, _ in meshes])
        lo = np.min(all_bounds[:, 0::2], axis=0)  # xmin, ymin, zmin
        hi = np.max(all_bounds[:, 1::2], axis=0)  # xmax, ymax, zmax
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo))
        dist = extent * 2.5

        # Per-view meshes: when clipping, each view gets the opposite
        # hemisphere so the camera always faces the exposed cross-section.
        origin = tuple(center)
        if clip:
            per_view = []
            for invert in (False, True):
                half = [
                    (m.clip(normal=clip, origin=origin, invert=invert), c)
                    for m, c in meshes
                ]
                per_view.append([(m, c) for m, c in half if m.n_points > 0])
        else:
            per_view = [meshes, meshes]

        if not any(per_view):
            return None

        # Two views: left 3/4 anterior, right 3/4 anterior
        cam_positions = [
            (
                center[0] - dist * 0.7,
                center[1] + dist * 0.7,
                center[2] + dist * 0.3,
            ),
            (
                center[0] + dist * 0.7,
                center[1] + dist * 0.7,
                center[2] + dist * 0.3,
            ),
        ]

        plotter = pv.Plotter(
            shape=(1, 2),
            off_screen=True,
            window_size=window_size,
        )
        plotter.set_background(BG_COLOR)

        for i, cam_pos in enumerate(cam_positions):
            plotter.subplot(0, i)
            for mesh, color in per_view[i]:
                plotter.add_mesh(
                    mesh,
                    color=color,
                    specular=0.3,
                    smooth_shading=True,
                    show_edges=False,
                )
            plotter.camera.position = cam_pos
            plotter.camera.focal_point = origin
            plotter.camera.up = (0, 0, 1)
            plotter.reset_camera()
            plotter.camera.zoom(1.5)

        img_arr = plotter.screenshot(return_img=True)
        plotter.close()
        return img_arr  # type: ignore[return-value]

    except Exception:
        return None


def _section_header(fig: plt.Figure, gs_row: Any, title: str) -> None:
    """Add a styled section header spanning the full row."""
    ax = fig.add_subplot(gs_row)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        0.4,
        title,
        fontsize=SECTION_FONTSIZE,
        fontweight="bold",
        color=ACCENT_COLOR,
        va="center",
        ha="left",
        family="monospace",
    )
    ax.axhline(0.15, color=ACCENT_COLOR, linewidth=0.8, xmin=0.0, xmax=1.0)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")


def _style_motion_ax(ax: plt.Axes) -> None:
    """Apply dark theme to a motion plot axis."""
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def _add_3d_panel(
    fig: plt.Figure,
    gs: GridSpec,
    row: int,
    render_img: np.ndarray | None,
    title_3d: str,
) -> plt.Axes:
    """Add a 3D surface panel (left col) and return the axes for the right panel.

    If *render_img* is None (pyvista unavailable), returns a full-width axes
    so the caller's lightbox fills the entire row.
    """
    if render_img is not None:
        ax_3d = fig.add_subplot(gs[row, :1])
        ax_3d.imshow(render_img)
        ax_3d.set_title(
            title_3d, fontsize=10, color=TEXT_COLOR, fontweight="bold", pad=4
        )
        ax_3d.set_facecolor(BG_COLOR)
        ax_3d.axis("off")
        return fig.add_subplot(gs[row, 1:])
    return fig.add_subplot(gs[row, :])


# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------


def plot_anatomical(manifest: dict, fig: plt.Figure, gs: GridSpec, row: int) -> int:
    """Plot brain extraction (3D + lightbox) and tissue segmentation.

    Returns the next row index.
    """
    anat = manifest["anat"]
    brain_data, _ = _load_vol(anat["brain"])

    # Row 1: 3D brain surface (left) + skull-stripped lightbox (right)
    brain_3d = _render_3d_surface([(anat["brain"], "#d4a574")])
    ax_lb = _add_3d_panel(fig, gs, row, brain_3d, "Brain surface")
    _render_lightbox(ax_lb, brain_data, title="Skull-stripped T1w")
    row += 1

    # Row 2: 3D tissue surfaces (left) + segmentation overlay lightbox (right)
    wm_data, _ = _load_vol(anat["wm_mask"])
    gm_data, _ = _load_vol(anat["gm_mask"])
    csf_data, _ = _load_vol(anat["csf_mask"])

    seg_3d = _render_3d_surface([(anat["wm_mask"], WM_COLOR)])
    ax_seg = _add_3d_panel(fig, gs, row, seg_3d, "WM surface")

    _render_mask_overlay(
        ax_seg,
        brain_data,
        masks=[
            (wm_data, WM_COLOR, "WM"),
            (gm_data, GM_COLOR, "GM"),
            (csf_data, CSF_COLOR, "CSF"),
        ],
        title="Tissue segmentation",
    )
    row += 1

    return row


def plot_registration(manifest: dict, fig: plt.Figure, gs: GridSpec, row: int) -> int:
    """Plot registration quality with 3D mask surfaces and lightbox overlays.

    Returns the next row index.
    """
    func = manifest["func"]
    template_brain_mask = manifest["template_brain_mask"]

    # Row: 3D BOLD mask surface (left) + mask overlay on native BOLD (right)
    bold_data, _ = _load_vol(func["skull_stripped_bold"])
    mask_data, _ = _load_vol(func["bold_mask"])

    native_3d = _render_3d_surface([(func["bold_mask"], MASK_COLOR)])
    ax_native = _add_3d_panel(fig, gs, row, native_3d, "BOLD mask surface")

    _render_mask_overlay(
        ax_native,
        bold_data,
        masks=[(mask_data, MASK_COLOR, "")],
        title="Coregistration: BOLD mask overlay",
        outline=0.9,
    )
    row += 1

    # Row: 3D template mask surface (left) + mask overlay on template BOLD (right)
    tmpl_data, _ = _load_vol(func["template_bold"])
    tmpl_mask_data, _ = _load_vol(template_brain_mask)

    tmpl_3d = _render_3d_surface([(template_brain_mask, MASK_COLOR)])
    ax_tmpl = _add_3d_panel(fig, gs, row, tmpl_3d, "Template mask surface")

    _render_mask_overlay(
        ax_tmpl,
        tmpl_data,
        masks=[(tmpl_mask_data, MASK_COLOR, "")],
        title="Normalization: brain mask overlay",
        outline=0.9,
    )
    row += 1

    return row


def plot_functional_bold(
    manifest: dict, fig: plt.Figure, gs: GridSpec, row: int
) -> int:
    """Plot template and cleaned BOLD lightboxes.

    Returns the next row index.
    """
    func = manifest["func"]

    # Template BOLD mean
    ax_tmpl = fig.add_subplot(gs[row, :])
    tmpl_data, _ = _load_vol(func["template_bold"])
    _render_lightbox(ax_tmpl, tmpl_data, title="Template BOLD (mean)")
    row += 1

    # Cleaned BOLD temporal std
    ax_clean = fig.add_subplot(gs[row, :])
    std_data, _ = _load_vol_std(func["cleaned_bold"])
    _render_lightbox(
        ax_clean,
        std_data,
        title="Cleaned BOLD (temporal std)",
        cmap="inferno",
    )
    row += 1

    return row


def plot_motion(manifest: dict, fig: plt.Figure, gs: GridSpec, row: int) -> int:
    """Plot motion parameters in dark-themed line plots.

    Returns the next row index.
    """
    func = manifest["func"]
    motion = np.loadtxt(func["motion_params"])
    rms_rel = np.loadtxt(func["rms_rel"])

    # Rotation
    ax_rot = fig.add_subplot(gs[row, 0])
    for i, label in enumerate(["rot_x", "rot_y", "rot_z"]):
        ax_rot.plot(np.degrees(motion[:, i]), label=label, linewidth=0.8)
    ax_rot.set_ylabel("Rotation (deg)", fontsize=8)
    ax_rot.set_xlabel("Volume", fontsize=8)
    ax_rot.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#333333",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )
    ax_rot.set_title("Rotation", fontsize=9, fontweight="bold")
    _style_motion_ax(ax_rot)

    # Translation
    ax_trans = fig.add_subplot(gs[row, 1])
    for i, label in enumerate(["trans_x", "trans_y", "trans_z"], start=3):
        ax_trans.plot(motion[:, i], label=label, linewidth=0.8)
    ax_trans.set_ylabel("Translation (mm)", fontsize=8)
    ax_trans.set_xlabel("Volume", fontsize=8)
    ax_trans.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#333333",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )
    ax_trans.set_title("Translation", fontsize=9, fontweight="bold")
    _style_motion_ax(ax_trans)

    # RMS displacement
    ax_rms = fig.add_subplot(gs[row, 2])
    ax_rms.plot(rms_rel, color="#ffb74d", linewidth=0.8)
    ax_rms.axhline(0.2, color="#ef5350", ls="--", alpha=0.7, label="0.2 mm threshold")
    ax_rms.set_ylabel("RMS (mm)", fontsize=8)
    ax_rms.set_xlabel("Volume", fontsize=8)
    ax_rms.set_title("Relative RMS displacement", fontsize=9, fontweight="bold")
    ax_rms.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#333333",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )
    _style_motion_ax(ax_rms)

    row += 1
    return row


def plot_metrics_maps(manifest: dict, fig: plt.Figure, gs: GridSpec, row: int) -> int:
    """Plot ALFF, fALFF, and ReHo z-scored stat overlays.

    Returns the next row index.
    """
    metrics = manifest["metrics"]
    func = manifest["func"]
    bg_data, _ = _load_vol(func["template_bold"])

    for i, (key, label) in enumerate(
        [
            ("alff_zscored", "ALFF (z-scored)"),
            ("falff_zscored", "fALFF (z-scored)"),
            ("reho_zscored", "ReHo (z-scored)"),
        ]
    ):
        ax = fig.add_subplot(gs[row, i])
        stat_data, _ = _load_vol(metrics[key])
        _render_stat_overlay(ax, bg_data, stat_data, threshold=2.0, title=label)

    row += 1
    return row


def plot_correlation_matrix(manifest: dict, ax: plt.Axes) -> None:
    """Plot the FC correlation matrix with dark theme."""
    corr = np.loadtxt(manifest["metrics"]["correlation_matrix"], delimiter="\t")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(
        "FC correlation matrix",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.set_xlabel("ROI", fontsize=8, color=TEXT_COLOR, labelpad=2)
    ax.set_ylabel("ROI", fontsize=8, color=TEXT_COLOR, labelpad=2)
    ax.tick_params(labelsize=6, colors=TEXT_COLOR, pad=1)
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Pearson r", fontsize=7, color=TEXT_COLOR, labelpad=2)
    cbar.ax.tick_params(labelsize=6, colors=TEXT_COLOR)
    cbar.outline.set_edgecolor(SPINE_COLOR)


def plot_qc(manifest: dict, ax: plt.Axes) -> None:
    """Render QC metrics as a summary table with dark theme."""
    qc = manifest["qc"]
    metrics = qc["metrics"]
    passed = qc["passed"]

    groups = [
        (
            "Motion",
            [
                ("meanFD", "Mean FD"),
                ("relMeansRMSMotion", "Mean RMS"),
                ("relMaxRMSMotion", "Max RMS"),
                ("nVolCensored", "Vols censored"),
            ],
        ),
        (
            "DVARS",
            [
                ("meanDVInit", "Mean DV (init)"),
                ("meanDVFinal", "Mean DV (final)"),
                ("motionDVCorrInit", "Motion-DV r (init)"),
                ("motionDVCorrFinal", "Motion-DV r (final)"),
            ],
        ),
        (
            "Coregistration",
            [
                ("coregDice", "Dice"),
                ("coregJaccard", "Jaccard"),
                ("coregCrossCorr", "Cross-corr"),
                ("coregCoverage", "Coverage"),
            ],
        ),
        (
            "Normalization",
            [
                ("normDice", "Dice"),
                ("normJaccard", "Jaccard"),
                ("normCrossCorr", "Cross-corr"),
                ("normCoverage", "Coverage"),
            ],
        ),
    ]

    cell_text = []
    for group_name, keys in groups:
        cell_text.append([f"  {group_name}", ""])
        for key, label in keys:
            val = metrics[key]
            cell_text.append(
                [
                    f"    {label}",
                    f"{val:.4f}" if isinstance(val, float) else str(val),
                ]
            )

    ax.axis("off")
    status = "PASSED" if passed else "FAILED"
    color = "#66bb6a" if passed else "#ef5350"
    ax.set_title(
        f"QC Summary \u2014 {status}",
        fontsize=11,
        fontweight="bold",
        color=color,
        loc="left",
    )
    ax.set_facecolor(BG_COLOR)

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="upper center",
        cellLoc="left",
        colWidths=[0.35, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Style all cells for dark theme
    for cell in table.get_celld().values():
        cell.set_facecolor("#262640")
        cell.set_edgecolor(SPINE_COLOR)
        cell.set_text_props(color=TEXT_COLOR)

    # Style header row
    for c in range(2):
        cell = table[0, c]
        cell.set_facecolor("#333355")
        cell.set_text_props(fontweight="bold", color=ACCENT_COLOR)

    # Style group headers
    row_idx = 1
    for _group_name, keys in groups:
        for col in range(2):
            cell = table[row_idx, col]
            cell.set_facecolor("#2a2a4a")
            cell.set_text_props(fontweight="bold", color=ACCENT_COLOR)
        row_idx += 1 + len(keys)


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    """Load the JSON manifest from a full-pipeline test run."""
    if not path.exists():
        print(f"Manifest not found: {path}", file=sys.stderr)
        print("Run the full-pipeline tests first:", file=sys.stderr)
        print("  uv run pytest tests/full_pipeline/ -v", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def build_report(manifest: dict, output: Path) -> None:
    """Build a multi-panel visualization report.

    This is the public API called by visualize_cpac.py.
    """
    has_metrics = "metrics" in manifest
    has_qc = "qc" in manifest

    heights: list[float] = []
    labels: list[str] = []

    # -- Anatomical: header + brain lightbox + segmentation --
    heights += [HEADER_HEIGHT, LIGHTBOX_ROW_HEIGHT, LIGHTBOX_ROW_HEIGHT]
    labels += ["anat_hdr", "anat_brain", "anat_seg"]

    # -- Registration: header + native overlay + template overlay --
    heights += [HEADER_HEIGHT, LIGHTBOX_ROW_HEIGHT, LIGHTBOX_ROW_HEIGHT]
    labels += ["reg_hdr", "reg_native", "reg_tmpl"]

    # -- Functional: header + template BOLD + cleaned BOLD + motion plots --
    heights += [
        HEADER_HEIGHT,
        LIGHTBOX_ROW_HEIGHT,
        LIGHTBOX_ROW_HEIGHT,
        PLOT_ROW_HEIGHT,
    ]
    labels += ["func_hdr", "func_tmpl", "func_clean", "motion"]

    # -- Metrics --
    if has_metrics:
        heights += [HEADER_HEIGHT, LIGHTBOX_ROW_HEIGHT]
        labels += ["met_hdr", "met_maps"]
        if has_qc:
            heights += [BOTTOM_ROW_HEIGHT]
            labels += ["bottom"]
        else:
            heights += [3.5]
            labels += ["bottom"]
    elif has_qc:
        heights += [HEADER_HEIGHT, 4.0]
        labels += ["qc_hdr", "qc"]

    total_height = sum(heights)
    fig = plt.figure(figsize=(FIG_WIDTH, total_height))
    fig.set_facecolor(BG_COLOR)

    gs = GridSpec(
        len(heights),
        3,
        figure=fig,
        height_ratios=heights,
        hspace=0.3,
        wspace=0.15,
        top=0.98,
        bottom=0.01,
        left=0.03,
        right=0.97,
    )

    row = 0

    # -- Anatomical --
    _section_header(fig, gs[row, :], "ANATOMICAL PREPROCESSING")
    row += 1
    row = plot_anatomical(manifest, fig, gs, row)

    # -- Registration --
    _section_header(fig, gs[row, :], "REGISTRATION")
    row += 1
    row = plot_registration(manifest, fig, gs, row)

    # -- Functional --
    _section_header(fig, gs[row, :], "FUNCTIONAL PREPROCESSING")
    row += 1
    row = plot_functional_bold(manifest, fig, gs, row)
    row = plot_motion(manifest, fig, gs, row)

    # -- Metrics --
    if has_metrics:
        _section_header(
            fig, gs[row, :], "DERIVATIVE METRICS" + (" & QC" if has_qc else "")
        )
        row += 1
        row = plot_metrics_maps(manifest, fig, gs, row)

        if has_qc:
            corr_ax = fig.add_subplot(gs[row, 0])
            plot_correlation_matrix(manifest, corr_ax)
            qc_ax = fig.add_subplot(gs[row, 1:])
            plot_qc(manifest, qc_ax)
        else:
            corr_ax = fig.add_subplot(gs[row, :])
            plot_correlation_matrix(manifest, corr_ax)
        row += 1

    elif has_qc:
        _section_header(fig, gs[row, :], "QC METRICS")
        row += 1
        qc_ax = fig.add_subplot(gs[row, :])
        plot_qc(manifest, qc_ax)
        row += 1

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Report saved to: {output}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = (
        Path(__file__).parent.parent / "tests" / "full_pipeline" / ".last_run.json"
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=default_manifest,
        help="Path to .last_run.json manifest "
        "(default: tests/full_pipeline/.last_run.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("pipeline_report.png"),
        help="Output image path (default: pipeline_report.png)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    build_report(manifest, args.output)


if __name__ == "__main__":
    main()
