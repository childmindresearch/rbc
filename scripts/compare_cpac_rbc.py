# /// script
# dependencies = ["matplotlib>=3.9", "nibabel>=5.0", "numpy>=1.26"]
# requires-python = ">=3.12"
# ///
r"""Compare RBC pipeline outputs against C-PAC reference outputs.

Generates a markdown report with figures suitable for sharing with
neuroimagers. Comparisons include Dice overlap for masks, spatial
Pearson correlation for volumetric maps, timeseries correlation for
motion/regressors, and element-wise correlation for connectivity
matrices.

Usage:
    uv run scripts/compare_cpac_rbc.py CPAC_DIR RBC_DIR \
        [-o OUTPUT_DIR]
    uv run scripts/compare_cpac_rbc.py CPAC_DIR RBC_DIR \
        --manifest tests/full_pipeline/.last_run.json

Example:
    uv run scripts/compare_cpac_rbc.py \
        tests/data/cpac_outputs/ds000001 \
        tests/data/rbc_run \
        --manifest tests/full_pipeline/.last_run.json \
        -o reports/comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------

ComparisonKind = Literal[
    "dice",
    "spatial_corr",
    "timeseries_corr",
    "column_corr",
    "matrix_corr",
    "scalar_table",
]


@dataclass
class ComparisonItem:
    """A single output pair to compare."""

    name: str
    cpac_glob: str
    rbc_glob: str
    kind: ComparisonKind
    tier: int
    category: str


@dataclass
class ComparisonResult:
    """Outcome of a single comparison."""

    item: ComparisonItem
    metric_name: str
    value: float | str | dict[str, float]
    cpac_path: Path | None = None
    rbc_path: Path | None = None
    figure_path: Path | None = None
    note: str = ""


# ------------------------------------------------------------------
# Comparison items
# ------------------------------------------------------------------

ITEMS: list[ComparisonItem] = [
    # --- Anatomical ---
    ComparisonItem(
        "Brain mask",
        "**/anat/*desc-brain_mask.nii.gz",
        "**/anat/*desc-T1w_mask.nii.gz",
        "dice",
        1,
        "anatomical",
    ),
    ComparisonItem(
        "Skull-stripped T1w",
        "**/anat/*desc-preproc_T1w.nii.gz",
        "**/anat/*desc-brain_T1w.nii.gz",
        "spatial_corr",
        2,
        "anatomical",
    ),
    ComparisonItem(
        "CSF mask",
        "**/anat/*label-CSF_mask.nii.gz",
        "**/anat/*desc-csf_mask.nii.gz",
        "dice",
        2,
        "anatomical",
    ),
    ComparisonItem(
        "GM mask",
        "**/anat/*label-GM_mask.nii.gz",
        "**/anat/*desc-gm_mask.nii.gz",
        "dice",
        2,
        "anatomical",
    ),
    ComparisonItem(
        "WM mask",
        "**/anat/*label-WM_mask.nii.gz",
        "**/anat/*desc-wm_mask.nii.gz",
        "dice",
        2,
        "anatomical",
    ),
    # --- Functional ---
    ComparisonItem(
        "Motion parameters",
        "**/func/*desc-movementParameters_motion.1D",
        "**/func/*desc-motionParams_motion.1D",
        "column_corr",
        1,
        "functional",
    ),
    # NOTE: C-PAC outputs FD Power, RBC outputs mcflirt RMS relative
    # displacement. These are different metrics, so no direct comparison.
    ComparisonItem(
        "BOLD brain mask",
        "**/func/*_desc-brain_mask.nii.gz",
        "**/func/*_desc-brain_mask.nii.gz",
        "dice",
        1,
        "functional",
    ),
    ComparisonItem(
        "Template-space BOLD",
        "**/func/*MNI152*desc-head_bold.nii.gz",
        "**/func/*MNI152*desc-preproc_bold.nii.gz",
        "spatial_corr",
        2,
        "functional",
    ),
    ComparisonItem(
        "Nuisance regressors (36-param, raw)",
        "**/nuisance_regressors_36*/**/nuisance_regressors.1D",
        "**/func/*36*regressors.1D",
        "column_corr",
        2,
        "functional",
    ),
    ComparisonItem(
        "Nuisance regressors (36-param, filtered)",
        "**/func/*reg-36Parameter_regressors.1D",
        "**/func/*36*regressors.1D",
        "column_corr",
        2,
        "functional",
    ),
    ComparisonItem(
        "Regressed BOLD (36-param)",
        "**/func/*MNI152*reg-36Parameter*preproc_bold.nii.gz",
        "**/func/*MNI152*reg-36*preproc_bold.nii.gz",
        "spatial_corr",
        2,
        "functional",
    ),
    ComparisonItem(
        "Template brain mask",
        "**/func/*MNI152*desc-bold_mask.nii.gz",
        "**/func/*MNI152*desc-bold_mask.nii.gz",
        "dice",
        2,
        "functional",
    ),
    # --- Metrics ---
    ComparisonItem(
        "ALFF (z-scored)",
        "**/func/*reg-36Parameter*smZstd_alff.nii.gz",
        "**/func/*reg-36*smoothZstd_alff.nii.gz",
        "spatial_corr",
        2,
        "metrics",
    ),
    ComparisonItem(
        "fALFF (z-scored)",
        "**/func/*reg-36Parameter*smZstd_falff.nii.gz",
        "**/func/*reg-36*smoothZstd_falff.nii.gz",
        "spatial_corr",
        2,
        "metrics",
    ),
    ComparisonItem(
        "ReHo (z-scored)",
        "**/func/*reg-36Parameter*smZstd_reho.nii.gz",
        "**/func/*reg-36*smoothZstd_reho.nii.gz",
        "spatial_corr",
        2,
        "metrics",
    ),
    ComparisonItem(
        "Correlation matrix (Schaefer 200)",
        "**/func/*Schaefer*200*reg-36Parameter*Pearson*correlations.tsv",
        "**/func/*schaefer*200*reg-36*pearson*correlations.tsv",
        "matrix_corr",
        2,
        "metrics",
    ),
    # --- QC ---
    ComparisonItem(
        "QC metrics",
        "**/func/*reg-36Parameter*xcp*quality.tsv",
        "**/func/*reg-36*xcp*quality.tsv",
        "scalar_table",
        2,
        "qc",
    ),
]


# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------


def _find(base: Path, pattern: str) -> Path | None:
    """Find a single file matching a glob pattern."""
    matches = sorted(base.rglob(pattern))
    if matches:
        return matches[0]
    # Fallback: fuzzy substring match
    for f in sorted(base.rglob("*")):
        if f.is_file() and _fuzzy_match(f, pattern):
            return f
    return None


def _fuzzy_match(filepath: Path, pattern: str) -> bool:
    """Check if a filepath loosely matches a glob pattern."""
    name = filepath.name.lower()
    parts = pattern.lower().replace("**/", "").replace("*", " ").split()
    return all(p in name for p in parts if len(p) > 2)


def _resolve_manifest(
    manifest_path: Path,
) -> dict[str, Path]:
    """Build an RBC file map from a .last_run.json manifest.

    Returns a dict keyed by comparison item name -> Path.
    """
    with manifest_path.open() as f:
        data = json.load(f)

    mapping: dict[str, Path] = {}

    # Anatomical
    anat = data.get("anat", {})
    mapping["Brain mask"] = Path(anat["brain_mask"])
    mapping["Skull-stripped T1w"] = Path(anat["brain"])
    mapping["CSF mask"] = Path(anat["csf_mask"])
    mapping["GM mask"] = Path(anat["gm_mask"])
    mapping["WM mask"] = Path(anat["wm_mask"])

    # Functional
    func = data.get("func", {})
    mapping["Motion parameters"] = Path(func["motion_params"])
    mapping["BOLD brain mask"] = Path(func["bold_mask"])
    mapping["Template-space BOLD"] = Path(func["template_bold"])
    # Use raw (unfiltered) regressors for comparison; the
    # filtered version is at func["regressor_file"].
    raw_reg = Path(func["regressor_file"]).parent / "regressors.1D"
    mapping["Nuisance regressors (36-param, raw)"] = (
        raw_reg if raw_reg.exists() else Path(func["regressor_file"])
    )
    mapping["Nuisance regressors (36-param, filtered)"] = Path(
        func["regressor_file"],
    )
    mapping["Regressed BOLD (36-param)"] = Path(
        func["regressed_bold"],
    )
    mapping["Template brain mask"] = Path(
        func["template_brain_mask"],
    )

    # Metrics
    metrics = data.get("metrics", {})
    mapping["ALFF (z-scored)"] = Path(metrics["alff_zscored"])
    mapping["fALFF (z-scored)"] = Path(metrics["falff_zscored"])
    mapping["ReHo (z-scored)"] = Path(metrics["reho_zscored"])
    mapping["Correlation matrix (Schaefer 200)"] = Path(
        metrics["correlation_matrix"],
    )

    # QC
    qc = data.get("qc", {})
    mapping["QC metrics"] = Path(qc["qc_file"])

    return mapping


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def dice_coefficient(
    a: NDArray[np.bool_],
    b: NDArray[np.bool_],
) -> float:
    """Dice overlap between two binary masks."""
    intersection = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def spatial_correlation(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Pearson correlation of voxel values within a mask."""
    if mask is not None:
        a_vals = a[mask].astype(np.float64)
        b_vals = b[mask].astype(np.float64)
    else:
        nonzero = (a != 0) | (b != 0)
        a_vals = a[nonzero].astype(np.float64)
        b_vals = b[nonzero].astype(np.float64)
    if len(a_vals) < 2:
        return 0.0
    r = np.corrcoef(a_vals, b_vals)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def mean_temporal_correlation(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> float:
    """Mean spatial Pearson r across timepoints for 4D data."""
    if a.ndim != 4 or b.ndim != 4:
        return 0.0
    n_vols = min(a.shape[-1], b.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])[:, :n_vols]
    b_flat = b.reshape(-1, b.shape[-1])[:, :n_vols]
    correlations = []
    for t in range(n_vols):
        av = a_flat[:, t].astype(np.float64)
        bv = b_flat[:, t].astype(np.float64)
        nonzero = (av != 0) | (bv != 0)
        if nonzero.sum() > 2:
            r = np.corrcoef(av[nonzero], bv[nonzero])[0, 1]
            if np.isfinite(r):
                correlations.append(r)
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def load_1d(path: Path) -> NDArray[np.floating]:
    """Load an AFNI .1D or .rms text file, skipping comments."""
    lines = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    if not lines:
        return np.array([])
    return np.loadtxt(lines, ndmin=2)


def load_tsv_matrix(path: Path) -> NDArray[np.floating]:
    """Load a TSV as a numeric matrix, skipping header if present."""
    with path.open() as f:
        first_line = f.readline().strip()
    try:
        float(first_line.split("\t")[0])
        skip = 0
    except ValueError:
        skip = 1
    return np.loadtxt(path, delimiter="\t", skiprows=skip)


# ------------------------------------------------------------------
# Figure generation
# ------------------------------------------------------------------

matplotlib.use("Agg")


def _get_middle_slices(
    data: NDArray,
) -> tuple[int, int, int]:
    """Get the middle slice index for each axis."""
    if data.ndim == 4:
        data = data[..., data.shape[-1] // 2]
    return (
        data.shape[0] // 2,
        data.shape[1] // 2,
        data.shape[2] // 2,
    )


# Orientation labels for RPI data: (left, right, bottom, top)
_ORIENT_LABELS: dict[int, tuple[str, str, str, str]] = {
    0: ("P", "A", "I", "S"),  # sagittal
    1: ("R", "L", "I", "S"),  # coronal
    2: ("R", "L", "P", "A"),  # axial
}


def _plot_slice(
    ax: plt.Axes,  # type: ignore[name-defined]
    data: NDArray,
    axis: int,
    idx: int,
    *,
    orient_labels: bool = True,
    **kwargs: float | str,
) -> None:
    """Plot a single 2D slice from a 3D volume (RPI convention)."""
    slices: list[slice | int] = [slice(None)] * 3
    slices[axis] = idx
    img = data[tuple(slices)].T
    ax.imshow(img, origin="lower", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if orient_labels:
        left, right, bottom, top = _ORIENT_LABELS[axis]
        h, w = img.shape[:2]
        kw = {"fontsize": 7, "color": "yellow", "ha": "center", "va": "center"}
        ax.text(0, h / 2, left, **kw)
        ax.text(w - 1, h / 2, right, **kw)
        ax.text(w / 2, 1, bottom, **kw)
        ax.text(w / 2, h - 2, top, **kw)


def fig_mask_comparison(
    cpac_data: NDArray,
    rbc_data: NDArray,
    name: str,
    out_path: Path,
) -> None:
    """Three-panel figure: C-PAC mask, RBC mask, overlap."""
    sx, sy, sz = _get_middle_slices(cpac_data)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    view_spec = [
        (0, sx, "Sagittal"),
        (1, sy, "Coronal"),
        (2, sz, "Axial"),
    ]
    for col, (axis, idx, label) in enumerate(view_spec):
        _plot_slice(
            axes[0, col],
            cpac_data.astype(float),
            axis,
            idx,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        if col == 0:
            axes[0, col].set_ylabel("C-PAC", fontsize=12)
        axes[0, col].set_title(label)

        _plot_slice(
            axes[1, col],
            rbc_data.astype(float),
            axis,
            idx,
            cmap="Oranges",
            vmin=0,
            vmax=1,
        )
        if col == 0:
            axes[1, col].set_ylabel("RBC", fontsize=12)

        overlap = np.zeros((*cpac_data.shape, 3))
        overlap[..., 2] = cpac_data.astype(float)
        overlap[..., 0] = rbc_data.astype(float)
        both = cpac_data & rbc_data
        overlap[..., 1] = both.astype(float) * 0.5

        sl: list[slice | int] = [slice(None)] * 3
        sl[axis] = idx
        ol_slice = overlap[tuple(sl)]
        # Transpose first two dims while keeping RGB
        ol_img = np.transpose(ol_slice, (1, 0, 2))
        axes[2, col].imshow(ol_img, origin="lower")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
        if col == 0:
            axes[2, col].set_ylabel("Overlap", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_volume_comparison(
    cpac_data: NDArray,
    rbc_data: NDArray,
    name: str,
    out_path: Path,
) -> None:
    """Side-by-side volume slices with difference map."""
    if cpac_data.ndim == 4:
        cpac_data = cpac_data.std(axis=-1)
    if rbc_data.ndim == 4:
        rbc_data = rbc_data.std(axis=-1)

    sx, sy, sz = _get_middle_slices(cpac_data)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    vmin = min(cpac_data.min(), rbc_data.min())
    vmax = max(
        np.percentile(cpac_data, 99),
        np.percentile(rbc_data, 99),
    )

    view_spec = [
        (0, sx, "Sagittal"),
        (1, sy, "Coronal"),
        (2, sz, "Axial"),
    ]
    for col, (axis, idx, label) in enumerate(view_spec):
        _plot_slice(
            axes[0, col],
            cpac_data,
            axis,
            idx,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        if col == 0:
            axes[0, col].set_ylabel("C-PAC", fontsize=12)
        axes[0, col].set_title(label)

        _plot_slice(
            axes[1, col],
            rbc_data,
            axis,
            idx,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        if col == 0:
            axes[1, col].set_ylabel("RBC", fontsize=12)

        diff = cpac_data.astype(np.float64) - rbc_data.astype(np.float64)
        d_lim = max(
            np.abs(np.percentile(diff, 1)),
            np.abs(np.percentile(diff, 99)),
            1e-10,
        )
        _plot_slice(
            axes[2, col],
            diff,
            axis,
            idx,
            cmap="RdBu_r",
            vmin=-d_lim,
            vmax=d_lim,
        )
        if col == 0:
            axes[2, col].set_ylabel("Difference", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_timeseries(
    cpac_data: NDArray,
    rbc_data: NDArray,
    name: str,
    out_path: Path,
    labels: list[str] | None = None,
) -> None:
    """Overlay timeseries plot."""
    if cpac_data.ndim == 1:
        cpac_data = cpac_data.reshape(-1, 1)
    if rbc_data.ndim == 1:
        rbc_data = rbc_data.reshape(-1, 1)

    n_cols = min(cpac_data.shape[1], rbc_data.shape[1], 9)
    fig, axes = plt.subplots(
        n_cols,
        1,
        figsize=(12, 2.5 * n_cols),
        squeeze=False,
    )
    fig.suptitle(name, fontsize=14, fontweight="bold")

    for i in range(n_cols):
        ax = axes[i, 0]
        n_tp = min(len(cpac_data[:, i]), len(rbc_data[:, i]))
        ax.plot(
            cpac_data[:n_tp, i],
            label="C-PAC",
            alpha=0.8,
            linewidth=1,
        )
        ax.plot(
            rbc_data[:n_tp, i],
            label="RBC",
            alpha=0.8,
            linewidth=1,
            linestyle="--",
        )
        col_label = labels[i] if labels and i < len(labels) else f"Column {i}"
        ax.set_ylabel(col_label, fontsize=9)
        if i == 0:
            ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1, 0].set_xlabel("Timepoint")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_correlation_matrices(
    cpac_mat: NDArray,
    rbc_mat: NDArray,
    name: str,
    out_path: Path,
) -> None:
    """Side-by-side heatmaps of correlation matrices."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        gridspec_kw={"wspace": 0.3},
    )
    fig.suptitle(name, fontsize=14, fontweight="bold")

    vmin = min(np.nanmin(cpac_mat), np.nanmin(rbc_mat))
    vmax = max(np.nanmax(cpac_mat), np.nanmax(rbc_mat))

    im0 = axes[0].imshow(
        cpac_mat,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[0].set_title("C-PAC")
    axes[0].set_xlabel("ROI")
    axes[0].set_ylabel("ROI")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        rbc_mat,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title("RBC")
    axes[1].set_xlabel("ROI")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = cpac_mat - rbc_mat
    d_lim = max(
        np.abs(np.nanpercentile(diff, 1)),
        np.abs(np.nanpercentile(diff, 99)),
        1e-10,
    )
    im2 = axes[2].imshow(
        diff,
        cmap="RdBu_r",
        vmin=-d_lim,
        vmax=d_lim,
        aspect="equal",
    )
    axes[2].set_title("Difference")
    axes[2].set_xlabel("ROI")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scalar_table(
    cpac_vals: dict[str, float],
    rbc_vals: dict[str, float],
    name: str,
    out_path: Path,
) -> None:
    """Table figure comparing scalar QC values."""
    keys = sorted(set(cpac_vals) & set(rbc_vals))
    if not keys:
        return

    rows = []
    for k in keys:
        cv, rv = cpac_vals[k], rbc_vals[k]
        pct = f"{abs(rv - cv) / abs(cv) * 100:.1f}%" if cv != 0 else "N/A"
        rows.append(
            [
                k,
                f"{cv:.4f}",
                f"{rv:.4f}",
                f"{rv - cv:.4f}",
                pct,
            ]
        )

    fig, ax = plt.subplots(
        figsize=(10, max(3, 0.4 * len(rows) + 1)),
    )
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "C-PAC", "RBC", "Diff", "% Diff"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)  # noqa: FBT003
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title(name, fontsize=14, fontweight="bold", pad=20)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Comparison runners
# ------------------------------------------------------------------


def compare_dice(
    cpac_path: Path,
    rbc_path: Path,
) -> tuple[float, NDArray, NDArray]:
    """Load two masks and compute Dice."""
    a = nib.load(cpac_path).get_fdata() > 0
    b = nib.load(rbc_path).get_fdata() > 0
    return dice_coefficient(a, b), a, b


def compare_spatial(
    cpac_path: Path,
    rbc_path: Path,
) -> tuple[float, NDArray, NDArray]:
    """Load two volumes and compute spatial correlation."""
    a = nib.load(cpac_path).get_fdata()
    b = nib.load(rbc_path).get_fdata()
    if a.ndim == 4 and b.ndim == 4:
        r = mean_temporal_correlation(a, b)
    else:
        if a.ndim == 4:
            a = a.mean(axis=-1)
        if b.ndim == 4:
            b = b.mean(axis=-1)
        r = spatial_correlation(a, b)
    return r, a, b


def compare_timeseries(
    cpac_path: Path,
    rbc_path: Path,
) -> tuple[float, NDArray, NDArray]:
    """Load two 1D files and compute correlation."""
    a = load_1d(cpac_path).ravel()
    b = load_1d(rbc_path).ravel()
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, a, b
    r = float(np.corrcoef(a[:n], b[:n])[0, 1])
    return (r if np.isfinite(r) else 0.0), a, b


# Map C-PAC column names to RBC column names for regressors.
_REGRESSOR_NAME_MAP: dict[str, str] = {
    "RotY": "roll",
    "RotYDelay": "roll_delayed",
    "RotYSq": "roll_sq",
    "RotYDelaySq": "roll_delayed_sq",
    "RotX": "pitch",
    "RotXDelay": "pitch_delayed",
    "RotXSq": "pitch_sq",
    "RotXDelaySq": "pitch_delayed_sq",
    "RotZ": "yaw",
    "RotZDelay": "yaw_delayed",
    "RotZSq": "yaw_sq",
    "RotZDelaySq": "yaw_delayed_sq",
    "Y": "dS",
    "YDelay": "dS_delayed",
    "YSq": "dS_sq",
    "YDelaySq": "dS_delayed_sq",
    "X": "dL",
    "XDelay": "dL_delayed",
    "XSq": "dL_sq",
    "XDelaySq": "dL_delayed_sq",
    "Z": "dP",
    "ZDelay": "dP_delayed",
    "ZSq": "dP_sq",
    "ZDelaySq": "dP_delayed_sq",
    "GlobalSignalMean0": "global",
    "GlobalSignalMean0Delay": "global_delayed",
    "GlobalSignalMean0Sq": "global_sq",
    "GlobalSignalMean0DelaySq": "global_delayed_sq",
    "CerebrospinalFluidMean0": "csf",
    "CerebrospinalFluidMean0Delay": "csf_delayed",
    "CerebrospinalFluidMean0Sq": "csf_sq",
    "CerebrospinalFluidMean0DelaySq": "csf_delayed_sq",
    "WhiteMatterMean0": "wm",
    "WhiteMatterMean0Delay": "wm_delayed",
    "WhiteMatterMean0Sq": "wm_sq",
    "WhiteMatterMean0DelaySq": "wm_delayed_sq",
}


def _parse_1d_headers(path: Path) -> list[str]:
    """Extract column names from an AFNI .1D header comment."""
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("# C-PAC"):
                names = stripped.lstrip("# ").split()
                if len(names) > 3:
                    return names
    return []


def compare_columns(
    cpac_path: Path,
    rbc_path: Path,
) -> tuple[dict[str, float], NDArray, NDArray, list[str]]:
    """Load multi-column 1D files, match by name, correlate.

    Returns correlations, name-aligned C-PAC array, name-aligned
    RBC array, and matched column labels.
    """
    a = load_1d(cpac_path)
    b = load_1d(rbc_path)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    cpac_names = _parse_1d_headers(cpac_path)
    rbc_names = _parse_1d_headers(rbc_path)

    n_rows = min(a.shape[0], b.shape[0])
    results: dict[str, float] = {}
    matched_a: list[NDArray] = []
    matched_b: list[NDArray] = []
    labels: list[str] = []

    if cpac_names and rbc_names:
        rbc_idx = {n: i for i, n in enumerate(rbc_names)}
        for ci, cname in enumerate(cpac_names):
            rbc_name = _REGRESSOR_NAME_MAP.get(cname, cname)
            if rbc_name in rbc_idx:
                ri = rbc_idx[rbc_name]
                r = np.corrcoef(
                    a[:n_rows, ci],
                    b[:n_rows, ri],
                )[0, 1]
                results[rbc_name] = float(r) if np.isfinite(r) else 0.0
                matched_a.append(a[:n_rows, ci])
                matched_b.append(b[:n_rows, ri])
                labels.append(rbc_name)
    else:
        # No headers found, fall back to positional matching
        _fallback = ["roll", "pitch", "yaw", "dS", "dL", "dP"]
        n_cols = min(a.shape[1], b.shape[1])
        for c in range(n_cols):
            r = np.corrcoef(
                a[:n_rows, c],
                b[:n_rows, c],
            )[0, 1]
            col_name = _fallback[c] if c < len(_fallback) else f"col_{c}"
            results[col_name] = float(r) if np.isfinite(r) else 0.0
            matched_a.append(a[:n_rows, c])
            matched_b.append(b[:n_rows, c])
            labels.append(col_name)

    a_aligned = np.column_stack(matched_a) if matched_a else a
    b_aligned = np.column_stack(matched_b) if matched_b else b
    return results, a_aligned, b_aligned, labels


def compare_matrix(
    cpac_path: Path,
    rbc_path: Path,
) -> tuple[float, NDArray, NDArray]:
    """Load two TSV matrices, compute element-wise correlation."""
    a = load_tsv_matrix(cpac_path)
    b = load_tsv_matrix(rbc_path)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n, :n], b[:n, :n]
    triu = np.triu_indices(n, k=1)
    a_vals, b_vals = a[triu], b[triu]
    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    if valid.sum() < 2:
        return 0.0, a, b
    r = float(np.corrcoef(a_vals[valid], b_vals[valid])[0, 1])
    return (r if np.isfinite(r) else 0.0), a, b


def parse_qc_tsv(path: Path) -> dict[str, float]:
    """Parse a single-row XCP QC TSV into a dict."""
    with path.open() as f:
        header = f.readline().strip().split("\t")
        values = f.readline().strip().split("\t")
    result: dict[str, float] = {}
    for k, v in zip(header, values, strict=False):
        try:
            result[k] = float(v)
        except ValueError:
            continue
    return result


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

KNOWN_DIVERGENCES = """\
### Known intentional divergences from C-PAC

These differences are deliberate bug fixes in RBC relative to the
C-PAC implementation. Lower agreement on affected outputs is expected
and desired.

1. **N4 bias field correction**: C-PAC runs N4 internally during ANTs
   brain extraction but then discards the corrected image, applying the
   brain mask to the original (uncorrected) T1w. RBC retains the
   N4-corrected output. This propagates to segmentation, registration,
   and all downstream steps that depend on the anatomical image.

2. **Despiking order**: C-PAC runs 3dDespike in template space (after
   resampling), which undermines motion estimation because voxels no
   longer correspond to consistent tissue locations. RBC despiked in
   native space before motion correction, which is the scientifically
   correct order. This affects motion parameters, nuisance regressors,
   and the functional outputs that depend on them.
"""


def _tier_label(tier: int) -> str:
    if tier == 1:
        return "Should match closely"
    return "Differences expected"


def generate_report(
    results: list[ComparisonResult],
    output_dir: Path,
) -> Path:
    """Write a markdown report summarizing all comparisons."""
    report_path = output_dir / "comparison_report.md"
    lines: list[str] = []

    lines.append("# RBC vs C-PAC Pipeline Comparison Report\n")
    lines.append(
        "This report compares outputs from the RBC standalone "
        "pipeline against the C-PAC RBC reference implementation, "
        "run on the same input data (ds000001, sub-01). The goal "
        "is to validate that RBC produces equivalent results where "
        "the implementations are aligned, and to document expected "
        "differences where RBC intentionally diverges.\n"
    )

    lines.append(KNOWN_DIVERGENCES)

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Step | Category | Metric | Value | Expectation |")
    lines.append("|------|----------|--------|-------|-------------|")

    for r in results:
        display = _format_value(r)
        lines.append(
            f"| {r.item.name} | {r.item.category} "
            f"| {display} | "
            f"{_tier_label(r.item.tier)} |"
        )
    lines.append("")

    # Detailed results by category
    for category in [
        "anatomical",
        "functional",
        "metrics",
        "qc",
    ]:
        cat_results = [r for r in results if r.item.category == category]
        if not cat_results:
            continue
        lines.append(f"## {category.title()}\n")
        for r in cat_results:
            _append_detail(lines, r, output_dir)

    # Interpretation guide
    lines.append("## Interpretation guide\n")
    lines.append(
        "- **Dice > 0.90**: Strong mask agreement. Minor boundary "
        "differences are normal due to implementation details in "
        "skull-stripping and thresholding.\n"
        "- **Spatial r > 0.90**: Strong volumetric agreement. "
        "Values above 0.95 indicate near-identical spatial "
        "patterns.\n"
        "- **Spatial r 0.70--0.90**: Moderate agreement. Expected "
        "for outputs affected by the N4 and despiking divergences, "
        "since these change the input to downstream steps.\n"
        "- **Spatial r < 0.70**: Substantial difference. Warrants "
        "investigation unless fully explained by a known "
        "divergence.\n"
        "- **Timeseries/column r > 0.95**: Motion parameters and "
        "regressors track closely. Small numerical differences "
        "from floating-point and tool version differences are "
        "normal.\n"
        "- **Matrix r > 0.80**: Connectivity structure is preserved "
        "across pipelines despite upstream processing "
        "differences.\n"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _format_value(r: ComparisonResult) -> str:
    """Format a result value for the summary table."""
    if isinstance(r.value, dict):
        vals = [v for v in r.value.values() if isinstance(v, float)]
        if vals:
            return f"mean r = {np.mean(vals):.4f}"
        return "see table"
    if isinstance(r.value, float):
        return f"{r.metric_name} = {r.value:.4f}"
    return str(r.value)


def _append_detail(
    lines: list[str],
    r: ComparisonResult,
    output_dir: Path,
) -> None:
    """Append detailed result section for one comparison."""
    lines.append(f"### {r.item.name}\n")
    lines.append(f"- **C-PAC**: `{r.cpac_path}`")
    lines.append(f"- **RBC**: `{r.rbc_path}`")

    if isinstance(r.value, dict):
        lines.append(f"- **Metric**: {r.metric_name}")
        for k, v in r.value.items():
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"  - {k}: {fmt}")
    elif isinstance(r.value, float):
        lines.append(f"- **{r.metric_name}**: {r.value:.4f}")
    else:
        lines.append(f"- **{r.metric_name}**: {r.value}")

    lines.append(f"- **Expectation**: {_tier_label(r.item.tier)}")
    if r.note:
        lines.append(f"- **Note**: {r.note}")
    if r.figure_path:
        rel = r.figure_path.relative_to(output_dir)
        lines.append(f"\n![{r.item.name}]({rel})\n")
    lines.append("")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def run_comparisons(
    cpac_dir: Path,
    rbc_dir: Path,
    output_dir: Path,
    rbc_manifest: dict[str, Path] | None = None,
) -> list[ComparisonResult]:
    """Run all comparisons and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    results: list[ComparisonResult] = []

    for item in ITEMS:
        cpac_path = _find(cpac_dir, item.cpac_glob)
        if rbc_manifest and item.name in rbc_manifest:
            rbc_path: Path | None = rbc_manifest[item.name]
            if not rbc_path.exists():
                rbc_path = None
        else:
            rbc_path = _find(rbc_dir, item.rbc_glob)

        if cpac_path is None or rbc_path is None:
            missing = []
            if cpac_path is None:
                missing.append(f"C-PAC ({item.cpac_glob})")
            if rbc_path is None:
                missing.append(f"RBC ({item.rbc_glob})")
            log.warning(
                "SKIP %s: missing %s",
                item.name,
                ", ".join(missing),
            )
            results.append(
                ComparisonResult(
                    item=item,
                    metric_name="N/A",
                    value="MISSING",
                    cpac_path=cpac_path,
                    rbc_path=rbc_path,
                    note=f"Missing: {', '.join(missing)}",
                )
            )
            continue

        fig_name = item.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig_path = fig_dir / f"{fig_name}.png"

        log.info("Comparing: %s ...", item.name)

        try:
            result = _run_single(
                item,
                cpac_path,
                rbc_path,
                fig_path,
            )
            results.append(result)
            log.info("  done")
        except Exception:
            log.exception("  ERROR on %s", item.name)
            results.append(
                ComparisonResult(
                    item=item,
                    metric_name="Error",
                    value="ERROR",
                    cpac_path=cpac_path,
                    rbc_path=rbc_path,
                )
            )

    return results


def _run_single(
    item: ComparisonItem,
    cpac_path: Path,
    rbc_path: Path,
    fig_path: Path,
) -> ComparisonResult:
    """Execute a single comparison and generate its figure."""
    if item.kind == "dice":
        val, cd, rd = compare_dice(cpac_path, rbc_path)
        fig_mask_comparison(cd, rd, item.name, fig_path)
        return ComparisonResult(
            item=item,
            metric_name="Dice",
            value=val,
            cpac_path=cpac_path,
            rbc_path=rbc_path,
            figure_path=fig_path,
        )

    if item.kind == "spatial_corr":
        val, cd, rd = compare_spatial(cpac_path, rbc_path)
        fig_volume_comparison(cd, rd, item.name, fig_path)
        return ComparisonResult(
            item=item,
            metric_name="Spatial r",
            value=val,
            cpac_path=cpac_path,
            rbc_path=rbc_path,
            figure_path=fig_path,
        )

    if item.kind == "timeseries_corr":
        val, cd, rd = compare_timeseries(cpac_path, rbc_path)
        fig_timeseries(
            cd.reshape(-1, 1),
            rd.reshape(-1, 1),
            item.name,
            fig_path,
        )
        return ComparisonResult(
            item=item,
            metric_name="Pearson r",
            value=val,
            cpac_path=cpac_path,
            rbc_path=rbc_path,
            figure_path=fig_path,
        )

    if item.kind == "column_corr":
        vals, cd, rd, col_labels = compare_columns(cpac_path, rbc_path)
        # Show only base columns (no delayed/sq) in the figure
        base_idx = [
            i
            for i, lb in enumerate(col_labels)
            if "_delayed" not in lb and "_sq" not in lb
        ]
        if base_idx:
            fig_timeseries(
                cd[:, base_idx],
                rd[:, base_idx],
                item.name,
                fig_path,
                labels=[col_labels[i] for i in base_idx],
            )
        else:
            fig_timeseries(
                cd,
                rd,
                item.name,
                fig_path,
                labels=col_labels,
            )
        return ComparisonResult(
            item=item,
            metric_name="Per-column Pearson r",
            value=vals,
            cpac_path=cpac_path,
            rbc_path=rbc_path,
            figure_path=fig_path,
        )

    if item.kind == "matrix_corr":
        val, cd, rd = compare_matrix(cpac_path, rbc_path)
        fig_correlation_matrices(cd, rd, item.name, fig_path)
        return ComparisonResult(
            item=item,
            metric_name="Element-wise r (upper tri)",
            value=val,
            cpac_path=cpac_path,
            rbc_path=rbc_path,
            figure_path=fig_path,
        )

    # scalar_table
    cpac_vals = parse_qc_tsv(cpac_path)
    rbc_vals = parse_qc_tsv(rbc_path)
    fig_scalar_table(cpac_vals, rbc_vals, item.name, fig_path)
    common = sorted(set(cpac_vals) & set(rbc_vals))
    val_dict = {k: rbc_vals[k] - cpac_vals[k] for k in common}
    return ComparisonResult(
        item=item,
        metric_name="QC metric differences (RBC - C-PAC)",
        value=val_dict,
        cpac_path=cpac_path,
        rbc_path=rbc_path,
        figure_path=fig_path,
    )


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compare RBC and C-PAC pipeline outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              uv run python scripts/compare_cpac_rbc.py \\
                  tests/data/cpac_outputs/ds000001 \\
                  tests/data/rbc_outputs \\
                  -o reports/comparison
        """),
    )
    parser.add_argument(
        "cpac_dir",
        type=Path,
        help="C-PAC output directory",
    )
    parser.add_argument(
        "rbc_dir",
        type=Path,
        help="RBC output directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports/comparison"),
        help="Output directory for report and figures",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="RBC .last_run.json manifest (bypass glob search)",
    )
    args = parser.parse_args()

    if not args.cpac_dir.exists():
        log.error("C-PAC directory not found: %s", args.cpac_dir)
        sys.exit(1)
    if not args.rbc_dir.exists() and args.manifest is None:
        log.error("RBC directory not found: %s", args.rbc_dir)
        sys.exit(1)

    rbc_manifest = None
    if args.manifest is not None:
        if not args.manifest.exists():
            log.error("Manifest not found: %s", args.manifest)
            sys.exit(1)
        rbc_manifest = _resolve_manifest(args.manifest)
        log.info("Using manifest: %s", args.manifest)

    log.info("C-PAC outputs: %s", args.cpac_dir)
    log.info("RBC outputs:   %s", args.rbc_dir)
    log.info("Report output: %s", args.output_dir)
    log.info("")

    results = run_comparisons(
        args.cpac_dir,
        args.rbc_dir,
        args.output_dir,
        rbc_manifest=rbc_manifest,
    )
    report_path = generate_report(results, args.output_dir)

    log.info("")
    log.info("Report written to: %s", report_path)
    log.info("Figures saved to:  %s", args.output_dir / "figures")

    # Quick summary to terminal
    log.info("")
    log.info("=== Quick Summary ===")
    for r in results:
        if isinstance(r.value, float):
            status = "PASS" if r.value > 0.80 else "CHECK"
            log.info(
                "  [%s] %s: %s = %.4f",
                status,
                r.item.name,
                r.metric_name,
                r.value,
            )
        elif isinstance(r.value, dict):
            vals = [v for v in r.value.values() if isinstance(v, float)]
            if vals:
                mean_v = float(np.mean(vals))
                status = "PASS" if mean_v > 0.80 else "CHECK"
                log.info(
                    "  [%s] %s: mean = %.4f",
                    status,
                    r.item.name,
                    mean_v,
                )
        elif isinstance(r.value, str):
            log.info(
                "  [SKIP] %s: %s",
                r.item.name,
                r.value,
            )


if __name__ == "__main__":
    main()
