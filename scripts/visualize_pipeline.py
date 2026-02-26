# /// script
# dependencies = ["nilearn>=0.11", "matplotlib>=3.9", "nibabel>=5.0", "numpy>=1.26"]
# requires-python = ">=3.12"
# ///
"""Visualize outputs from a full-pipeline e2e test run.

Reads the manifest written by ``tests/full_pipeline/conftest.py`` and generates
a multi-panel PNG report of the key anatomical, functional, QC, and metrics
outputs.

Usage::

    uv run scripts/visualize_pipeline.py                          # default manifest
    uv run scripts/visualize_pipeline.py path/to/.last_run.json   # custom path
    uv run scripts/visualize_pipeline.py --output report.png      # custom output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from nilearn import image, plotting

# -- Style constants --
SECTION_COLOR = "#2c3e50"
SECTION_FONTSIZE = 11
NILEARN_ROW_HEIGHT = 2.2
PLOT_ROW_HEIGHT = 2.0
FIG_WIDTH = 20

# Good default MNI cut coordinates for ortho views (sagittal, coronal, axial).
MNI_CUTS = (2, -10, 8)


def _robust_vmax(img) -> float:
    """Return a robust upper intensity for display (98th percentile of brain)."""
    data = img.get_fdata()
    values = data[data > 0]
    return float(np.percentile(values, 98)) if len(values) > 0 else 1.0


def load_manifest(path: Path) -> dict:
    """Load the JSON manifest from a full-pipeline test run."""
    if not path.exists():
        print(f"Manifest not found: {path}", file=sys.stderr)
        print("Run the full-pipeline tests first:", file=sys.stderr)
        print("  uv run pytest tests/full_pipeline/ -v", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _section_header(fig, gs_row, title: str) -> None:
    """Add a section header spanning the full row."""
    ax = fig.add_subplot(gs_row)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        0.3,
        title,
        fontsize=SECTION_FONTSIZE,
        fontweight="bold",
        color=SECTION_COLOR,
        va="center",
        ha="left",
        family="monospace",
    )
    ax.axhline(0.1, color=SECTION_COLOR, linewidth=0.8, xmin=0.0, xmax=1.0)
    ax.axis("off")


def plot_anatomical(manifest: dict, axes: list) -> None:
    """Plot brain extraction and tissue segmentation."""
    anat = manifest["anat"]

    plotting.plot_anat(
        anat["brain"],
        title="Brain extraction",
        display_mode="ortho",
        cut_coords=plotting.find_xyz_cut_coords(anat["brain"]),
        axes=axes[0],
    )
    cuts = plotting.find_xyz_cut_coords(anat["brain"])
    for ax, key, label in [
        (axes[1], "wm_mask", "WM mask"),
        (axes[2], "gm_mask", "GM mask"),
        (axes[3], "csf_mask", "CSF mask"),
    ]:
        plotting.plot_roi(
            anat[key],
            bg_img=anat["brain"],
            title=label,
            display_mode="ortho",
            alpha=0.4,
            cut_coords=cuts,
            axes=ax,
        )


def plot_registration(manifest: dict, axes: list) -> None:
    """Plot registration quality overlays."""
    func = manifest["func"]
    template_brain_mask = manifest["template_brain_mask"]

    bold_cuts = plotting.find_xyz_cut_coords(func["skull_stripped_bold"])
    plotting.plot_roi(
        func["bold_mask"],
        bg_img=func["skull_stripped_bold"],
        title="BOLD mask on native BOLD ref",
        display_mode="ortho",
        alpha=0.3,
        cut_coords=bold_cuts,
        axes=axes[0],
    )
    tmpl_mean = image.mean_img(func["template_bold"])
    plotting.plot_roi(
        template_brain_mask,
        bg_img=tmpl_mean,
        title="Template brain mask on template BOLD",
        display_mode="ortho",
        alpha=0.3,
        vmax=_robust_vmax(tmpl_mean),
        cut_coords=MNI_CUTS,
        axes=axes[1],
    )


def plot_functional_bold(manifest: dict, axes: list) -> None:
    """Plot template and cleaned BOLD images."""
    func = manifest["func"]

    tmpl_mean = image.mean_img(func["template_bold"])
    plotting.plot_epi(
        tmpl_mean,
        title="Template BOLD (mean)",
        display_mode="ortho",
        vmax=_robust_vmax(tmpl_mean),
        cut_coords=MNI_CUTS,
        axes=axes[0],
    )
    clean_std = image.math_img("np.std(img, axis=-1)", img=func["cleaned_bold"])
    plotting.plot_epi(
        clean_std,
        title="Cleaned BOLD (temporal std)",
        display_mode="ortho",
        vmax=_robust_vmax(clean_std),
        cut_coords=MNI_CUTS,
        axes=axes[1],
    )


def plot_motion(manifest: dict, axes: list) -> None:
    """Plot motion parameters — rotation, translation, RMS in one row."""
    func = manifest["func"]
    motion = np.loadtxt(func["motion_params"])
    rms_rel = np.loadtxt(func["rms_rel"])

    # Rotation
    ax = axes[0]
    for i, label in enumerate(["rot_x", "rot_y", "rot_z"]):
        ax.plot(np.degrees(motion[:, i]), label=label, linewidth=0.8)
    ax.set_ylabel("Rotation (deg)", fontsize=8)
    ax.set_xlabel("Volume", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Rotation", fontsize=9)
    ax.tick_params(labelsize=7)

    # Translation
    ax2 = axes[1]
    for i, label in enumerate(["trans_x", "trans_y", "trans_z"], start=3):
        ax2.plot(motion[:, i], label=label, linewidth=0.8)
    ax2.set_ylabel("Translation (mm)", fontsize=8)
    ax2.set_xlabel("Volume", fontsize=8)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("Translation", fontsize=9)
    ax2.tick_params(labelsize=7)

    # RMS displacement
    ax3 = axes[2]
    ax3.plot(rms_rel, color="tab:orange", linewidth=0.8)
    ax3.axhline(0.2, color="tab:red", ls="--", alpha=0.6, label="0.2 mm threshold")
    ax3.set_ylabel("RMS (mm)", fontsize=8)
    ax3.set_xlabel("Volume", fontsize=8)
    ax3.set_title("Relative RMS displacement", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.tick_params(labelsize=7)


def plot_metrics_maps(manifest: dict, axes: list) -> None:
    """Plot ALFF, fALFF, and ReHo z-scored maps."""
    metrics = manifest["metrics"]
    func = manifest["func"]
    bg = image.mean_img(func["template_bold"])

    for ax, key, label in [
        (axes[0], "alff_zscored", "ALFF (z-scored)"),
        (axes[1], "falff_zscored", "fALFF (z-scored)"),
        (axes[2], "reho_zscored", "ReHo (z-scored)"),
    ]:
        plotting.plot_stat_map(
            metrics[key],
            bg_img=bg,
            title=label,
            display_mode="ortho",
            threshold="auto",
            cut_coords=MNI_CUTS,
            axes=ax,
            vmax=5,
            black_bg=True,
            dim=-0.5,
        )


def plot_correlation_matrix(manifest: dict, ax: plt.Axes) -> None:
    """Plot the FC correlation matrix."""
    corr = np.loadtxt(manifest["metrics"]["correlation_matrix"], delimiter="\t")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title("FC correlation matrix", fontsize=9, fontweight="bold")
    ax.set_xlabel("ROI", fontsize=8)
    ax.set_ylabel("ROI", fontsize=8)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_qc(manifest: dict, ax: plt.Axes) -> None:
    """Render QC metrics as a summary table."""
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
    color = "#27ae60" if passed else "#e74c3c"
    ax.set_title(
        f"QC Summary \u2014 {status}",
        fontsize=11,
        fontweight="bold",
        color=color,
        loc="left",
    )
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Style group headers
    for i, (group_name, keys) in enumerate(groups):
        row_idx = sum(1 + len(k) for k in [keys for _, keys in groups[:i]]) + i + 1
        for col in range(2):
            cell = table[row_idx, col]
            cell.set_text_props(fontweight="bold", color=SECTION_COLOR)
            cell.set_facecolor("#ecf0f1")


def build_report(manifest: dict, output: Path) -> None:
    """Build a multi-panel visualization report."""
    has_metrics = "metrics" in manifest
    has_qc = "qc" in manifest

    # Compute total height
    # Sections: header(0.4) + nilearn_row(2.2) each + plot_row(2.0)
    heights = []
    labels = []

    # Anatomical: header + 4 nilearn rows
    heights += [0.4] + [NILEARN_ROW_HEIGHT] * 4
    labels += ["anat_hdr"] + [f"anat_{i}" for i in range(4)]

    # Registration: header + 2 nilearn rows
    heights += [0.4] + [NILEARN_ROW_HEIGHT] * 2
    labels += ["reg_hdr"] + [f"reg_{i}" for i in range(2)]

    # Functional: header + 2 nilearn + 1 motion row (3 cols)
    heights += [0.4] + [NILEARN_ROW_HEIGHT] * 2 + [PLOT_ROW_HEIGHT]
    labels += ["func_hdr", "func_0", "func_1", "motion"]

    # Metrics: header + 3 nilearn + 1 bottom row (corr + qc)
    if has_metrics:
        heights += [0.4] + [NILEARN_ROW_HEIGHT] * 3
        labels += ["met_hdr"] + [f"met_{i}" for i in range(3)]
        if has_qc:
            heights += [5.0]  # taller row for corr matrix + QC table
            labels += ["bottom"]
        else:
            heights += [3.5]
            labels += ["bottom"]
    elif has_qc:
        heights += [0.4, 4.0]
        labels += ["qc_hdr", "qc"]

    total_height = sum(heights)
    fig = plt.figure(figsize=(FIG_WIDTH, total_height))
    gs = GridSpec(
        len(heights),
        3,
        figure=fig,
        height_ratios=heights,
        hspace=0.25,
        wspace=0.3,
        top=0.98,
        bottom=0.01,
        left=0.03,
        right=0.97,
    )

    row = 0

    # -- Anatomical --
    _section_header(fig, gs[row, :], "ANATOMICAL PREPROCESSING")
    row += 1
    anat_axes = [fig.add_subplot(gs[row + i, :]) for i in range(4)]
    plot_anatomical(manifest, anat_axes)
    row += 4

    # -- Registration --
    _section_header(fig, gs[row, :], "REGISTRATION")
    row += 1
    reg_axes = [fig.add_subplot(gs[row + i, :]) for i in range(2)]
    plot_registration(manifest, reg_axes)
    row += 2

    # -- Functional --
    _section_header(fig, gs[row, :], "FUNCTIONAL PREPROCESSING")
    row += 1
    bold_axes = [fig.add_subplot(gs[row + i, :]) for i in range(2)]
    plot_functional_bold(manifest, bold_axes)
    row += 2

    # Motion: 3 side-by-side plots
    motion_axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
    plot_motion(manifest, motion_axes)
    row += 1

    # -- Metrics --
    if has_metrics:
        _section_header(
            fig, gs[row, :], "DERIVATIVE METRICS" + (" & QC" if has_qc else "")
        )
        row += 1
        met_axes = [fig.add_subplot(gs[row + i, :]) for i in range(3)]
        plot_metrics_maps(manifest, met_axes)
        row += 3

        # Bottom row: correlation matrix + QC table side by side
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

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
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
        help="Path to .last_run.json manifest (default: tests/full_pipeline/.last_run.json)",
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
