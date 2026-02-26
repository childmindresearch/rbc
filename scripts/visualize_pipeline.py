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
from nilearn import image, plotting


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


def plot_anatomical(manifest: dict, axes: list) -> None:
    """Plot brain extraction and tissue segmentation."""
    anat = manifest["anat"]

    plotting.plot_anat(
        anat["brain"],
        title="Brain extraction",
        display_mode="ortho",
        axes=axes[0],
    )
    plotting.plot_roi(
        anat["wm_mask"],
        bg_img=anat["brain"],
        title="WM mask",
        display_mode="ortho",
        alpha=0.4,
        axes=axes[1],
    )
    plotting.plot_roi(
        anat["gm_mask"],
        bg_img=anat["brain"],
        title="GM mask",
        display_mode="ortho",
        alpha=0.4,
        axes=axes[2],
    )
    plotting.plot_roi(
        anat["csf_mask"],
        bg_img=anat["brain"],
        title="CSF mask",
        display_mode="ortho",
        alpha=0.4,
        axes=axes[3],
    )


def plot_functional(manifest: dict, axes: list) -> None:
    """Plot template BOLD, cleaned BOLD, and motion parameters."""
    func = manifest["func"]

    tmpl_mean = image.mean_img(func["template_bold"])
    plotting.plot_epi(
        tmpl_mean,
        title="Template BOLD (mean)",
        display_mode="ortho",
        vmax=_robust_vmax(tmpl_mean),
        axes=axes[0],
    )
    # Cleaned BOLD is demeaned — show temporal std instead
    clean_std = image.math_img("np.std(img, axis=-1)", img=func["cleaned_bold"])
    plotting.plot_epi(
        clean_std,
        title="Cleaned BOLD (temporal std)",
        display_mode="ortho",
        vmax=_robust_vmax(clean_std),
        axes=axes[1],
    )

    # Rotation parameters
    motion = np.loadtxt(func["motion_params"])
    ax = axes[2]
    for i, label in enumerate(["rot_x", "rot_y", "rot_z"]):
        ax.plot(np.degrees(motion[:, i]), label=label)
    ax.set_ylabel("Rotation (deg)")
    ax.set_xlabel("Volume")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Motion parameters (rotation)")

    # Translation parameters
    ax2 = axes[3]
    for i, label in enumerate(["trans_x", "trans_y", "trans_z"], start=3):
        ax2.plot(motion[:, i], label=label)
    ax2.set_ylabel("Translation (mm)")
    ax2.set_xlabel("Volume")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("Motion parameters (translation)")

    # RMS displacement
    rms_rel = np.loadtxt(func["rms_rel"])
    ax3 = axes[4]
    ax3.plot(rms_rel, color="tab:orange")
    ax3.axhline(0.2, color="tab:red", ls="--", alpha=0.6, label="0.2 mm threshold")
    ax3.set_ylabel("RMS (mm)")
    ax3.set_xlabel("Volume")
    ax3.set_title("Relative RMS displacement")
    ax3.legend(fontsize=7, loc="upper right")


def plot_registration(manifest: dict, axes: list) -> None:
    """Plot registration quality overlays."""
    func = manifest["func"]
    template_brain_mask = manifest["template_brain_mask"]

    plotting.plot_roi(
        func["bold_mask"],
        bg_img=func["skull_stripped_bold"],
        title="BOLD mask on native BOLD ref",
        display_mode="ortho",
        alpha=0.3,
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
        axes=axes[1],
    )


def plot_metrics(manifest: dict, axes: list) -> None:
    """Plot ALFF, ReHo, and correlation matrix from metrics outputs."""
    metrics = manifest["metrics"]
    template_brain_mask = manifest["template_brain_mask"]

    plotting.plot_stat_map(
        metrics["alff_zscored"],
        bg_img=metrics["alff"],
        title="ALFF (z-scored)",
        display_mode="ortho",
        threshold=1.5,
        axes=axes[0],
    )
    plotting.plot_stat_map(
        metrics["falff_zscored"],
        bg_img=metrics["falff"],
        title="fALFF (z-scored)",
        display_mode="ortho",
        threshold=1.5,
        axes=axes[1],
    )
    plotting.plot_stat_map(
        metrics["reho_zscored"],
        title="ReHo (z-scored)",
        display_mode="ortho",
        threshold=1.5,
        axes=axes[2],
    )

    # Correlation matrix
    corr = np.loadtxt(metrics["correlation_matrix"], delimiter="\t")
    ax = axes[3]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title("FC correlation matrix")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_qc(manifest: dict, ax: plt.Axes) -> None:
    """Render QC metrics as a summary table."""
    qc = manifest["qc"]
    metrics = qc["metrics"]
    passed = qc["passed"]

    # Select numeric metrics for display
    display_keys = [
        "meanFD",
        "relMeansRMSMotion",
        "relMaxRMSMotion",
        "nVolCensored",
        "meanDVInit",
        "meanDVFinal",
        "motionDVCorrInit",
        "motionDVCorrFinal",
        "coregDice",
        "coregJaccard",
        "coregCrossCorr",
        "coregCoverage",
        "normDice",
        "normJaccard",
        "normCrossCorr",
        "normCoverage",
    ]

    cell_text = []
    for key in display_keys:
        val = metrics[key]
        cell_text.append([key, f"{val:.4f}" if isinstance(val, float) else str(val)])

    ax.axis("off")
    status = "PASSED" if passed else "FAILED"
    color = "green" if passed else "red"
    ax.set_title(f"QC Summary — {status}", fontsize=12, fontweight="bold", color=color)
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)


def build_report(manifest: dict, output: Path) -> None:
    """Build a multi-panel visualization report."""
    has_metrics = "metrics" in manifest
    has_qc = "qc" in manifest

    n_rows = 11 + (4 if has_metrics else 0) + (1 if has_qc else 0)
    fig = plt.figure(figsize=(18, 2.5 * n_rows))
    fig.suptitle("RBC Full Pipeline Report", fontsize=16, fontweight="bold", y=0.995)
    gs = fig.add_gridspec(n_rows, 1, hspace=0.45, top=0.98, bottom=0.02)

    row = 0

    # -- Anatomical (4 rows) --
    anat_axes = [fig.add_subplot(gs[row + i]) for i in range(4)]
    plot_anatomical(manifest, anat_axes)
    row += 4

    # -- Registration (2 rows) --
    reg_axes = [fig.add_subplot(gs[row + i]) for i in range(2)]
    plot_registration(manifest, reg_axes)
    row += 2

    # -- Functional (5 rows: 2 nilearn + 3 motion) --
    func_axes = [fig.add_subplot(gs[row + i]) for i in range(5)]
    plot_functional(manifest, func_axes)
    row += 5

    # -- Metrics (4 rows) --
    if has_metrics:
        metrics_axes = [fig.add_subplot(gs[row + i]) for i in range(3)]
        corr_ax = fig.add_subplot(gs[row + 3])
        plot_metrics(manifest, metrics_axes + [corr_ax])
        row += 4

    # -- QC table (1 row) --
    if has_qc:
        qc_ax = fig.add_subplot(gs[row])
        plot_qc(manifest, qc_ax)
        row += 1

    fig.savefig(output, dpi=150, bbox_inches="tight")
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
