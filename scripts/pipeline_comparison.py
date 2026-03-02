# /// script
# dependencies = ["nibabel>=5.0", "numpy>=1.26", "scipy>=1.13", "matplotlib>=3.9"]
# requires-python = ">=3.12"
# ///
"""Compare intermediate outputs between RBC and CPAC.

Reads the RBC manifest from .last_run.json and walks the C-PAC output directory
to find matching files for comparison.

Computes:
  - Dice overlap for brain and tissue masks
  - Pearson r for motion parameters
  - Voxelwise temporal correlation for BOLD
  - Spatial correlation for ALFF, fALFF, and ReHo maps
  - Pearson r for ROI timeseries
  - RMSE for FC correlation matrices

Usage:
    uv run pipeline_comparison.py path/to/.last_run.json /path/to/cpac
        --output /path/to/report.md
    uv run pipeline_comparison.py path/to/.last_run.json /path/to/cpac
        --output /path/to/report.md --plots /path/to/plots.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr, zscore

# TODO: adjust thresholds for each metric
_THRESHOLD = 0.97
_RMSE_THRESHOLD = 0.05

_ANAT_KEYS = ["brain_mask", "csf_mask", "wm_mask", "gm_mask"]
_MAP_KEYS = ["alff_zscored", "falff_zscored", "reho_zscored"]
_BOLD_KEYS = ["template_bold"]


# -- Loading manifest --
def _load_manifest(path: Path) -> dict:
    if not path.exists():
        sys.exit(1)
    return json.loads(path.read_text())


def _find_sub_ses(cpac_dir: Path) -> tuple[str, str]:
    """Auto-detect the first subject/session under output/pipeline_*/."""
    pipelines = list((cpac_dir / "output").glob("pipeline_*"))
    if not pipelines:
        sys.exit(1)
    pipeline = pipelines[0]
    subs = [d for d in pipeline.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    if not subs:
        sys.exit(1)
    sub_dir = subs[0]
    sess = [d for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
    if not sess:
        sys.exit(1)
    ses_dir = sess[0]
    return sub_dir.name, ses_dir.name


def _find_task_run(func_dir: Path) -> tuple[str, str]:
    """Auto-detect task and run from the first BOLD file in func/."""
    for f in func_dir.iterdir():
        name = f.name
        if "task-" in name and "run-" in name:
            task = name.split("task-")[1].split("_")[0]
            run = name.split("run-")[1].split("_")[0]
            return task, run
    sys.exit(1)


def _build_cpac_manifest(cpac_dir: Path, reg: str, atlas: str) -> dict:
    """Walk C-PAC outputs and build a manifest dict mirroring RBC manifest."""
    sub, ses = _find_sub_ses(cpac_dir)
    pipeline = next((cpac_dir / "output").glob("pipeline_*"))
    base = pipeline / sub / ses
    anat_dir = base / "anat"
    func_dir = base / "func"

    prefix_anat = f"{sub}_{ses}"
    task, run = _find_task_run(func_dir)
    prefix_func = f"{sub}_{ses}_task-{task}_run-{run}"

    return {
        "anat": {
            "brain_mask": str(anat_dir / f"{prefix_anat}_desc-brain_mask.nii.gz"),
            "csf_mask": str(anat_dir / f"{prefix_anat}_label-CSF_mask.nii.gz"),
            "wm_mask": str(anat_dir / f"{prefix_anat}_label-WM_mask.nii.gz"),
            "gm_mask": str(anat_dir / f"{prefix_anat}_label-GM_mask.nii.gz"),
        },
        "func": {
            "skull_stripped_bold": str(
                func_dir / f"{prefix_func}_desc-mean_bold.nii.gz"
            ),
            "bold_mask": str(func_dir / f"{prefix_func}_desc-brain_mask.nii.gz"),
            "template_bold": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_desc-head_bold.nii.gz"
            ),
            "motion_params": str(
                func_dir / f"{prefix_func}_desc-movementParameters_motion.1D"
            ),
        },
        "template_brain_mask": str(
            func_dir / f"{prefix_func}_space-MNI152NLin6ASym_desc-bold_mask.nii.gz"
        ),
        "metrics": {
            "alff_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_"
                "alff.nii.gz"
            ),
            "falff_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_"
                "falff.nii.gz"
            ),
            "reho_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_"
                "reho.nii.gz"
            ),
            "correlation_matrix": str(
                func_dir
                / f"{prefix_func}_atlas-{atlas}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-PearsonNilearn_correlations.tsv"
            ),
            "timeseries": str(
                func_dir
                / f"{prefix_func}_atlas-{atlas}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-Mean_timeseries.1D"
            ),
        },
    }


def _load_nifti(path: Path) -> np.ndarray:
    return nib.nifti1.load(path).get_fdata()


# -- Metrics --
def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def _spatial_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    r, _ = pearsonr(a[mask.astype(bool)], b[mask.astype(bool)])
    return float(r)


def _voxelwise_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a_voxels = a[mask.astype(bool)]
    b_voxels = b[mask.astype(bool)]
    a_z = zscore(a_voxels, axis=1)
    b_z = zscore(b_voxels, axis=1)
    r_values = (a_z * b_z).mean(axis=1)
    return float(np.nanmean(r_values))


def _correlate_timeseries(a: np.ndarray, b: np.ndarray) -> float:
    n_rois = a.shape[1]
    rs = [pearsonr(a[:, i], b[:, i])[0] for i in range(n_rois)]
    return float(np.mean(rs))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -- Comparisons --
def compare_masks(rbc: dict, cpac: dict) -> dict:
    """Brain extraction + tissue segmentation - Dice overlap per mask."""
    results = {}
    for key in _ANAT_KEYS:
        rbc_arr = _load_nifti(rbc["anat"][key])
        cpac_arr = _load_nifti(cpac["anat"][key])
        d = _dice(rbc_arr, cpac_arr)
        results[key] = round(d, 6)
    return results


def compare_motion(rbc: dict, cpac: dict) -> dict:
    """Motion correction — Pearson r per parameter."""
    results = {}
    labels = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    # TODO: verify column order matches between RBC and CPAC
    rbc_motion = np.loadtxt(rbc["func"]["motion_params"])
    cpac_motion = np.loadtxt(cpac["func"]["motion_params"])
    if rbc_motion.shape != cpac_motion.shape:
        raise ValueError(f"Shape mismatch: {rbc_motion.shape} vs {cpac_motion.shape}")
    for i, label in enumerate(labels):
        r, _ = pearsonr(rbc_motion[:, i], cpac_motion[:, i])
        results[label] = round(float(r), 6)
    return results


def compare_bold(rbc: dict, cpac: dict) -> dict:
    """BOLD images - Voxelwise temporal correlation within template mask."""
    results = {}
    mask = _load_nifti(cpac["template_brain_mask"])
    for key in _BOLD_KEYS:
        rbc_arr = _load_nifti(rbc["func"][key])
        cpac_arr = _load_nifti(cpac["func"][key])
        if rbc_arr.shape != cpac_arr.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {rbc_arr.shape} vs {cpac_arr.shape}"
            )
        results[key] = round(_voxelwise_correlation(rbc_arr, cpac_arr, mask), 6)
    return results


def compare_maps(rbc: dict, cpac: dict) -> dict:
    """ALFF, fALFF, ReHo - spatial correlation within brain mask."""
    results = {}
    mask = _load_nifti(cpac["template_brain_mask"])
    for key in _MAP_KEYS:
        rbc_arr = _load_nifti(rbc["metrics"][key])
        cpac_arr = _load_nifti(cpac["metrics"][key])
        results[key] = round(_spatial_correlation(rbc_arr, cpac_arr, mask), 6)
    return results


def compare_timeseries(rbc: dict, cpac: dict) -> dict:
    """Timeseries extraction - mean Pearson r across ROIs."""
    # RBC: (n_rois, n_timepoints) - transpose to (n_timepoints, n_rois)
    rbc_ts = np.loadtxt(rbc["metrics"]["timeseries"], delimiter="\t").T
    # CPAC: (n_timepoints, n_rois), header line starting with #
    cpac_ts = np.loadtxt(cpac["metrics"]["timeseries"], delimiter=",", skiprows=1)
    if rbc_ts.shape != cpac_ts.shape:
        raise ValueError(
            f"Timeseries shape mismatch: {rbc_ts.shape} vs {cpac_ts.shape}"
        )
    return {"timeseries": round(_correlate_timeseries(rbc_ts, cpac_ts), 6)}


def compare_correlation_matrix(rbc: dict, cpac: dict) -> dict:
    """FC correlation matrices - element-wise RMSE."""
    rbc_cm = np.loadtxt(rbc["metrics"]["correlation_matrix"], delimiter="\t")
    cpac_cm = np.loadtxt(cpac["metrics"]["correlation_matrix"], delimiter="\t")
    if rbc_cm.shape != cpac_cm.shape:
        raise ValueError(f"Shape mismatch: {rbc_cm.shape} vs {cpac_cm.shape}")
    return {"correlation_matrix": round(_rmse(rbc_cm, cpac_cm), 6)}


# -- Report --
_REPORT_ROWS = [
    ("brain_mask", _THRESHOLD),
    ("csf_mask", _THRESHOLD),
    ("wm_mask", _THRESHOLD),
    ("gm_mask", _THRESHOLD),
    ("rot_x", _THRESHOLD),
    ("rot_y", _THRESHOLD),
    ("rot_z", _THRESHOLD),
    ("trans_x", _THRESHOLD),
    ("trans_y", _THRESHOLD),
    ("trans_z", _THRESHOLD),
    ("template_bold", _THRESHOLD),
    ("alff_zscored", _THRESHOLD),
    ("falff_zscored", _THRESHOLD),
    ("reho_zscored", _THRESHOLD),
    ("timeseries", _THRESHOLD),
    ("correlation_matrix", _RMSE_THRESHOLD),
]


def _make_report(results: dict) -> str:
    lines = [
        "# RBC vs CPAC Comparison Report",
        "",
        "| Step | Value | Threshold | Status |",
        "|------|-------|-----------|--------|",
    ]

    for key, thr in _REPORT_ROWS:
        if key not in results:
            continue
        val = results[key]
        passed = val <= thr if key == "correlation_matrix" else val >= thr
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {key} | {val:.4f} | {thr} | {status} |")

    return "\n".join(lines) + "\n"


# -- Plots --
def _draw_map_row(
    axes: np.ndarray, rbc_arr: np.ndarray, cpac_arr: np.ndarray, label: str
) -> None:
    """Single row: RBC, CPAC, and Diff for a nifti map."""
    mid_z = rbc_arr.shape[2] // 2
    rbc_sl = rbc_arr[:, :, mid_z].T
    cpac_sl = cpac_arr[:, :, mid_z].T
    diff = np.abs(rbc_sl - cpac_sl)

    # Auto-scale based on data
    brain = (rbc_sl != 0) | (cpac_sl != 0)
    vmin = rbc_sl[brain].min() if brain.any() else 0
    vmax = rbc_sl[brain].max() if brain.any() else 1

    axes[0].imshow(rbc_sl, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_ylabel(label, fontsize=12, fontweight="bold")
    axes[1].imshow(cpac_sl, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    im = axes[2].imshow(diff, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def generate_combined_report(rbc: dict, cpac: dict, out_path: Path) -> None:
    """Creates a single dashboard PNG."""
    fig, all_axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle("RBC vs CPAC Plots", fontsize=16, fontweight="bold")

    # Titles for the columns
    all_axes[0, 0].set_title("RBC", fontsize=14)
    all_axes[0, 1].set_title("CPAC", fontsize=14)
    all_axes[0, 2].set_title("Absolute Difference", fontsize=14)

    # 1-3. Maps: ALFF, fALFF, ReHo
    for i, key in enumerate(_MAP_KEYS):
        rbc_arr = _load_nifti(rbc["metrics"][key])
        cpac_arr = _load_nifti(cpac["metrics"][key])
        _draw_map_row(all_axes[i], rbc_arr, cpac_arr, key)

    # 4. Correlation Matrix
    rbc_cm = np.loadtxt(rbc["metrics"]["correlation_matrix"], delimiter="\t")
    cpac_cm = np.loadtxt(cpac["metrics"]["correlation_matrix"], delimiter="\t")

    all_axes[3, 0].imshow(rbc_cm, cmap="RdBu_r", vmin=-1, vmax=1)
    all_axes[3, 0].set_ylabel("FC Matrix", fontsize=12, fontweight="bold")
    all_axes[3, 1].imshow(cpac_cm, cmap="RdBu_r", vmin=-1, vmax=1)
    diff_im = all_axes[3, 2].imshow(np.abs(rbc_cm - cpac_cm), cmap="viridis")
    plt.colorbar(diff_im, ax=all_axes[3, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    """Entry point."""
    default_manifest = (
        Path(__file__).parent.parent / "tests" / "full_pipeline" / ".last_run.json"
    )
    default_dir = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "data"
        / "cpac_outputs"
        / "ds000001"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=default_manifest,
        help="Path to RBC .last_run.json (default: tests/full_pipeline/.last_run.json)",
    )
    parser.add_argument(
        "cpac_dir",
        nargs="?",
        type=Path,
        default=default_dir,
        help="Path to C-PAC output dataset (default: tests/data/cpac_outputs/ds000001)",
    )
    parser.add_argument(
        "--reg",
        default="36Parameter",
        help="C-PAC regression strategy (default: 36Parameter)",
    )
    parser.add_argument(
        "--atlas",
        default="Schaefer2018p200n17",
        help="Atlas name (default: Schaefer2018p200n17)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Save markdown report (e.g., report.md).",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        help="Path to save diagnostic plots (e.g., plots.png).",
    )
    args = parser.parse_args()

    rbc = _load_manifest(args.manifest)
    cpac = _build_cpac_manifest(args.cpac_dir, args.reg, args.atlas)

    results = {}
    results.update(compare_masks(rbc, cpac))
    results.update(compare_motion(rbc, cpac))
    results.update(compare_bold(rbc, cpac))
    results.update(compare_maps(rbc, cpac))
    results.update(compare_timeseries(rbc, cpac))
    results.update(compare_correlation_matrix(rbc, cpac))

    report = _make_report(results)
    args.output.write_text(report)

    if args.plots:
        generate_combined_report(rbc, cpac, args.plots)


if __name__ == "__main__":
    main()
