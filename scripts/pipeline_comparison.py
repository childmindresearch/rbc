# /// script
# dependencies = ["nibabel>=5.0", "numpy>=1.26", "scipy>=1.13"]
# requires-python = ">=3.12"
# ///
"""Compare intermediate steps between RBC and CPAC.

Reads the RBC manifest from .last_run.json and walks the C-PAC output directory
to find matching files for comparison.

Usage::

    uv run pipeline_comparison.py path/to/.last_run.json /path/to/cpac --mni-mask
    uv run pipeline_comparison.py path/to/.last_run.json /path/to/cpac --mni-mask --output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

_THRESHOLD=0.97  # adjust threshold

# -- Manifest loading --


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        print(f"Manifest not found: {path}", file=sys.stderr)
        print("Run the full-pipeline tests first:", file=sys.stderr)
        print("  uv run pytest tests/full_pipeline/ -v", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _find_sub_ses(cpac_dir: Path) -> tuple[str, str]:
    pipelines = list((cpac_dir / "output").glob("pipeline_*"))
    if not pipelines:
        print(
            f"No pipeline_* directory found under {cpac_dir / 'output'}",
            file=sys.stderr,
        )
        sys.exit(1)
    pipeline = pipelines[0]
    subs = [d for d in pipeline.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    if not subs:
        print(f"No sub-* directory found under {pipeline}", file=sys.stderr)
        sys.exit(1)
    sub_dir = subs[0]
    sess = [d for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
    if not sess:
        print(f"No ses-* directory found under {sub_dir}", file=sys.stderr)
        sys.exit(1)
    return sub_dir.name, sess[0].name


def _find_task_run(func_dir: Path) -> tuple[str, str]:
    for f in func_dir.iterdir():
        name = f.name
        if "task-" in name and "run-" in name:
            task = name.split("task-")[1].split("_")[0]
            run = name.split("run-")[1].split("_")[0]
            return task, run
    print(f"Could not detect task/run in {func_dir}", file=sys.stderr)
    sys.exit(1)


def _build_cpac_manifest(cpac_dir: Path, reg: str) -> dict:
    """Walk C-PAC outputs and build a manifest dict mirroring the RBC manifest structure."""
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
            "cleaned_bold": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-preproc_bold.nii.gz"
            ),
            "motion_params": str(
                func_dir / f"{prefix_func}_desc-movementParameters_motion.1D"
            ),
        },
        "metrics": {
            "alff_zscored": str(
                func_dir
                / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_alff.nii.gz"
            ),
            "falff_zscored": str(
                func_dir
                / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_falff.nii.gz"
            ),
            "reho_zscored": str(
                func_dir
                / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-smZstd_reho.nii.gz"
            ),
        },
    }


# -- Metrics --


def _load_nifti(path: Path) -> np.ndarray:
    return nib.nifti1.load(path).get_fdata()


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def _spatial_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    r, _ = pearsonr(a[mask.astype(bool)], b[mask.astype(bool)])
    return float(r)


def _voxelwise_correlation(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    mask = mask.astype(bool)
    a_voxels = a[mask].astype(np.float32)
    b_voxels = b[mask].astype(np.float32)

    a_voxels = a_voxels - a_voxels.mean(axis=1, keepdims=True)
    b_voxels = b_voxels - b_voxels.mean(axis=1, keepdims=True)

    numerator = (a_voxels * b_voxels).sum(axis=1)
    denominator = np.sqrt((a_voxels**2).sum(axis=1) * (b_voxels**2).sum(axis=1))
    r = np.zeros(mask.sum())
    valid_voxels = denominator > 0
    r[valid_voxels] = numerator[valid_voxels] / denominator[valid_voxels]
    r = np.clip(r, -1, 1)

    return float(np.nanmean(r)), float(np.nanmedian(r))


def compare_masks(rbc_manifest: dict, cpac_manifest: dict) -> dict:
    """Compare brain, CSF, WM, and GM masks using Dice coefficient."""
    results = {}
    for key in ["brain_mask", "csf_mask", "wm_mask", "gm_mask"]:
        rbc_mask = _load_nifti(Path(rbc_manifest["anat"][key]))
        cpac_mask = _load_nifti(Path(cpac_manifest["anat"][key]))
        d = _dice(rbc_mask, cpac_mask)
        results[key] = {"dice": round(d, 6), "passed": d >= _THRESHOLD}
    return results


def compare_bold(rbc_manifest: dict, cpac_manifest: dict) -> dict:
    """Compare template and cleaned BOLD using voxelwise correlation.""" #same space?
    results = {}
    mask = _load_nifti(Path(rbc_manifest["template_brain_mask"]))
    for key in ["template_bold", "cleaned_bold"]:
        rbc_data = _load_nifti(Path(rbc_manifest["func"][key]))
        cpac_data = _load_nifti(Path(cpac_manifest["func"][key]))
        if rbc_data.shape != cpac_data.shape:
            raise ValueError(
                f"Shape mismatch: {rbc_data.shape} vs {cpac_data.shape}"
            )
        mean_r, median_r = _voxelwise_correlation(rbc_data, cpac_data, mask)
        results[key] = {
            "mean_r": round(mean_r, 6),
            "median_r": round(median_r, 6),
            "passed": mean_r >= _THRESHOLD,
        }
    return results


def compare_maps(rbc_manifest: dict, cpac_manifest: dict, template_mask: Path) -> dict:
    """Compare ALFF, fALFF, and ReHo maps using spatial correlation within template mask."""
    results = {}
    mask = _load_nifti(template_mask)
    for key in ["alff_zscored", "falff_zscored", "reho_zscored"]:
        rbc_data = _load_nifti(Path(rbc_manifest["metrics"][key]))
        cpac_data = _load_nifti(Path(cpac_manifest["metrics"][key]))
        r = _spatial_correlation(rbc_data, cpac_data, mask)
        results[key] = {"r": round(r, 6), "passed": r >= _THRESHOLD}
    return results


def compare_motion(rbc_manifest: dict, cpac_manifest: dict) -> dict:
    """Compare motion parameters using Pearson correlation."""
    results = {}
    labels = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]  # same order?
    rbc_motion = np.loadtxt(rbc_manifest["func"]["motion_params"])
    cpac_motion = np.loadtxt(cpac_manifest["func"]["motion_params"])

    if rbc_motion.shape != cpac_motion.shape:
        raise ValueError(
            f"Shape mismatch: {rbc_motion.shape} vs {cpac_motion.shape}"
        )

    rs = {}
    for i, label in enumerate(labels):
        r, _ = pearsonr(rbc_motion[:, i], cpac_motion[:, i])
        rs[label] = round(float(r), 6)

    results["motion_params"] = {
        "per_parameter": rs,
        "mean_r": round(float(np.mean(list(rs.values()))), 6),
        "min_r": round(float(np.min(list(rs.values()))), 6),
        "passed": bool(np.mean(list(rs.values())) >= _THRESHOLD),
    }
    return results


def main() -> None:
    default_manifest = (
        Path(__file__).parent.parent / "tests" / "full_pipeline" / ".last_run.json"
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
        type=Path,
        help="Path to C-PAC output dataset (default: tests/data/cpac_outputs/ds000001)",
    )
    parser.add_argument(
        "--reg",
        default="36Parameter",
        help="C-PAC regression strategy (default: 36Parameter)",
    )
    parser.add_argument(
        "--template-mask",
        type=Path,
        required=True,
        help="Template brain mask for functional map comparisons.", # figure out what this is
    )  
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rbc_manifest = _load_manifest(args.manifest)
    cpac_manifest = _build_cpac_manifest(args.cpac_dir, args.reg)

    results = {}
    results.update(compare_masks(rbc_manifest, cpac_manifest))
    results.update(compare_bold(rbc_manifest, cpac_manifest))
    results.update(compare_maps(rbc_manifest, cpac_manifest, args.template_mask))
    results.update(compare_motion(rbc_manifest, cpac_manifest))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))

    if any("error" in v or not v.get("passed") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
