# /// script
# dependencies = ["nilearn>=0.11", "matplotlib>=3.9", "nibabel>=5.0", "numpy>=1.26"]
# requires-python = ">=3.12"
# ///
"""Visualize C-PAC RBC outputs using the same report layout as visualize_pipeline.py.

Walks a C-PAC output directory, builds a manifest dict, and delegates to
``build_report()`` from ``visualize_pipeline.py`` to produce a multi-panel PNG.

Usage::

    uv run scripts/visualize_cpac.py                                    # defaults
    uv run scripts/visualize_cpac.py tests/data/cpac_outputs/ds000001   # explicit dir
    uv run scripts/visualize_cpac.py --reg aCompCor --output report.png # options
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow importing the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_pipeline import build_report


def _find_sub_ses(cpac_dir: Path) -> tuple[str, str]:
    """Auto-detect the first subject/session under output/pipeline_*/."""
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
    print(f"Could not detect task/run in {func_dir}", file=sys.stderr)
    sys.exit(1)


def _parse_qc_tsv(tsv_path: Path) -> dict:
    """Parse the XCP-D quality TSV into a QC manifest entry."""
    with tsv_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        row = next(reader)

    float_keys = [
        "meanFD",
        "relMeansRMSMotion",
        "relMaxRMSMotion",
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
    int_keys = ["nVolCensored"]

    metrics: dict[str, float | int] = {}
    for k in float_keys:
        metrics[k] = float(row[k])
    for k in int_keys:
        metrics[k] = int(row[k])

    passed = metrics["meanFD"] <= 0.2 and metrics["normCrossCorr"] >= 0.8
    return {"metrics": metrics, "passed": passed}


def build_manifest(cpac_dir: Path, reg: str, atlas: str) -> dict:
    """Walk C-PAC outputs and build a manifest dict for build_report()."""
    sub, ses = _find_sub_ses(cpac_dir)
    pipeline = next((cpac_dir / "output").glob("pipeline_*"))
    base = pipeline / sub / ses
    anat_dir = base / "anat"
    func_dir = base / "func"

    prefix_anat = f"{sub}_{ses}"
    task, run = _find_task_run(func_dir)
    prefix_func = f"{sub}_{ses}_task-{task}_run-{run}"

    manifest: dict = {
        "anat": {
            "brain": str(anat_dir / f"{prefix_anat}_desc-preproc_T1w.nii.gz"),
            "wm_mask": str(anat_dir / f"{prefix_anat}_label-WM_mask.nii.gz"),
            "gm_mask": str(anat_dir / f"{prefix_anat}_label-GM_mask.nii.gz"),
            "csf_mask": str(anat_dir / f"{prefix_anat}_label-CSF_mask.nii.gz"),
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
            "rms_rel": str(func_dir / f"{prefix_func}_desc-FDPower_motion.1D"),
        },
        "template_brain_mask": str(
            func_dir / f"{prefix_func}_space-MNI152NLin6ASym_desc-bold_mask.nii.gz"
        ),
        "metrics": {
            "alff_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-smZstd_alff.nii.gz"
            ),
            "falff_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-smZstd_falff.nii.gz"
            ),
            "reho_zscored": str(
                func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-smZstd_reho.nii.gz"
            ),
            "correlation_matrix": str(
                func_dir
                / f"{prefix_func}_atlas-{atlas}_space-MNI152NLin6ASym_reg-{reg}_"
                "desc-PearsonNilearn_correlations.tsv"
            ),
        },
    }

    # QC
    qc_tsv = (
        func_dir / f"{prefix_func}_space-MNI152NLin6ASym_reg-{reg}_desc-xcp_quality.tsv"
    )
    if qc_tsv.exists():
        manifest["qc"] = _parse_qc_tsv(qc_tsv)

    return manifest


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "data"
        / "cpac_outputs"
        / "ds000001"
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
        help="Regression strategy (default: 36Parameter)",
    )
    parser.add_argument(
        "--atlas",
        default="Schaefer2018p200n17",
        help="Atlas name (default: Schaefer2018p200n17)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("cpac_report.png"),
        help="Output image path (default: cpac_report.png)",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.cpac_dir, args.reg, args.atlas)

    # Verify all files exist before rendering.
    missing = []
    for section, value in manifest.items():
        if section == "qc":
            continue
        if isinstance(value, dict):
            for key, path in value.items():
                if not Path(path).exists():
                    missing.append(f"{section}.{key}: {path}")
        elif not Path(value).exists():
            missing.append(f"{section}: {value}")
    if missing:
        print("Missing files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    build_report(manifest, args.output)


if __name__ == "__main__":
    main()
