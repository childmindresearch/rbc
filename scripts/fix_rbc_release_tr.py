r"""Fix the TR bandpass bug in a downloaded RBC data release.

For each functional run in ``--input-dir``, writes a corrected cleaned BOLD
(and downstream derivatives) into a parallel BIDS-derivatives tree under
``--output-dir``:

1. Read the correct TR from the native-space ``desc-preproc_bold.nii.gz``
   header (or accept ``--tr-override``).
2. Patch the template-space ``desc-head_bold.nii.gz`` header, which ships
   with pixdim[4]=0.0 (zeroed by ANTs single-step resampling and later
   silently coerced to 1.0 by AFNI inside C-PAC, which is what drove the
   bandpass off by a factor of two).
3. Re-run nuisance regression + bandpass via AFNI ``3dTproject -bandpass``,
   using C-PAC's raw (unfiltered) regressors as the ort matrix.
4. Re-run nuisance regression without bandpass (input to ALFF/fALFF).
5. Recompute downstream metrics (ALFF, fALFF, ReHo at sm6 / smZstd / zstd
   variants; atlas-based mean timeseries and Pearson/Partial connectivity
   matrices) from the fixed BOLDs using RBC's library functions.

Only the bandpass bug (#4 in ``docs/cpac_comparison.md``) is addressed.
Atlas timeseries write as tab-separated ``.1D`` in ``(T, n_rois)`` order
matching the release. Connectivity ``.tsv`` files are bare numeric.
JSON sidecars are not regenerated.

Usage::

    uv run scripts/fix_rbc_release_tr.py \\
        --input-dir /path/to/rbc_release --output-dir /path/to/fixed \\
        [--participant-label sub-X ...] [--bandpass 0.01 0.1] \\
        [--tr-override 2.0] [--runner auto] [--skip-metrics] \\
        [--work-dir /scratch] [--overwrite] [--dry-run | --verify]

Or standalone, no clone::

    uv run --script fix_rbc_release_tr.py --input-dir ... --output-dir ...

Note for local development: ``uv run scripts/...py`` resolves the PEP 723
inline metadata and provisions an isolated env from the pinned ``rbc`` SHA
below -- it does NOT use your local editable ``rbc`` install. To test
script changes against your working tree, run via the project venv
directly: ``.venv/Scripts/python scripts/fix_rbc_release_tr.py ...`` (or
``uv run --no-project ...``).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rbc @ git+https://github.com/childmindresearch/rbc.git@7cfa758",
#     "nibabel>=5.3.3",
#     "nilearn>=0.10.4",
#     "numpy>=2.0",
# ]
# ///

# ruff: noqa: T201

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from rbc.bids import parse_bids_name
from rbc.core.functional.nuisance import (
    apply_regression,
    apply_regression_bandpass,
    bandpass_regressor_file,
)
from rbc.core.metrics.standardization import compute_zscore
from rbc.core.niwrap import setup_runner
from rbc.workflows.metrics import single_session_metrics
from rbc_resources import ATLAS_REGISTRY, get_atlas

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from rbc.workflows.metrics import MetricsOutputs
    from rbc_resources import AtlasName

LOG = logging.getLogger("rbc.fix_release_tr")

REG_SETS = ("36Parameter", "aCompCor")

# Release-style atlas name -> ATLAS_REGISTRY short name. Resolved via
# ``get_atlas`` so file renames inside ``rbc_resources`` don't silently break.
RELEASE_ATLASES: dict[str, AtlasName] = {
    "AAL": "aal",
    "Brodmann": "brodmann",
    "CC200": "craddock_200",
    "CC400": "craddock_400",
    "Glasser": "glasser",
    "HOCPATh25": "harvard_oxford_cortical",
    "HOSPATh25": "harvard_oxford_subcortical",
    "Juelich": "juelich",
    "Schaefer2018p200n17": "schaefer_200",
    # Schaefer2018p300n17 deliberately omitted: the upstream
    # ``ReproBrainChart/sourcedata-atlases @ rbc-labels`` atlas has 619 unique
    # labels instead of 300, and the release used a different (correct) file.
    # Re-add once upstream ships a fixed atlas.
    "Schaefer2018p400n17": "schaefer_400",
    "Schaefer2018p1000n17": "schaefer_1000",
    "Slab": "slab_907",
    "Yeo7": "yeo_7",
    "Yeo7liberal": "yeo_7_liberal",
    "Yeo17": "yeo_17",
    "Yeo17liberal": "yeo_17_liberal",
}

METRIC_VARIANTS = ("sm6", "smZstd", "zstd")
SCALAR_METRICS = ("alff", "falff", "reho")


@dataclass(frozen=True)
class Run:
    """A single functional run with resolved input paths."""

    sub: str
    ses: str | None
    task: str
    run: str | None
    space: str
    head_bold: Path
    bold_mask: Path
    native_bold: Path
    regressors: dict[str, Path]


_HEAD_BOLD_TAIL_RE = re.compile(r"_space-[^_]+_desc-head_bold\.nii\.gz$")


def _run_stem(run: Run, *, with_space: bool = False) -> str:
    """Reconstruct the BIDS stem by stripping the head_bold tail.

    Preserves every entity in the source filename (``acq-``, ``ce-``, etc.),
    not just sub/ses/task/run -- many release subjects carry ``acq-VARIANT*``
    tokens that get dropped if the stem is rebuilt from parsed entities.
    """
    base = _HEAD_BOLD_TAIL_RE.sub("", run.head_bold.name)
    if with_space:
        return f"{base}_space-{run.space}"
    return base


def discover_runs(input_dir: Path) -> Iterator[Run]:
    """Walk *input_dir* and yield one :class:`Run` per discoverable functional run.

    Discovery is anchored on ``*_space-*_desc-head_bold.nii.gz`` (the
    pre-regression template-space BOLD); each match is paired with its
    sibling native ``desc-preproc_bold`` (no ``space`` entity), template
    ``desc-bold_mask``, and raw ``reg-*_regressors.1D`` files. Runs missing
    any required input are skipped with a warning.
    """
    for head_bold in sorted(
        input_dir.glob("sub-*/**/func/*_space-*_desc-head_bold.nii.gz")
    ):
        func_dir = head_bold.parent
        ents = parse_bids_name(head_bold.name).entities
        sub = ents.get("sub")
        task = ents.get("task")
        space = ents.get("space")
        if not (sub and task and space):
            LOG.warning("Skipping %s: missing sub/task/space entity", head_bold.name)
            continue
        ses = ents.get("ses")
        run_ent = ents.get("run")
        # Strip the head_bold tail; preserves ``acq-*`` etc. See ``_run_stem``.
        stem = _HEAD_BOLD_TAIL_RE.sub("", head_bold.name)

        native_bold = func_dir / f"{stem}_desc-preproc_bold.nii.gz"
        bold_mask = func_dir / f"{stem}_space-{space}_desc-bold_mask.nii.gz"
        regressors = {
            reg: func_dir / f"{stem}_reg-{reg}_regressors.1D" for reg in REG_SETS
        }
        regressors = {k: v for k, v in regressors.items() if v.exists()}

        missing: list[str] = []
        if not native_bold.exists():
            missing.append(native_bold.name)
        elif "space" in parse_bids_name(native_bold.name).entities:
            # Defensive: a sibling with the same stem but a stray space- token
            # is not the native preproc_bold we want.
            missing.append(f"{native_bold.name} (has unexpected space entity)")
        if not bold_mask.exists():
            missing.append(bold_mask.name)
        if not regressors:
            missing.append("any reg-*_regressors.1D")
        if missing:
            LOG.warning(
                "Skipping %s: missing inputs (%s)",
                head_bold.name,
                ", ".join(missing),
            )
            continue

        yield Run(
            sub=sub,
            ses=ses,
            task=task,
            run=run_ent,
            space=space,
            head_bold=head_bold,
            bold_mask=bold_mask,
            native_bold=native_bold,
            regressors=regressors,
        )


def _detect_tr(native_bold: Path, override: float | None) -> float:
    if override is not None:
        return override
    tr = float(nib.nifti1.load(native_bold).header.get_zooms()[3])
    if not 0.1 <= tr <= 10.0:
        raise ValueError(
            f"Implausible TR ({tr}s) read from {native_bold}; "
            "pass --tr-override explicitly"
        )
    return tr


def _restore_tr(resampled: Path, source: Path) -> None:
    """Copy pixdim[4] (TR) from *source* into *resampled* NIfTI header.

    Mirrors ``rbc.core.functional.resampling._restore_tr``; inlined to keep
    the script standalone-runnable against an unmodified ``rbc @ main``.
    """
    src_img = nib.nifti1.load(source)
    res_img = nib.nifti1.load(resampled)
    data = np.asarray(res_img.dataobj)
    zooms = res_img.header.get_zooms()[:3] + src_img.header.get_zooms()[3:]
    res_img.header.set_zooms(zooms)
    nib.save(nib.Nifti1Image(data, res_img.affine, res_img.header), resampled)


def _patch_head_bold(head_bold: Path, native_bold: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / head_bold.name
    shutil.copyfile(head_bold, patched)
    _restore_tr(patched, native_bold)
    return patched


def _stage_mask(mask: Path, reference: Path, work_dir: Path) -> Path:
    """Stage *mask* with sform/qform/xyzt_units lifted from *reference*.

    The release ships ``desc-bold_mask`` with sform_code=0; AFNI outputs
    inherit SCANNER codes from the BOLD. Forcing codes alone breaks AFNI's
    grid check because the mask's qform_matrix doesn't byte-match the
    BOLD's sform_matrix. Lift the whole spatial header instead.

    Raises on shape mismatch (header rewrite can't fix it -- run 3dresample);
    warns on affine drift > 1e-3 (assumes data is co-aligned).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / mask.name
    ref_img = nib.nifti1.load(reference)
    mask_img = nib.nifti1.load(mask)

    if mask_img.shape[:3] != ref_img.shape[:3]:
        raise ValueError(
            f"Mask {mask.name} shape {mask_img.shape[:3]} != "
            f"BOLD {reference.name} shape {ref_img.shape[:3]}; "
            "run 3dresample first."
        )
    if not np.allclose(ref_img.affine, mask_img.affine, atol=1e-3):
        LOG.warning(
            "Mask %s vs BOLD %s affine drift > 1e-3; using BOLD's affine.",
            mask.name,
            reference.name,
        )

    ref_hdr = ref_img.header
    new_hdr = mask_img.header.copy()
    new_hdr.set_sform(ref_img.affine, code=int(ref_hdr["sform_code"]))
    new_hdr.set_qform(ref_img.affine, code=int(ref_hdr["qform_code"]))
    # Copy the raw xyzt_units byte; nibabel's ``set_xyzt_units`` expects raw
    # NIfTI codes (8/16/24), not the small ints from masking. AFNI reads
    # this field directly for its grid check.
    new_hdr["xyzt_units"] = ref_hdr["xyzt_units"]
    nib.save(
        nib.Nifti1Image(np.asarray(mask_img.dataobj), ref_img.affine, new_hdr),
        out,
    )
    return out


def _resolve_atlases() -> dict[str, Path]:
    """Return release-style atlas name -> resolved NIfTI path.

    Missing atlases are skipped with a warning; raises if none resolve.
    """
    out: dict[str, Path] = {}
    for release_name, short_name in RELEASE_ATLASES.items():
        if short_name not in ATLAS_REGISTRY:
            LOG.warning(
                "Atlas short name %r (release %s) not in ATLAS_REGISTRY; skipping",
                short_name,
                release_name,
            )
            continue
        try:
            out[release_name] = get_atlas(short_name)
        except FileNotFoundError as exc:
            LOG.warning("Atlas %s missing: %s", release_name, exc)
    if not out:
        raise FileNotFoundError(
            "No atlases resolved from rbc_resources; is the package installed?"
        )
    return out


def _compute_partial_correlation(timeseries_path: Path, out_path: Path) -> None:
    """Write a Nilearn-style partial-correlation matrix from a ``(T, n_rois)`` TSV."""
    from nilearn.connectome import ConnectivityMeasure

    ts = np.loadtxt(timeseries_path)
    cm = ConnectivityMeasure(kind="partial correlation")
    partial = cm.fit_transform([ts])[0]
    np.savetxt(out_path, partial, delimiter="\t")


def _metric_output_paths(
    out_func_dir: Path,
    run: Run,
    reg_set: str,
    atlases: Mapping[str, Path],
) -> list[Path]:
    """Return every release-named metric output file for a (run, regressor)."""
    stem_with_space = _run_stem(run, with_space=True)
    stem_no_space = _run_stem(run)
    paths: list[Path] = [
        out_func_dir / f"{stem_with_space}_reg-{reg_set}_desc-{variant}_{metric}.nii.gz"
        for metric in SCALAR_METRICS
        for variant in METRIC_VARIANTS
    ]
    for atl in atlases:
        base = f"{stem_no_space}_atlas-{atl}_space-{run.space}_reg-{reg_set}"
        paths.extend(
            [
                out_func_dir / f"{base}_desc-Mean_timeseries.1D",
                out_func_dir / f"{base}_desc-PearsonNilearn_correlations.tsv",
                out_func_dir / f"{base}_desc-PartialNilearn_correlations.tsv",
            ]
        )
    return paths


def _export_metrics(
    metrics: MetricsOutputs,
    out_func_dir: Path,
    run: Run,
    reg_set: str,
    template_brain_mask: Path,
    atlases: Mapping[str, Path],
) -> None:
    """Copy metrics outputs into release-named files under ``out_func_dir``."""
    stem_with_space = _run_stem(run, with_space=True)
    stem_no_space = _run_stem(run)

    scalar_sources: dict[str, dict[str, Path]] = {
        "alff": {
            "raw": metrics.alff,
            "sm6": metrics.alff_smooth,
            "smZstd": metrics.alff_zscored,
        },
        "falff": {
            "raw": metrics.falff,
            "sm6": metrics.falff_smooth,
            "smZstd": metrics.falff_zscored,
        },
        "reho": {
            "raw": metrics.reho,
            "sm6": metrics.reho_smooth,
            "smZstd": metrics.reho_zscored,
        },
    }
    for metric, srcs in scalar_sources.items():
        sm6_dst = (
            out_func_dir / f"{stem_with_space}_reg-{reg_set}_desc-sm6_{metric}.nii.gz"
        )
        smzstd_dst = (
            out_func_dir
            / f"{stem_with_space}_reg-{reg_set}_desc-smZstd_{metric}.nii.gz"
        )
        zstd_dst = (
            out_func_dir / f"{stem_with_space}_reg-{reg_set}_desc-zstd_{metric}.nii.gz"
        )
        shutil.copyfile(srcs["sm6"], sm6_dst)
        shutil.copyfile(srcs["smZstd"], smzstd_dst)
        # ``zstd`` = z-scored raw (no smoothing); not in MetricsOutputs.
        zstd_src = compute_zscore(srcs["raw"], template_brain_mask)
        shutil.copyfile(zstd_src, zstd_dst)

    for atl in atlases:
        base = f"{stem_no_space}_atlas-{atl}_space-{run.space}_reg-{reg_set}"
        ts_dst = out_func_dir / f"{base}_desc-Mean_timeseries.1D"
        pearson_dst = out_func_dir / f"{base}_desc-PearsonNilearn_correlations.tsv"
        partial_dst = out_func_dir / f"{base}_desc-PartialNilearn_correlations.tsv"
        # Transpose to ``(T, n_rois)`` to match AFNI's ``.1D`` and the release.
        ts_arr = np.loadtxt(metrics.timeseries[atl])
        np.savetxt(ts_dst, ts_arr.T, delimiter="\t")
        shutil.copyfile(metrics.correlation_matrix[atl], pearson_dst)
        _compute_partial_correlation(ts_dst, partial_dst)


def _process_run(
    run: Run,
    input_dir: Path,
    output_dir: Path,
    work_root: Path,
    *,
    bandpass: tuple[float, float],
    tr_override: float | None,
    overwrite: bool,
    skip_metrics: bool,
    atlases: Mapping[str, Path],
    fwhm: float = 6.0,
) -> None:
    rel = run.head_bold.parent.relative_to(input_dir)
    out_func_dir = output_dir / rel
    out_func_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, Path, Path, Path, bool, bool]] = []
    for reg_set, reg_file in run.regressors.items():
        out_bold = out_func_dir / (
            f"{_run_stem(run, with_space=True)}_reg-{reg_set}_desc-preproc_bold.nii.gz"
        )
        out_reg = out_func_dir / (
            f"{_run_stem(run)}_reg-{reg_set}_desc-bandpassed_regressors.1D"
        )
        bold_done = not overwrite and out_bold.exists() and out_reg.exists()
        metrics_done = skip_metrics or (
            not overwrite
            and all(
                p.exists()
                for p in _metric_output_paths(out_func_dir, run, reg_set, atlases)
            )
        )
        if bold_done and metrics_done:
            LOG.info("  skip reg-%s: outputs already exist", reg_set)
            continue
        pending.append((reg_set, reg_file, out_bold, out_reg, bold_done, metrics_done))

    if not pending:
        return

    tr = _detect_tr(run.native_bold, tr_override)
    LOG.info(
        "sub-%s ses-%s task-%s run-%s: TR=%.3fs",
        run.sub,
        run.ses or "-",
        run.task,
        run.run or "-",
        tr,
    )

    run_id = "_".join(filter(None, [run.sub, run.ses, run.task, run.run]))
    for reg_set, reg_file, out_bold, out_reg, bold_done, metrics_done in pending:
        work_dir = work_root / run_id / reg_set
        patched = _patch_head_bold(run.head_bold, run.native_bold, work_dir)
        staged_mask = _stage_mask(run.bold_mask, patched, work_dir)

        if bold_done:
            cleaned_bold = out_bold
        else:
            bp_result = apply_regression_bandpass(
                bold_file=patched,
                brain_mask_file=staged_mask,
                regressor_file=reg_file,
                f_low=bandpass[0],
                f_high=bandpass[1],
            )
            shutil.copyfile(bp_result.regressed_bold, out_bold)
            cleaned_bold = bp_result.regressed_bold

            bpf_reg = bandpass_regressor_file(
                reg_file, tr=tr, f_low=bandpass[0], f_high=bandpass[1]
            )
            shutil.copyfile(bpf_reg, out_reg)
            LOG.info("  wrote reg-%s -> %s", reg_set, out_bold.name)

        if metrics_done:
            continue

        reg_result = apply_regression(
            bold_file=patched,
            brain_mask_file=staged_mask,
            regressor_file=reg_file,
        )
        metrics = single_session_metrics(
            regressed_bold=reg_result.regressed_bold,
            cleaned_bold=cleaned_bold,
            template_brain_mask=staged_mask,
            tr=tr,
            atlas_files=atlases,
            fwhm=fwhm,
        )
        _export_metrics(metrics, out_func_dir, run, reg_set, staged_mask, atlases)
        LOG.info(
            "  wrote reg-%s metrics (%d atlases, %d scalar variants)",
            reg_set,
            len(atlases),
            len(SCALAR_METRICS) * len(METRIC_VARIANTS),
        )


def _verify_release(input_dir: Path) -> int:
    """Check every run's head_bold for the TR bug; report counts per state.

    Reads only the NIfTI header (no data), so this is cheap even on full
    releases. Exit codes:

    * ``0`` -- bug detected on at least one run (safe to run the fix).
    * ``1`` -- no runs discovered.
    * ``2`` -- no buggy runs found (already patched, or a different release).
    """
    runs = list(discover_runs(input_dir))
    if not runs:
        print(f"No runs discovered under {input_dir}", file=sys.stderr)
        return 1

    buggy: list[Path] = []
    clean: list[Path] = []
    inconclusive: list[tuple[Path, float, float]] = []
    for run in runs:
        head_tr = float(nib.nifti1.load(run.head_bold).header.get_zooms()[3])
        native_tr = float(nib.nifti1.load(run.native_bold).header.get_zooms()[3])
        if head_tr != native_tr and (head_tr <= 0.0 or head_tr == 1.0):
            buggy.append(run.head_bold)
        elif head_tr == native_tr:
            clean.append(run.head_bold)
        else:
            inconclusive.append((run.head_bold, native_tr, head_tr))

    sample = runs[0]
    print(f"Sample run : {sample.head_bold.relative_to(input_dir)}")
    print(
        f"  native preproc_bold TR : "
        f"{nib.nifti1.load(sample.native_bold).header.get_zooms()[3]:.4f}s"
    )
    print(
        f"  template head_bold  TR : "
        f"{nib.nifti1.load(sample.head_bold).header.get_zooms()[3]:.4f}s"
    )
    print(
        f"\nScanned {len(runs)} run(s): "
        f"{len(buggy)} buggy, {len(clean)} clean, {len(inconclusive)} inconclusive."
    )
    for path, ntr, htr in inconclusive[:5]:
        print(
            f"  inconclusive: {path.relative_to(input_dir)} "
            f"(head TR={htr}, native TR={ntr})"
        )
    if buggy:
        print(f"\nBug present in {len(buggy)} run(s); safe to run the full fix.")
        return 0
    print(
        "\nNo buggy runs found. "
        "Either the release was already patched or this is a different release."
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root of the downloaded RBC release (containing sub-* folders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write the parallel fixed-derivatives tree. "
        "Required unless --dry-run or --verify is set.",
    )
    parser.add_argument(
        "--participant-label",
        nargs="+",
        default=None,
        help="Restrict to specific subject(s) (with or without 'sub-' prefix).",
    )
    parser.add_argument(
        "--bandpass",
        nargs=2,
        type=float,
        default=(0.01, 0.1),
        metavar=("F_LOW", "F_HIGH"),
        help="Bandpass cutoffs in Hz (default: 0.01 0.1).",
    )
    parser.add_argument(
        "--tr-override",
        type=float,
        default=None,
        help="Use this TR instead of auto-detecting from native preproc_bold.",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=6.0,
        help="Smoothing kernel FWHM in mm for the sm6 metric variants "
        "(default: 6.0; matches the release ``desc-sm6`` label).",
    )
    parser.add_argument(
        "--runner",
        default="auto",
        choices=["auto", "local", "docker", "podman", "singularity"],
        help="NiWrap runner for AFNI 3dTproject (default: auto).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Parent directory under which to create the (auto-cleaned) scratch "
        "folder for patched headers and niwrap exec dirs. Defaults to the "
        "system temp dir (honors $TMPDIR/$TEMP/$TMP). Point this at a roomy "
        "disk for multi-thousand-run releases.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only regenerate the cleaned BOLD + bandpassed regressors; do not "
        "recompute ALFF/fALFF/ReHo/timeseries/correlations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if they already exist (default: skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered runs and exit without writing anything.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Inspect one run and report whether the TR bug is present; exit.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first run that fails instead of continuing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase log verbosity."
    )
    return parser


def _filter_runs(runs: list[Run], participant_label: list[str] | None) -> list[Run]:
    if not participant_label:
        return runs
    wanted = {p.removeprefix("sub-") for p in participant_label}
    return [r for r in runs if r.sub in wanted]


def _process_all(
    runs: list[Run],
    args: argparse.Namespace,
    work_root: Path,
    atlases: Mapping[str, Path],
) -> int:
    failures: list[tuple[Run, Exception]] = []
    for idx, run in enumerate(runs, start=1):
        LOG.info(
            "[%d/%d] sub-%s ses-%s task-%s run-%s",
            idx,
            len(runs),
            run.sub,
            run.ses or "-",
            run.task,
            run.run or "-",
        )
        try:
            _process_run(
                run,
                args.input_dir,
                args.output_dir,
                work_root,
                bandpass=tuple(args.bandpass),
                tr_override=args.tr_override,
                overwrite=args.overwrite,
                skip_metrics=args.skip_metrics,
                atlases=atlases,
                fwhm=args.fwhm,
            )
        except Exception as exc:
            LOG.error("Failed sub-%s task-%s: %s", run.sub, run.task, exc)
            failures.append((run, exc))
            if args.fail_fast:
                raise

    if failures:
        LOG.error("Done with %d failure(s); first: %s", len(failures), failures[0][1])
        return 1
    LOG.info("Done. Fixed %d run(s).", len(runs))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the TR-fix script."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.verify:
        return _verify_release(args.input_dir)

    runs = _filter_runs(list(discover_runs(args.input_dir)), args.participant_label)
    LOG.info("Discovered %d run(s) to process", len(runs))

    if args.dry_run:
        for r in runs:
            stem = _run_stem(r, with_space=True)
            for reg_set in r.regressors:
                print(f"  would write {stem}_reg-{reg_set}_desc-preproc_bold.nii.gz")
        return 0

    if args.output_dir is None:
        print("--output-dir is required unless --dry-run/--verify", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atlases: Mapping[str, Path] = {} if args.skip_metrics else _resolve_atlases()
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rbc_tr_fix_", dir=args.work_dir
    ) as work_str:
        # Route niwrap exec folders under the same temp root so they get
        # cleaned up; else ``generate_exec_folder`` accumulates GBs across
        # a multi-thousand-run release.
        prev_work_dir = os.environ.get("RBC_WORK_DIR")
        os.environ["RBC_WORK_DIR"] = work_str
        try:
            setup_runner(runner=args.runner, verbose=args.verbose)
            return _process_all(runs, args, Path(work_str), atlases)
        finally:
            if prev_work_dir is None:
                os.environ.pop("RBC_WORK_DIR", None)
            else:
                os.environ["RBC_WORK_DIR"] = prev_work_dir


if __name__ == "__main__":
    sys.exit(main())
