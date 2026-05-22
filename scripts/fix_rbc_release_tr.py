r"""Fix the TR bandpass bug in a downloaded RBC data release.

For each functional run in ``--input-dir``, writes a corrected cleaned BOLD
into a parallel BIDS-derivatives tree under ``--output-dir``:

1. Read the correct TR from the native-space ``desc-preproc_bold.nii.gz``
   header (or accept ``--tr-override``).
2. Patch the template-space ``desc-head_bold.nii.gz`` header, which ships
   with pixdim[4]=0.0 (zeroed by ANTs single-step resampling and later
   silently coerced to 1.0 by AFNI inside C-PAC, which is what drove the
   bandpass off by a factor of two).
3. Re-run nuisance regression + bandpass via AFNI ``3dTproject -bandpass``,
   using C-PAC's raw (unfiltered) regressors as the ort matrix.
4. Write the fixed cleaned BOLD, the matching bandpass-filtered regressors,
   and a JSON sidecar recording provenance.

Only the bandpass bug (#4 in ``docs/cpac_comparison.md``) is addressed.
Downstream derivatives (ALFF, fALFF, ReHo, atlas timeseries, connectivity
matrices) are NOT regenerated; users wanting those should recompute from
the fixed BOLD with their own tooling.

Usage::

    uv run scripts/fix_rbc_release_tr.py \\
        --input-dir  /path/to/rbc_release \\
        --output-dir /path/to/fixed_release \\
        [--participant-label sub-X ...] \\
        [--bandpass 0.01 0.1] \\
        [--tr-override 2.0] \\
        [--runner auto] \\
        [--dry-run | --verify]
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib

from rbc.bids import parse_bids_name
from rbc.core.functional.nuisance import (
    apply_regression_bandpass,
    bandpass_regressor_file,
)
from rbc.core.functional.resampling import restore_tr
from rbc.core.niwrap import setup_runner

if TYPE_CHECKING:
    from collections.abc import Iterator

LOG = logging.getLogger("rbc.fix_release_tr")

REG_SETS = ("36Parameter", "aCompCor")


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


def _run_stem(run: Run, *, with_space: bool = False) -> str:
    parts = [f"sub-{run.sub}"]
    if run.ses:
        parts.append(f"ses-{run.ses}")
    parts.append(f"task-{run.task}")
    if run.run:
        parts.append(f"run-{run.run}")
    if with_space:
        parts.append(f"space-{run.space}")
    return "_".join(parts)


def _stem_from_entities(sub: str, ses: str | None, task: str, run: str | None) -> str:
    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts)


def discover_runs(input_dir: Path) -> Iterator[Run]:
    """Walk *input_dir* and yield one :class:`Run` per discoverable functional run.

    Discovery is anchored on ``*_space-*_desc-head_bold.nii.gz`` (the
    pre-regression template-space BOLD); each match is paired with its
    sibling native ``desc-preproc_bold``, template ``desc-bold_mask``, and
    raw ``reg-*_regressors.1D`` files. Runs missing any required input are
    skipped with a warning.
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
        stem = _stem_from_entities(sub, ses, task, run_ent)

        native_bold = func_dir / f"{stem}_desc-preproc_bold.nii.gz"
        bold_mask = func_dir / f"{stem}_space-{space}_desc-bold_mask.nii.gz"
        regressors = {
            reg: func_dir / f"{stem}_reg-{reg}_regressors.1D" for reg in REG_SETS
        }
        regressors = {k: v for k, v in regressors.items() if v.exists()}

        missing: list[str] = []
        if not native_bold.exists():
            missing.append(native_bold.name)
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


def _patch_head_bold(head_bold: Path, native_bold: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / head_bold.name
    shutil.copy2(head_bold, patched)
    restore_tr(patched, native_bold)
    return patched


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _write_sidecar(
    target_nii: Path,
    *,
    head_bold: Path,
    detected_tr: float,
    bandpass: tuple[float, float],
    regressor_file: Path,
    regressor_set: str,
) -> None:
    sidecar = target_nii.parent / (target_nii.name.replace(".nii.gz", ".json"))
    payload = {
        "Description": (
            "Cleaned BOLD re-derived from the RBC release's head_bold with "
            "a TR-patched header, then re-bandpassed via AFNI 3dTproject. "
            "Replaces the cleaned BOLD shipped in the release, which was "
            "bandpassed at the wrong TR due to a C-PAC bug "
            "(see docs/cpac_comparison.md #4)."
        ),
        "Sources": [head_bold.name, regressor_file.name],
        "DetectedTR": detected_tr,
        "BandpassFreqs": list(bandpass),
        "RegressorSet": regressor_set,
        "RegressorFileSha256": _sha256(regressor_file),
        "OriginalHeadBoldSha256": _sha256(head_bold),
        "GeneratedBy": [
            {
                "Name": "rbc.scripts.fix_rbc_release_tr",
                "CodeURL": "https://github.com/childmindresearch/rbc",
            }
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2))


def _process_run(
    run: Run,
    input_dir: Path,
    output_dir: Path,
    work_root: Path,
    *,
    bandpass: tuple[float, float],
    tr_override: float | None,
) -> None:
    tr = _detect_tr(run.native_bold, tr_override)
    LOG.info(
        "sub-%s ses-%s task-%s run-%s: TR=%.3fs",
        run.sub,
        run.ses or "-",
        run.task,
        run.run or "-",
        tr,
    )

    rel = run.head_bold.parent.relative_to(input_dir)
    out_func_dir = output_dir / rel
    out_func_dir.mkdir(parents=True, exist_ok=True)

    run_id = "_".join(filter(None, [run.sub, run.ses, run.task, run.run]))
    for reg_set, reg_file in run.regressors.items():
        work_dir = work_root / run_id / reg_set
        patched = _patch_head_bold(run.head_bold, run.native_bold, work_dir)

        result = apply_regression_bandpass(
            bold_file=patched,
            brain_mask_file=run.bold_mask,
            regressor_file=reg_file,
            f_low=bandpass[0],
            f_high=bandpass[1],
        )
        out_bold = out_func_dir / (
            f"{_run_stem(run, with_space=True)}_reg-{reg_set}_desc-preproc_bold.nii.gz"
        )
        shutil.copy2(result.regressed_bold, out_bold)

        bpf_reg = bandpass_regressor_file(
            reg_file, tr=tr, f_low=bandpass[0], f_high=bandpass[1]
        )
        out_reg = out_func_dir / (
            f"{_run_stem(run)}_reg-{reg_set}_desc-bandpassed_regressors.1D"
        )
        shutil.copy2(bpf_reg, out_reg)

        _write_sidecar(
            out_bold,
            head_bold=run.head_bold,
            detected_tr=tr,
            bandpass=bandpass,
            regressor_file=reg_file,
            regressor_set=reg_set,
        )
        LOG.info("  wrote reg-%s -> %s", reg_set, out_bold.name)


def _verify_release(input_dir: Path) -> int:
    """Inspect the first run in the release and report whether the bug is present."""
    runs = list(discover_runs(input_dir))
    if not runs:
        print(f"No runs discovered under {input_dir}", file=sys.stderr)
        return 1
    sample = runs[0]
    head_tr = float(nib.nifti1.load(sample.head_bold).header.get_zooms()[3])
    native_tr = float(nib.nifti1.load(sample.native_bold).header.get_zooms()[3])
    print(f"Inspected: {sample.head_bold.relative_to(input_dir)}")
    print(f"  native preproc_bold TR : {native_tr:.4f}s")
    print(f"  template head_bold  TR : {head_tr:.4f}s")
    bug_present = head_tr != native_tr and (head_tr <= 0.0 or head_tr == 1.0)
    if bug_present:
        print(
            f"\nBug present: head_bold pixdim[4]={head_tr}, expected {native_tr}.\n"
            f"Discovered {len(runs)} runs total; safe to run the full fix."
        )
        return 0
    print(
        f"\nNo bug detected on this run (head_bold TR matches native TR={native_tr}).\n"
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
        "--runner",
        default="auto",
        choices=["auto", "local", "docker", "podman", "singularity"],
        help="NiWrap runner for AFNI 3dTproject (default: auto).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch directory for patched head_bold copies "
        "(default: a fresh temp dir).",
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
) -> int:
    failures: list[tuple[Run, Exception]] = []
    for run in runs:
        try:
            _process_run(
                run,
                args.input_dir,
                args.output_dir,
                work_root,
                bandpass=tuple(args.bandpass),
                tr_override=args.tr_override,
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
        level=logging.WARNING - 10 * min(args.verbose, 2),
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

    setup_runner(runner=args.runner, verbose=args.verbose)
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="rbc_tr_fix_"))
    return _process_all(runs, args, work_root)


if __name__ == "__main__":
    sys.exit(main())
