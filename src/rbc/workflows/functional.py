"""Functional preprocessing workflow.

Chains the functional stream -- reorientation, TR truncation, motion-reference
extraction, and motion correction -- and writes BIDS-named outputs to disk.
"""

from __future__ import annotations

import shutil
from functools import partial
from typing import TYPE_CHECKING

from rbc.core.bids import bids_path, parse_bids_name
from rbc.core.common import deoblique_and_reorient
from rbc.core.fileops import file_copy_many, file_rename
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
    truncate_trs,
)

if TYPE_CHECKING:
    from pathlib import Path


def single_session_preprocess(
    in_bold: Path, output_dir: Path, start_tr: int = 2
) -> None:
    """Run the functional preprocessing pipeline for one session.

    Pipeline steps (see ``rbc_reimplementation_guide.md``):

    1. Deoblique and reorient BOLD to RPI.
    2. Truncate first *start_tr* volumes (steady-state equilibration).
    3. Extract middle-volume motion reference.
    4. Motion correction via FSL mcflirt (6-DOF rigid-body).

    All outputs (reoriented BOLD, truncated BOLD, sbref, motion-corrected
    BOLD, motion parameters, displacement metrics, and per-volume transform
    matrices) are renamed to BIDS convention and saved into
    ``<output_dir>/sub-<label>/[ses-<label>/]func/``.

    Args:
        in_bold: Raw BOLD timeseries (BIDS-named) to preprocess.
        output_dir: Root output directory (e.g. ``derivatives/rbc``).
        start_tr: Number of initial TRs to discard (default: 2).
    """
    entities = parse_bids_name(in_bold.name).entities
    sub = entities.get("sub")
    ses = entities.get("ses")
    task = entities.get("task")
    run = int(entities["run"]) if "run" in entities else None
    name = partial(bids_path, sub=sub, ses=ses, task=task, run=run, datatype="func")

    reoriented = deoblique_and_reorient(in_file=in_bold)
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=start_tr)
    motion_ref = extract_motion_reference(in_file=truncated.output_file)
    motion_corrected = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=motion_ref.output_file,
    )

    # Rename outputs to BIDS-compliant names
    reoriented_bold = file_rename(
        reoriented.out_file,
        name(desc="reorient", suffix="bold", extension=".nii.gz").name,
    )
    truncated_bold = file_rename(
        truncated.output_file,
        name(desc="truncated", suffix="bold", extension=".nii.gz").name,
    )
    sbref = file_rename(
        motion_ref.output_file,
        name(suffix="sbref", extension=".nii.gz").name,
    )
    mc_bold = file_rename(
        motion_corrected.bold.with_suffix(".nii.gz"),
        name(desc="motion", suffix="bold", extension=".nii.gz").name,
    )
    mc_par = file_rename(
        motion_corrected.par,
        name(desc="motionParams", suffix="motion", extension=".txt").name,
    )
    mc_rms_rel = file_rename(
        motion_corrected.rms_rel,
        name(desc="relsDisplacement", suffix="motion", extension=".rms").name,
    )
    mc_rms_abs = file_rename(
        motion_corrected.rms_abs,
        name(desc="maxDisplacement", suffix="motion", extension=".rms").name,
    )

    func_out_dir = output_dir / name(suffix="bold", extension=".nii.gz").parent

    file_copy_many(
        [
            reoriented_bold,
            truncated_bold,
            sbref,
            mc_bold,
            mc_par,
            mc_rms_rel,
            mc_rms_abs,
        ],
        out_dir=func_out_dir,
    )

    # Save transform .mat directory
    mat_target = func_out_dir / name(desc="motion", suffix="mat", extension="").name
    shutil.copytree(motion_corrected.mat_dir, mat_target, dirs_exist_ok=True)
