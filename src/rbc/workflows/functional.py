"""Functional workflows."""

import shutil
from functools import partial
from pathlib import Path
from typing import Any

from rbc.core.bids import Datatype, Suffix, bids_name, bids_path, parse_bids_name
from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    generate_motion_reference,
    motion_correction,
    truncate_trs,
)


def single_session(in_bold: Path, output_dir: Path, start_tr: int = 2) -> None:
    """Workflow for preprocessing functional data.

    Args:
        in_bold: Input BOLD timeseries to process.
        output_dir: Parent output directory to save data to.
        start_tr: Number of initial TRs to remove (default: 2).
    """
    parsed = parse_bids_name(in_bold.name)
    entities: dict[str, Any] = parsed.entities

    bn = partial(bids_name, **entities)
    bp = partial(bids_path, **entities, datatype=Datatype.FUNC)

    reoriented = deoblique_and_reorient(
        in_file=in_bold,
        output_fname=bn(desc="reoriented", suffix=Suffix.BOLD, extension=".nii.gz"),
    )

    truncated = truncate_trs(
        in_file=reoriented.out_file,
        output_fname=bn(suffix=Suffix.BOLD, extension=".nii.gz"),
        start_tr=start_tr,
    )

    motion_ref = generate_motion_reference(
        in_file=truncated.output_file,
        output_fname=bn(suffix=Suffix.SBREF, extension=".nii.gz"),
    )

    motion_corrected = motion_correction(
        in_file=truncated.output_file,
        ref_file=motion_ref.output_file,
        output_prefix=bn(suffix=Suffix.MOTION, extension=""),
    )

    func_out_dir = output_dir / bp(suffix=Suffix.BOLD, extension=".nii.gz").parent
    func_out_dir.mkdir(parents=True, exist_ok=True)

    # Save transform .mat directory
    mat_target = func_out_dir / bn(desc="motion", suffix="mat", extension="")
    shutil.copytree(motion_corrected.mat_dir, mat_target, dirs_exist_ok=True)

    # Output files
    outputs = [
        (reoriented.out_file, "reorient", Suffix.BOLD, ".nii.gz"),
        (truncated.output_file, "truncated", Suffix.BOLD, ".nii.gz"),
        (motion_ref.output_file, None, Suffix.SBREF, ".nii.gz"),
        (
            motion_corrected.bold.with_suffix(".nii.gz"),
            "motion",
            Suffix.BOLD,
            ".nii.gz",
        ),
        (motion_corrected.par, "motionParams", Suffix.MOTION, ".txt"),
        (motion_corrected.rms_rel, "relsDisplacement", Suffix.MOTION, ".rms"),
        (motion_corrected.rms_abs, "maxDisplacement", Suffix.MOTION, ".rms"),
    ]

    for out_file, desc, suffix, ext in outputs:
        dest_path = func_out_dir / bn(desc=desc, suffix=suffix, extension=ext)
        shutil.move(str(out_file), str(dest_path))
