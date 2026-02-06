"""Functional workflows."""

from functools import partial
from pathlib import Path

import niwrap_helper

from rbc.core.common import reorient
from rbc.core.functional import truncate_trs, generate_motion_reference, motion_correction
from rbc.core.utils import get_base_entities, rename, save_directory



def single_session(in_bold: Path, output_dir: Path, start_tr: int = 2) -> None:
    """Workflow for preprocessing functional data.
    
    Args:
        in_bold: Input BOLD timeseries to process.
        output_dir: Parent output directory to save data to.
        start_tr: Number of initial TRs to remove (default: 2).
    """

    bids_entities = get_base_entities(in_bold)
    bids = partial(niwrap_helper.bids_path, **bids_entities)
    
    reoriented_bold = reorient(
        in_file=in_bold,
        output_fname=str(bids(desc="reoriented", suffix="bold", ext=".nii.gz")) 
    )

    truncated_bold = truncate_trs(
        in_file=reoriented_bold.out_file,
        output_fname=str(bids(suffix="bold", ext=".nii.gz")),
        start_tr=start_tr
    )  
    
    motion_reference = generate_motion_reference( 
        in_file=truncated_bold.output_file,
        output_fname=str(bids(suffix="sbref", ext=".nii.gz"))
    )

    motion_corrected = motion_correction(
        in_file=truncated_bold.output_file,
        ref_file=motion_reference.output_file,
        output_prefix=str(bids())
    )

    # Prep files to save
    outputs = [
        (reoriented_bold.out_file, "reorient", "bold", ".nii.gz"),
        (truncated_bold.output_file, "truncated", "bold", ".nii.gz"),
        (motion_reference.output_file, None, "sbref", ".nii.gz"),
        (motion_corrected.bold.with_suffix(".nii.gz"), "motion", "bold", ".nii.gz"),
        (motion_corrected.par, "movementParameters", "motion", ".1D"),
        (motion_corrected.rms_rel, "relsDisplacement", "motion", ".rms"),
        (motion_corrected.rms_abs, "absDisplacement", "motion", ".rms"),
    ]   

    func_out_dir = output_dir / bids(datatype="func", directory=True)

    save_directory(
        motion_corrected.mat_dir, 
        func_out_dir, 
        bids(desc="motion", suffix="mat")
    )

    renamed_files = [
        rename(out_file, bids(desc=desc, suffix=suffix, ext=ext) if desc else bids(suffix=suffix, ext=ext))
        for out_file, desc, suffix, ext in outputs
    ]

    niwrap_helper.save(renamed_files, out_dir=func_out_dir)
    
