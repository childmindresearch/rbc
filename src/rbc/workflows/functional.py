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
        output_fname="reorient_bold.nii.gz" 
    )

    truncated_bold = truncate_trs(
        in_file=reoriented_bold.out_file,
        output_fname="bold.nii.gz",
        start_tr=start_tr
    )  
    
    motion_reference = generate_motion_reference( 
        in_file=truncated_bold.output_file,
        output_fname="sbref.nii.gz"
    )

    motion_corrected_bold = motion_correction(
        in_file=truncated_bold.output_file,
        ref_file=motion_reference.output_file,
        output_prefix="mc_bold"
    )

    # Prep files to save
    outputs = [
        (reoriented_bold.root / "reorient_bold.nii.gz", "reorient", "bold", ".nii.gz"),
        (truncated_bold.root / "bold.nii.gz", None, "bold", ".nii.gz"),
        (motion_reference.root / "sbref.nii.gz", None, "sbref", ".nii.gz"),
        (motion_corrected_bold.out_file.with_suffix(".nii.gz"), "motion", "bold", ".nii.gz"),
        (motion_corrected_bold.par_file.with_suffix(".par"), "movementParameters", "motion", ".1D"),
        (motion_corrected_bold.rmsrel_files, "relsDisplacement", "motion", ".rms"),
        (motion_corrected_bold.rmsabs_files, "maxDisplacement", "motion", ".rms"),
        (motion_corrected_bold.root / "mc_bold.mat", None, "motionMatrices", "")
    ]   

    func_out_dir = output_dir / bids(datatype="func", directory=True)

    renamed_files = []
    for out_file, desc, suffix, ext in outputs:
        bids_name = bids(desc=desc, suffix=suffix, ext=ext) if desc else bids(suffix=suffix, ext=ext)
        
        if suffix == "motionMatrices":
            save_directory(out_file, func_out_dir, bids_name)
        else:
            renamed_files.append(rename(out_file, bids_name))
        
    niwrap_helper.save(renamed_files, out_dir=func_out_dir)
