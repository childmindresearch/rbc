"""Functional workflows."""

from functools import partial
from pathlib import Path

import niwrap_helper

from rbc.core.common import reorient
from rbc.core.functional import truncate_trs
from rbc.core.utils import get_base_entities, rename


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

    # Prep files to save
    outputs = [
        (reoriented_bold.root / "reorient_bold.nii.gz", "reorient", "bold"),
        (truncated_bold.root / "bold.nii.gz", None, "bold"),
    ]   


    renamed_files = [
        rename(out_file, bids(desc=desc, suffix=suffix, ext=".nii.gz"))
        if desc else rename(out_file, bids(suffix=suffix, ext=".nii.gz"))
        for out_file, desc, suffix in outputs
    ]

    niwrap_helper.save(
        renamed_files,
        out_dir=output_dir / bids(datatype="func", directory=True),
    )

    