"""Anatomical workflows."""

from functools import partial
from pathlib import Path

import niwrap_helper

from rbc.core.anatomical import ants_brain_extraction, ants_registration
from rbc.core.common import reorient
from rbc.core.utils import get_base_entities, rename


def single_session(in_t1w: Path, output_dir: Path) -> None:
    """Workflow for preprocessing anatomical data.

    Args:
        in_t1w: Input T1w image to process.
        output_dir: Parent output directory to save data to.

    Raises:
        FileNotFoundError: If brain extracted file could not be found.
    """
    bids_entities = get_base_entities(in_t1w)
    bids = partial(niwrap_helper.bids_path, **bids_entities)

    reoriented_t1w = reorient(
        in_file=in_t1w,
        output_fname=str(bids(desc="reoriented", suffix="T1w", ext=".nii.gz")),
    )
    extracted_t1w = ants_brain_extraction(
        in_file=reoriented_t1w.out_file, output_prefix=str(bids())
    )
    transforms = ants_registration(
        extracted_t1w.brain_extracted_image, output_prefix=str(bids())
    )

    # Prep files to save
    extracted_t1w_outputs = [
        (extracted_t1w.brain_extracted_image, "brain", "T1w"),
        (extracted_t1w.brain_mask, "T1w", "mask"),
        (extracted_t1w.csf_segmentation, "csf", "dseg"),
        (extracted_t1w.wm_segmentation, "wm", "dseg"),
        (extracted_t1w.gm_segmentation, "gm", "dseg"),
    ]
    renamed_files = [
        rename(out_file, bids(desc=desc, suffix=suffix, ext=".nii.gz"))
        for out_file, desc, suffix in extracted_t1w_outputs
    ]
    niwrap_helper.save(
        [*renamed_files, transforms.forward, transforms.inverse],
        out_dir=output_dir / bids(datatype="anat", directory=True),
    )
