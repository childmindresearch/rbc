"""Anatomical preprocessing workflow.

Chains the full anatomical stream -- reorientation, brain extraction,
tissue segmentation, and template registration -- and writes BIDS-named
outputs to disk.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from rbc.core.anatomical import (
    ants_brain_extraction,
    ants_registration,
    fsl_tissue_segmentation,
)
from rbc.core.bids import bids_path, parse_bids_name
from rbc.core.common import deoblique_and_reorient
from rbc.core.fileops import file_copy_many, file_rename

if TYPE_CHECKING:
    from pathlib import Path


def single_session(in_t1w: Path, output_dir: Path) -> None:
    """Run the full anatomical preprocessing pipeline for one session.

    Pipeline steps (see ``rbc_reimplementation_guide.md``):

    1. Deoblique and reorient T1w to RPI.
    2. ANTs brain extraction (N4 bias correction + skull-stripping).
    3. FSL FAST tissue segmentation (CSF / GM / WM masks).
    4. ANTs registration to MNI152 template (forward + inverse transforms).

    All outputs are renamed to BIDS convention and copied into
    ``<output_dir>/sub-<label>/[ses-<label>/]anat/``.

    Args:
        in_t1w: Raw T1w image (BIDS-named) to preprocess.
        output_dir: Root output directory (e.g. ``derivatives/rbc``).

    Raises:
        FileNotFoundError: If brain extracted file could not be found.
    """
    entities = parse_bids_name(in_t1w.name).entities
    sub = entities.get("sub")
    ses = entities.get("ses")
    run = int(entities["run"]) if "run" in entities else None
    name = partial(
        bids_path, sub=sub, ses=ses, run=run, datatype="anat", extension=".nii.gz"
    )

    reoriented_t1w = deoblique_and_reorient(in_file=in_t1w)
    extracted_t1w = ants_brain_extraction(in_file=reoriented_t1w.out_file)
    tissue_masks = fsl_tissue_segmentation(in_file=extracted_t1w.brain_extracted_image)
    transforms = ants_registration(in_file=extracted_t1w.brain_extracted_image)

    # Rename outputs to BIDS-compliant names
    brain = file_rename(
        extracted_t1w.brain_extracted_image,
        name(desc="brain", suffix="T1w").name,
    )
    brain_mask = file_rename(
        extracted_t1w.brain_mask, name(desc="T1w", suffix="mask").name
    )
    csf_mask = file_rename(tissue_masks.csf, name(desc="csf", suffix="mask").name)
    gm_mask = file_rename(tissue_masks.gm, name(desc="gm", suffix="mask").name)
    wm_mask = file_rename(tissue_masks.wm, name(desc="wm", suffix="mask").name)
    fwd_xfm = file_rename(
        transforms.forward,
        name(
            extra={"from": "T1w", "to": "template", "mode": "image"}, suffix="xfm"
        ).name,
    )
    inv_xfm = file_rename(
        transforms.inverse,
        name(
            extra={"from": "template", "to": "T1w", "mode": "image"}, suffix="xfm"
        ).name,
    )

    file_copy_many(
        [brain, brain_mask, csf_mask, gm_mask, wm_mask, fwd_xfm, inv_xfm],
        out_dir=output_dir / name(desc="brain", suffix="T1w").parent,
    )
