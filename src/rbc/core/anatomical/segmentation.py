"""Brain extraction and tissue segmentation.

Brain extraction (skull-stripping) isolates brain tissue from the T1w image
using ANTs ``antsBrainExtraction.sh``, which also performs N4 bias-field
correction. The bias-corrected, skull-stripped brain is then segmented into
CSF, gray matter, and white matter using FSL FAST. The resulting tissue masks
are used downstream for nuisance regression and coregistration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

from niwrap import ants, fsl

from rbc.core.resources import OASIS_TEMPLATES


class TissueMasks(NamedTuple):
    """Paths to tissue segmentation masks."""

    csf: Path
    gm: Path
    wm: Path


def ants_brain_extraction(
    in_file: Path,
) -> ants.AntsBrainExtractionShOutputs:
    """Skull-strip a T1w image using ANTs ``antsBrainExtraction.sh``.

    Internally performs N4 bias-field correction, registers the input to
    the OASIS template, maps a brain probability mask back to subject
    space, and thresholds it to produce a binary brain mask. The key
    outputs are the bias-corrected brain image and the brain mask.

    Args:
        in_file: Input anatomical T1w image to perform brain extraction on. In RBC this is the Reoriented (RPI) T1w.

    Returns:
        ANTs brain extraction outputs (brain image, brain mask, N4-corrected
        full-head image, etc.).
    """
    return ants.ants_brain_extraction_sh(
        image_dimension=3,
        anatomical_image=in_file,
        template=OASIS_TEMPLATES.template,
        probability_mask=OASIS_TEMPLATES.probability_mask,
        brain_extraction_registration_mask=OASIS_TEMPLATES.registration_mask,
        output_prefix="ants_be",
        image_file_suffix="nii.gz",
        random_seeding=False,
    )


def fsl_tissue_segmentation(in_file: Path) -> TissueMasks:
    """Segment a brain into CSF, gray matter, and white matter with FSL FAST.

    Runs three-class tissue classification on a skull-stripped brain image,
    then thresholds each partial-volume estimate at 0.95 to produce binary
    tissue masks. These masks are used later for nuisance regression (mean
    CSF/WM signals) and boundary-based coregistration (WM boundary).

    Args:
        in_file: Skull-stripped brain image (output of brain extraction).

    Returns:
        Paths to binary CSF, GM, and WM masks (thresholded at 0.95).
    """
    prefix = "tissue_seg"
    tissues = fsl.fast(
        in_files=[in_file],
        img_type=1,
        number_classes=3,
        segments=True,
        out_basename=prefix,
    )
    masks = {
        tissue_type: fsl.fslmaths(
            input_files=[tissues.root / f"{prefix}_pve_{idx}.nii.gz"],
            operations=[
                fsl.fslmaths_operation_thr(thr=0.95),
                fsl.fslmaths_operation_bin(bin_=True),
            ],
            output=f"{tissue_type}_mask.nii.gz",
        ).output_file
        for idx, tissue_type in enumerate(["csf", "gm", "wm"])
    }
    return TissueMasks(**masks)
