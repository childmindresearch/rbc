"""RBC skull stripping method."""

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
    """ANTs N4 bias correction and brain extraction.

    Args:
        in_file: Input anatomical file to perform brain extraction on.

    Returns:
        ANTs brain extraction output object.
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
    """FSL Fast tissue classification.

    Args:
        in_file: Input anatomical file to perform tissue classification on.

    Returns:
        Namespace with paths to each tissue mask.
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
