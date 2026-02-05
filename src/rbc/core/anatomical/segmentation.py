"""RBC skull stripping method."""

from pathlib import Path
from types import SimpleNamespace

from niwrap import ants, fsl

from rbc.core.resources import OASIS_TEMPLATES


def ants_brain_extraction(
    in_file: Path, output_prefix: str
) -> ants.AntsBrainExtractionShOutputs:
    """ANTs N4 bias correction and brain extraction.

    Args:
        in_file: Input anatomical file to perform brain extraction on.
        output_prefix: Prefix for output file names

    Returns:
        ANTs brain extraction output object.
    """
    return ants.ants_brain_extraction_sh(
        image_dimension=3,
        anatomical_image=in_file,
        template=OASIS_TEMPLATES.template,
        probability_mask=OASIS_TEMPLATES.probability_mask,
        brain_extraction_registration_mask=OASIS_TEMPLATES.registration_mask,
        output_prefix=output_prefix,
        image_file_suffix="nii.gz",
        random_seeding=False,
    )


def fsl_tissue_segmentation(in_file: Path, output_prefix: str) -> SimpleNamespace:
    """FSL Fast tissue classification.

    Args:
        in_file: Input anatomical file to perform tissue classification on.
        output_prefix: Prefix for output file names

    Returns:
        Namespace with paths to each tissue mask.
    """
    tissues = fsl.fast(
        in_files=[in_file],
        img_type=1,
        number_classes=3,
        segments=True,
        out_basename=output_prefix,
    )
    masks = {
        tissue_type: fsl.fslmaths(
            input_files=[tissues.root / f"{output_prefix}_pve_{idx}.nii.gz"],
            operations=[
                fsl.fslmaths_operation(thr=0.95),
                fsl.fslmaths_operation(bin_=True),
            ],
            output=f"{tissue_type}_mask.nii.gz",
        ).output_file
        for idx, tissue_type in enumerate(["csf", "gm", "wm"])
    }
    return SimpleNamespace(**masks)
