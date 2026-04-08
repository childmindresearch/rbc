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

from rbc_resources import BRAIN_EXTRACTION_TEMPLATES, BrainExtractionTemplates

_SEG_PREFIX = "tissue_seg"


class BrainExtractionOutputs(NamedTuple):
    """Outputs from ANTs brain extraction.

    Attributes:
        brain: Skull-stripped, bias-corrected brain image.
        brain_mask: Binary brain mask.
    """

    brain: Path
    brain_mask: Path


class TissueMasks(NamedTuple):
    """Paths to tissue segmentation masks."""

    csf: Path
    gm: Path
    wm: Path


def ants_brain_extraction(
    in_file: Path,
    brain_extraction_templates: BrainExtractionTemplates = BRAIN_EXTRACTION_TEMPLATES,
) -> BrainExtractionOutputs:
    """Skull-strip a T1w image using ANTs ``antsBrainExtraction.sh``.

    Internally performs N4 bias-field correction, registers the input to
    the brain extraction template, maps a brain probability mask back to
    subject space, and thresholds it to produce a binary brain mask. The
    key outputs are the bias-corrected brain image and the brain mask.

    Args:
        in_file: Input anatomical T1w image to perform brain extraction on.
            In RBC this is the reoriented (RPI) T1w.
        brain_extraction_templates: Brain extraction template bundle.
            Defaults to the bundled OASIS templates.

    Returns:
        Brain extraction outputs (brain image and brain mask).
    """
    result = ants.ants_brain_extraction_sh(
        image_dimension=3,
        anatomical_image=in_file,
        template=brain_extraction_templates.template,
        probability_mask=brain_extraction_templates.probability_mask,
        brain_extraction_registration_mask=brain_extraction_templates.registration_mask,
        output_prefix="ants_be",
        image_file_suffix="nii.gz",
        random_seeding=False,
    )
    assert result.brain_extracted_image is not None  # noqa: S101
    assert result.brain_mask is not None  # noqa: S101
    return BrainExtractionOutputs(
        brain=result.brain_extracted_image,
        brain_mask=result.brain_mask,
    )


def fsl_segmentation(in_file: Path) -> fsl.FastOutputs:
    """Run FSL FAST tissue segmentation on a skull-stripped brain.

    Args:
        in_file: Skull-stripped brain image.

    Returns:
        FSL FAST outputs containing partial volume estimates, tissue probability
        maps, and hard-label segmentation for GM, WM, and CSF.
    """
    return fsl.fast(
        in_files=[in_file],
        img_type=1,
        number_classes=3,
        segments=True,
        out_basename=_SEG_PREFIX,
    )


def fsl_tissue_masks(fast_result: fsl.FastOutputs) -> TissueMasks:
    """Derive binary CSF, GM, and WM masks from FSL FAST probability maps.

    Thresholds each partial-volume estimate at 0.95 to produce binary
    tissue masks. These masks are used later for nuisance regression (mean
    CSF/WM signals).

    Args:
        fast_result: FSL FAST segmentation outputs.

    Returns:
        Paths to binary CSF, GM, and WM masks (thresholded at 0.95).
    """
    masks = {
        tissue_type: fsl.fslmaths(
            input_files=[fast_result.root / f"{_SEG_PREFIX}_pve_{idx}.nii.gz"],
            operations=[
                fsl.fslmaths_operation_thr(thr=0.95),
                fsl.fslmaths_operation_bin(bin_=True),
            ],
            output=f"{tissue_type}_mask.nii.gz",
        ).output_file
        for idx, tissue_type in enumerate(["csf", "gm", "wm"])
    }
    return TissueMasks(**masks)


def fsl_wm_bbr_mask(fast_result: fsl.FastOutputs) -> Path:
    """Derive a WM mask from the FAST segmentation for BBR coregistration.

    Uses the hard-label tissue segmentation to produce a binary mask covering the
    white matter boundary. This mask will be used later for BBR coregistration of
    functional to anatomical images.

    Args:
        fast_result: FSL FAST segmentation outputs.

    Returns:
        Binary WM mask derived from pveseg.
    """
    return fsl.fslmaths(
        input_files=[fast_result.root / f"{_SEG_PREFIX}_pveseg.nii.gz"],
        operations=[
            fsl.fslmaths_operation_thr(thr=2.5),
            fsl.fslmaths_operation_uthr(uthr=3.5),
            fsl.fslmaths_operation_bin(bin_=True),
        ],
        output="wm_bbr_mask.nii.gz",
    ).output_file
