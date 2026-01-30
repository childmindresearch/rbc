"""RBC skull stripping method."""

from pathlib import Path

from niwrap import ants

from rbc.core.resources import OASIS_TEMPLATES


def ants_brain_extraction(
    in_file: Path, output_prefix: str
) -> ants.AntsBrainExtractionShOutputs:
    """ANTs N4 bias correction, brain extraction, and tissue classification.

    NOTE: Performing tissue classification as well, but original workflow
    used FSL to separately perform tissue classification.

    Args:
        in_file: Input anatomical file to perform brain extraction on.
        output_prefix: Prefix for output file names
    """
    return ants.ants_brain_extraction_sh(
        image_dimension=3,
        anatomical_image=in_file,
        template=OASIS_TEMPLATES.template,
        probability_mask=OASIS_TEMPLATES.probability_mask,
        brain_extraction_registration_mask=OASIS_TEMPLATES.registration_mask,
        output_prefix=output_prefix,
        image_file_suffix="nii.gz",
        tissue_classification="3x1x2x3",
        random_seeding=False,
    )
