"""Bundled template images used by the pipeline.

Provides resolved paths to:

- **OASIS templates** -- used by ANTs brain extraction to map a brain
  probability mask into subject space.
- **MNI152 templates** -- the standard-space target for anatomical registration
  and the reference grid for template-space outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

RESOURCES_DIR = Path(__file__).parent.resolve()


class OasisTemplates(NamedTuple):
    """Paths to OASIS brain-extraction templates.

    Attributes:
        template: OASIS T1w group template.
        probability_mask: Brain+cerebellum probability mask for extraction.
        registration_mask: Brain+cerebellum registration mask.
    """

    template: Path
    probability_mask: Path
    registration_mask: Path


class MniTemplates(NamedTuple):
    """Paths to MNI152 standard-space templates.

    Attributes:
        brain_1mm: MNI152 T1w brain at 1 mm resolution (registration target).
        brain_mask_2mm: MNI152 T1w brain mask at 2 mm resolution.
        bold_ref: MNI152 bold reference image.
    """

    brain_1mm: Path
    brain_mask_2mm: Path
    bold_ref: Path


class FSL(NamedTuple):
    """Paths to FSL resources."""

    bbr_schedule: Path


# OASIS
# (sourced from C-PAC container ghcr.io/fcp-indi/c-pac:many_pipes)
OASIS_DIR = RESOURCES_DIR / "oasis"
OASIS_TEMPLATES = OasisTemplates(
    template=OASIS_DIR / "T_template0.nii.gz",
    probability_mask=OASIS_DIR / "T_template0_BrainCerebellumProbabilityMask.nii.gz",
    registration_mask=OASIS_DIR / "T_template0_BrainCerebellumRegistrationMask.nii.gz",
)
# MNI152
# FSL 6.0 via C-PAC (ghcr.io/fcp-indi/c-pac:many_pipes & cpindi/c-pac:release-v1.8.5.dev1)
MNI_DIR = RESOURCES_DIR / "mni"
MNI_TEMPLATES = MniTemplates(
    brain_1mm=MNI_DIR / "MNI152_T1_1mm_brain.nii.gz",
    brain_mask_2mm=MNI_DIR / "MNI152_T1_2mm_brain_mask.nii.gz",
    bold_ref=MNI_DIR / "tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref.nii.gz",
)

# FSL
FSL_DIR = RESOURCES_DIR / "fsl"
FSL_RESOURCES = FSL(bbr_schedule=FSL_DIR / "bbr.sch")

__all__ = ["FSL_RESOURCES", "MNI_TEMPLATES", "OASIS_TEMPLATES"]
