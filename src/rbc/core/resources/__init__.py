"""Bundled template images used by the pipeline.

Provides resolved paths to:

- **OASIS templates** -- used by ANTs brain extraction (step 2) to map a brain
  probability mask into subject space.
- **MNI152 templates** -- the standard-space target for anatomical registration
  (step 5) and the reference grid for template-space outputs.
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
    """

    brain_1mm: Path


# OASIS
# (sourced from C-PAC container ghcr.io/fcp-indi/c-pac:many_pipes)
OASIS_DIR = RESOURCES_DIR / "oasis"
OASIS_TEMPLATES = OasisTemplates(
    template=OASIS_DIR / "T_template0.nii.gz",
    probability_mask=OASIS_DIR / "T_template0_BrainCerebellumProbabilityMask.nii.gz",
    registration_mask=OASIS_DIR / "T_template0_BrainCerebellumRegistrationMask.nii.gz",
)
# MNI152
# (sourced from FSL6.0 in C-PAC container: ghcr.io/fcp-indi/c-pac:many_pipes
MNI_DIR = RESOURCES_DIR / "mni"
MNI_TEMPLATES = MniTemplates(brain_1mm=MNI_DIR / "MNI152_T1_1mm_brain.nii.gz")

__all__ = ["MNI_TEMPLATES", "OASIS_TEMPLATES"]
