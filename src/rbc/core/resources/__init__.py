"""Module containing paths to resources."""

from pathlib import Path
from types import SimpleNamespace

RESOURCES_DIR = Path(__file__).parent.resolve()

# OASIS
OASIS_DIR = RESOURCES_DIR / "oasis"
OASIS_TEMPLATES = SimpleNamespace(
    template=OASIS_DIR / "T_template0.nii.gz",
    probability_mask=OASIS_DIR / "T_template0_BrainCerebellumProbabilityMask.nii.gz",
    registration_mask=OASIS_DIR / "T_template0_BrainCerebellumRegistrationMask.nii.gz",
)
# MNI152
MNI_DIR = RESOURCES_DIR / "mni"
MNI_TEMPLATES = SimpleNamespace(brain_1mm=MNI_DIR / "MNI152_T1_1mm_brain.nii.gz")

__all__ = ["MNI_TEMPLATES", "OASIS_TEMPLATES"]
