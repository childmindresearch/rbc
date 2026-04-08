"""Bundled resources for the RBC pipeline.

Provides resolved paths to:

- **OASIS templates** -- used by ANTs brain extraction to map a brain
  probability mask into subject space.
- **MNI152 templates** -- the standard-space target for anatomical registration
  and the reference grid for template-space outputs.
- **FSL resources** -- schedule files for FLIRT registration.
- **Atlases** -- brain parcellation NIfTI files for timeseries extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Literal

    AtlasName = Literal[
        "aal",
        "brodmann",
        "craddock_200",
        "craddock_400",
        "glasser",
        "harvard_oxford_cortical",
        "harvard_oxford_subcortical",
        "juelich",
        "schaefer_200",
        "schaefer_300",
        "schaefer_400",
        "schaefer_1000",
        "slab_907",
        "yeo_7",
        "yeo_7_liberal",
        "yeo_17",
        "yeo_17_liberal",
    ]

_ROOT = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class BrainExtractionTemplates(NamedTuple):
    """Paths to brain-extraction templates.

    Defaults to OASIS templates. Can be overridden via CLI flags.

    Attributes:
        template: T1w group template for brain extraction.
        probability_mask: Brain+cerebellum probability mask for extraction.
        registration_mask: Brain+cerebellum registration mask.
    """

    template: Path
    probability_mask: Path
    registration_mask: Path


class RegistrationTemplates(NamedTuple):
    """Paths to standard-space registration templates.

    Defaults to MNI152 templates. Can be overridden via CLI flags.

    Attributes:
        brain_1mm: T1w brain at 1 mm resolution (registration target).
        brain_2mm: T1w brain at 2 mm resolution.
        brain_mask_2mm: T1w brain mask at 2 mm resolution.
        bold_ref: BOLD reference image.
    """

    brain_1mm: Path
    brain_2mm: Path
    brain_mask_2mm: Path
    bold_ref: Path


# Backward-compatible aliases
OasisTemplates = BrainExtractionTemplates
MniTemplates = RegistrationTemplates


class FSL(NamedTuple):
    """Paths to FSL resources."""

    bbr_schedule: Path


_TEMPLATES = _ROOT / "templates"

BRAIN_EXTRACTION_TEMPLATES = BrainExtractionTemplates(
    template=_TEMPLATES / "oasis_template.nii.gz",
    probability_mask=_TEMPLATES / "oasis_probability_mask.nii.gz",
    registration_mask=_TEMPLATES / "oasis_registration_mask.nii.gz",
)

REGISTRATION_TEMPLATES = RegistrationTemplates(
    brain_1mm=_TEMPLATES / "mni152_T1w_1mm_brain.nii.gz",
    brain_2mm=_TEMPLATES / "mni152_T1w_2mm_brain.nii.gz",
    brain_mask_2mm=_TEMPLATES / "mni152_T1w_2mm_brain_mask.nii.gz",
    bold_ref=_TEMPLATES / "mni152_bold_ref_2mm.nii.gz",
)

# Backward-compatible aliases
OASIS_TEMPLATES = BRAIN_EXTRACTION_TEMPLATES
MNI_TEMPLATES = REGISTRATION_TEMPLATES

FSL_RESOURCES = FSL(bbr_schedule=_ROOT / "configs" / "flirt_bbr_schedule.sch")


# ---------------------------------------------------------------------------
# Atlases
# ---------------------------------------------------------------------------

_ATLAS_DIR = _ROOT / "atlases"

ATLAS_REGISTRY: dict[AtlasName, str] = {
    "aal": "atlas-AAL_space-MNI152NLin6_res-2_dseg.nii.gz",
    "brodmann": "atlas-Brodmann_space-MNI152NLin6_res-2_dseg.nii.gz",
    "craddock_200": "atlas-CC200_space-MNI152NLin6_res-2_dseg.nii.gz",
    "craddock_400": "atlas-CC400_space-MNI152NLin6_res-2_dseg.nii.gz",
    "glasser": "atlas-Glasser_space-MNI152NLin6_res-2_dseg.nii.gz",
    "harvard_oxford_cortical": (
        "atlas-HarvardOxfordcortMaxprobThr25_space-MNI152NLin6_res-2_dseg.nii.gz"
    ),
    "harvard_oxford_subcortical": (
        "atlas-HarvardOxfordsubMaxprobThr25_space-MNI152NLin6_res-2_dseg.nii.gz"
    ),
    "juelich": "atlas-Juelich_space-MNI152NLin6_res-2_dseg.nii.gz",
    "schaefer_200": (
        "atlas-Schaefer2018_space-MNI152NLin6_res-2_"
        "desc-200Parcels17NetworksOrder_dseg.nii.gz"
    ),
    "schaefer_300": (
        "atlas-Schaefer2018_space-MNI152NLin6_res-2_"
        "desc-300Parcels17NetworksOrder_dseg.nii.gz"
    ),
    "schaefer_400": (
        "atlas-Schaefer2018_space-MNI152NLin6_res-2_"
        "desc-400Parcels17NetworksOrder_dseg.nii.gz"
    ),
    "schaefer_1000": (
        "atlas-Schaefer2018_space-MNI152NLin6_res-2_"
        "desc-1000Parcels17NetworksOrder_dseg.nii.gz"
    ),
    "slab_907": "atlas-Slab907_space-MNI152NLin6_res-2_dseg.nii.gz",
    "yeo_7": "atlas-Yeo7_space-MNI152NLin6_res-2_dseg.nii.gz",
    "yeo_7_liberal": "atlas-Yeo7liberal_space-MNI152NLin6_res-2_dseg.nii.gz",
    "yeo_17": "atlas-Yeo17_space-MNI152NLin6_res-2_dseg.nii.gz",
    "yeo_17_liberal": "atlas-Yeo17liberal_space-MNI152NLin6_res-2_dseg.nii.gz",
}


def get_atlas(name: AtlasName) -> Path:
    """Return the full path for a named atlas.

    Args:
        name: One of the supported atlas short names (see ``ATLAS_REGISTRY``).

    Returns:
        Absolute path to the atlas NIfTI file.

    Raises:
        ValueError: If *name* is not in the registry.
        FileNotFoundError: If the atlas file does not exist on disk.
    """
    filename = ATLAS_REGISTRY.get(name)
    if filename is None:
        raise ValueError(
            f"Unknown atlas {name!r}. Supported: {', '.join(sorted(ATLAS_REGISTRY))}"
        )
    path = _ATLAS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Atlas file not found: {path}.")
    return path


def resolve_atlas(name_or_path: str) -> tuple[str, Path]:
    """Resolve an atlas by registry name or filesystem path.

    If *name_or_path* matches a key in :data:`ATLAS_REGISTRY`, the
    corresponding bundled atlas is returned. Otherwise it is treated as a
    filesystem path to a user-supplied NIfTI file.

    Args:
        name_or_path: Either a registry short name (e.g. ``"schaefer_200"``)
            or an absolute/relative path to a NIfTI atlas file.

    Returns:
        A ``(label, path)`` tuple. For registry atlases the label is the
        short name; for custom paths the label is derived from the filename
        stem (without ``.nii.gz``).

    Raises:
        FileNotFoundError: If a custom atlas path does not exist on disk.
    """
    if name_or_path in ATLAS_REGISTRY:
        return (name_or_path, get_atlas(name_or_path))  # type: ignore[arg-type]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom atlas not found: {path}")
    label = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    return (label, path.resolve())


__all__ = [
    "ATLAS_REGISTRY",
    "BRAIN_EXTRACTION_TEMPLATES",
    "FSL_RESOURCES",
    "MNI_TEMPLATES",
    "OASIS_TEMPLATES",
    "REGISTRATION_TEMPLATES",
    "get_atlas",
    "resolve_atlas",
]
