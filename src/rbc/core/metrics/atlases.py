"""Atlas registry mapping short names to on-disk NIfTI paths.

Atlases are sourced from the C-PAC Docker container
(``fcpindi/c-pac:release-v1.8.5.dev1``) and committed in
``resources/atlases/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    AtlasName = Literal[
        "schaefer_200",
        "schaefer_300",
        "schaefer_400",
        "schaefer_1000",
        "aal",
        "harvard_oxford_cortical",
        "harvard_oxford_subcortical",
        "glasser",
        "yeo_7",
        "yeo_17",
        "cc200",
        "cc400",
    ]

ATLAS_DIR: Path = Path(__file__).resolve().parents[4] / "resources" / "atlases"

_SCHAEFER = "Schaefer2018_space-FSLMNI152_res-2mm_desc"

ATLAS_REGISTRY: dict[str, str] = {
    "schaefer_200": f"{_SCHAEFER}-200Parcels17NetworksOrder.nii.gz",
    "schaefer_300": f"{_SCHAEFER}-300Parcels17NetworksOrder.nii.gz",
    "schaefer_400": f"{_SCHAEFER}-400Parcels17NetworksOrder.nii.gz",
    "schaefer_1000": f"{_SCHAEFER}-1000Parcels17NetworksOrder.nii.gz",
    "aal": "aal_mask_pad.nii.gz",
    "harvard_oxford_cortical": (
        "HarvardOxfordcort-maxprob-thr25"
        "_space-MNI152NLin6_res-1x1x1.nii.gz"
    ),
    "harvard_oxford_subcortical": (
        "HarvardOxfordsub-maxprob-thr25"
        "_space-MNI152NLin6_res-1x1x1.nii.gz"
    ),
    "glasser": "Glasser_space-MNI152NLin6_res-1x1x1.nii.gz",
    "yeo_7": "Yeo-7_space-MNI152NLin6_res-1x1x1.nii.gz",
    "yeo_17": "Yeo-17_space-MNI152NLin6_res-1x1x1.nii.gz",
    "cc200": "CC200.nii.gz",
    "cc400": "CC400.nii.gz",
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
            f"Unknown atlas {name!r}. "
            f"Supported: {', '.join(sorted(ATLAS_REGISTRY))}"
        )
    path = ATLAS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Atlas file not found: {path}. "
            "Expected in resources/atlases/."
        )
    return path
