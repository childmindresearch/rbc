"""Anatomical processing.

This module defines anatomical MRI processing methods.

Anatomical processing prepares structural brain images (e.g. T1-weighted) for
analysis. The T1w provides high-resolution anatomy that can be used to align
other modalities (e.g. functional) and identify different tissue types
(gray matter, white matter, CSF).
"""

from __future__ import annotations

from .registration import ants_registration
from .segmentation import ants_brain_extraction, fsl_segmentation, fsl_tissue_masks

__all__ = [
    "ants_brain_extraction",
    "ants_registration",
    "fsl_segmentation",
    "fsl_tissue_masks",
]
