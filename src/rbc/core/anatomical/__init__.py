"""Anatomical processing.

This module defines anatomical MRI processing methods.

Anatomical processing prepares structural brain images (e.g. T1-weighted) for
analysis. The T1w provides high-resolution anatomy that can be used to align
other modalities (e.g. functional) and identify different tissue types
(gray matter, white matter, CSF).
"""

from .registration import ants_registration
from .skull_stripping import ants_brain_extraction

__all__ = ["ants_brain_extraction", "ants_registration"]
