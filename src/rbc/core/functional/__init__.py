"""Functional processing.

This module defines functional MRI processing methods.

Functional MRI measures brain activity over time via the BOLD signal. Raw
fMRI contains motion artifacts, timing differences, and distortions that
must be corrected before analysis.
"""

from __future__ import annotations

from .initialization import scale_bold, truncate_trs
from .motion import extract_motion_reference, fsl_motion_correction

__all__ = [
    "extract_motion_reference",
    "fsl_motion_correction",
    "scale_bold",
    "truncate_trs",
]
