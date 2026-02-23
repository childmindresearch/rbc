"""Functional processing.

This module defines functional MRI processing methods.

Functional MRI measures brain activity over time via the BOLD signal. Raw
fMRI contains motion artifacts, timing differences, and distortions that
must be corrected before analysis.
"""

from __future__ import annotations

from .despiking import despike_bold
from .initialization import scale_bold, truncate_trs
from .motion import extract_motion_reference, fsl_motion_correction
from .resampling import resample_bold_to_template
from .timing import slice_timing_correction

__all__ = [
    "despike_bold",
    "extract_motion_reference",
    "fsl_motion_correction",
    "resample_bold_to_template",
    "scale_bold",
    "slice_timing_correction",
    "truncate_trs",
]
