"""Functional processing.

This module defines functional MRI processing methods.

Functional MRI measures brain activity over time via the BOLD signal. Raw
fMRI contains motion artifacts, timing differences, and distortions that
must be corrected before analysis.
"""

from .initialization import scale, truncate_trs
from .motion import generate_motion_reference, motion_correction

__all__ = ["generate_motion_reference", "motion_correction", "scale", "truncate_trs"]
