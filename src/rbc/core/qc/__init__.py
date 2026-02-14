"""Quality control.

This module defines methods for computing quality control metrics, including metrics
like framewise displacement (FD), DVARS, motion-DVARS correlation, and tSNR.
"""

from __future__ import annotations

from rbc.core.qc.motion import (
    MotionQCMetrics,
    count_censored_volumes,
    framewise_displacement_jenkinson,
    framewise_displacement_power,
    motion_qc_metrics,
    rms_motion,
)

__all__ = [
    "MotionQCMetrics",
    "count_censored_volumes",
    "framewise_displacement_jenkinson",
    "framewise_displacement_power",
    "motion_qc_metrics",
    "rms_motion",
]
