"""Quality control.

This module defines methods for computing quality control metrics, including metrics
like framewise displacement (FD), DVARS, motion-DVARS correlation, and tSNR.
"""

from __future__ import annotations

from rbc.core.qc.dvars import (
    DVARSQCMetrics,
    dvars,
    dvars_qc_metrics,
    motion_dvars_correlation,
)
from rbc.core.qc.motion import (
    MotionQCMetrics,
    count_censored_volumes,
    framewise_displacement_jenkinson,
    framewise_displacement_power,
    motion_qc_metrics,
    rms_motion,
)
from rbc.core.qc.registration import (
    RegistrationQCMetrics,
    coverage,
    cross_correlation,
    dice_coefficient,
    jaccard_index,
    registration_qc_metrics,
)

__all__ = [
    "DVARSQCMetrics",
    "MotionQCMetrics",
    "RegistrationQCMetrics",
    "count_censored_volumes",
    "coverage",
    "cross_correlation",
    "dice_coefficient",
    "dvars",
    "dvars_qc_metrics",
    "framewise_displacement_jenkinson",
    "framewise_displacement_power",
    "jaccard_index",
    "motion_dvars_correlation",
    "motion_qc_metrics",
    "registration_qc_metrics",
    "rms_motion",
]
