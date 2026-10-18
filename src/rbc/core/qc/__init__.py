"""Quality control.

This module defines methods for computing quality control metrics, including
framewise displacement (FD), DVARS, motion-DVARS correlation, and registration
overlap metrics, plus HTML report generation for QC outcomes.
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
from rbc.core.qc.report import (
    ReportSection,
    generate_qc_report,
)
from rbc.core.qc.xcp import (
    XCPQCMetrics,
    generate_xcp_qc,
    passes_rbc_qc,
    write_xcp_qc,
)

__all__ = [
    "DVARSQCMetrics",
    "MotionQCMetrics",
    "RegistrationQCMetrics",
    "ReportSection",
    "XCPQCMetrics",
    "count_censored_volumes",
    "coverage",
    "cross_correlation",
    "dice_coefficient",
    "dvars",
    "dvars_qc_metrics",
    "framewise_displacement_jenkinson",
    "framewise_displacement_power",
    "generate_qc_report",
    "generate_xcp_qc",
    "jaccard_index",
    "motion_dvars_correlation",
    "motion_qc_metrics",
    "passes_rbc_qc",
    "registration_qc_metrics",
    "rms_motion",
    "write_xcp_qc",
]
