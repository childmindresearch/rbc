"""Orchestration for the longitudinal workflow."""

from __future__ import annotations

from rbc.orchestration.longitudinal._iter import iter_sessions_with_template
from rbc.orchestration.longitudinal.anatomical import process_anat
from rbc.orchestration.longitudinal.functional import process_func
from rbc.orchestration.longitudinal.metrics import process_metrics
from rbc.orchestration.longitudinal.qc import process_qc

__all__ = [
    "iter_sessions_with_template",
    "process_anat",
    "process_func",
    "process_metrics",
    "process_qc",
]
