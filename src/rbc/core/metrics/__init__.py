"""Metrics.

This module computes voxelwise metrics from preprocessed BOLD timeseries.
"""

from rbc.core.metrics.atlases import get_atlas
from rbc.core.metrics.timeseries import (
    compute_timeseries,
    correlation_matrix,
    extract_timeseries,
)

__all__ = [
    "compute_timeseries",
    "correlation_matrix",
    "extract_timeseries",
    "get_atlas",
]
