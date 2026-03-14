"""Core processing steps for the RBC pipeline.

Each submodule implements one or more steps from the RBC preprocessing pipeline
(reorientation, brain extraction, segmentation, registration, motion correction,
etc.) as thin wrappers around neuroimaging tools (AFNI, FSL, ANTs). Workflows in
``rbc.workflows`` compose these steps into end-to-end pipelines.
"""

from __future__ import annotations

CPAC_ANTS_SEED = 77742777

__all__ = [
    "CPAC_ANTS_SEED",
    "anatomical",
    "bids",
    "bids2table",
    "common",
    "fileops",
    "functional",
    "longitudinal",
    "metrics",
    "nifti",
    "niwrap",
    "qc",
    "resources",
]
