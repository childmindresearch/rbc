"""Functional processing.

This module defines functional MRI processing methods.

Functional MRI measures brain activity over time via the BOLD signal. Raw
fMRI contains motion artifacts, timing differences, and distortions that
must be corrected before analysis.
"""

from __future__ import annotations

from .coregistration import coregister_bold_to_t1w
from .despiking import despike_bold
from .distortion import (
    BidsPhaseEncoding,
    PEPolarFieldmap,
    PhaseDiffFieldmap,
    correct_distortion_pepolar,
    correct_distortion_phasediff,
    is_valid_pe_direction,
)
from .erosion import (
    erode_brain_mask,
    erode_csf_mask,
    erode_mask_by_distance,
    erode_mask_to_proportion,
    erode_wm_mask,
)
from .initialization import scale_bold, truncate_trs
from .masking import bold_masking
from .motion import extract_motion_reference, fsl_motion_correction
from .nuisance import (
    apply_regression,
    apply_regression_bandpass,
    bandpass_regressor_file,
    compute_regressors,
    nuisance_regression,
)
from .regressors import (
    assemble_36param_regressors,
    assemble_acompcor_regressors,
    check_regressor_rank,
    compute_acompcor,
    expand_motion_params,
    extract_mean_signal,
    write_regressor_file,
)
from .resampling import apply_motion_transforms, resample_bold_to_template
from .timing import slice_timing_correction

__all__ = [
    "BidsPhaseEncoding",
    "PEPolarFieldmap",
    "PhaseDiffFieldmap",
    "apply_motion_transforms",
    "apply_regression",
    "apply_regression_bandpass",
    "assemble_36param_regressors",
    "assemble_acompcor_regressors",
    "bandpass_regressor_file",
    "bold_masking",
    "check_regressor_rank",
    "compute_acompcor",
    "compute_regressors",
    "coregister_bold_to_t1w",
    "correct_distortion_pepolar",
    "correct_distortion_phasediff",
    "despike_bold",
    "erode_brain_mask",
    "erode_csf_mask",
    "erode_mask_by_distance",
    "erode_mask_to_proportion",
    "erode_wm_mask",
    "expand_motion_params",
    "extract_mean_signal",
    "extract_motion_reference",
    "fsl_motion_correction",
    "is_valid_pe_direction",
    "nuisance_regression",
    "resample_bold_to_template",
    "scale_bold",
    "slice_timing_correction",
    "truncate_trs",
    "write_regressor_file",
]
