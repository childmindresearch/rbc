"""Project-specific BIDS constants for code generation.

These are merged into the generated ``src/rbc/core/bids.py`` alongside
constants extracted from the BIDS schema.  To add a new constant, add an
entry here and re-run ``uv run scripts/generate_bids_tools.py``.
"""

from __future__ import annotations

SUFFIXES: list[tuple[str, str]] = [
    ("xfm", "Spatial transformation"),
    ("regressors", "Nuisance regressors"),
    ("alff", "Amplitude of low frequency fluctuations"),
    ("falff", "Fractional ALFF"),
    ("reho", "Regional homogeneity"),
    ("timeseries", "Atlas time series"),
    ("correlations", "Correlation matrix"),
    ("quality", "Quality metrics"),
]

SPACES: list[tuple[str, str]] = [
    ("MNI152NLin6ASym", "MNI 152 non-linear 6th generation asymmetric"),
    ("MNI152NLin2009cAsym", "MNI 152 non-linear 2009c asymmetric"),
    ("longitudinal", "Subject-specific longitudinal template"),
]

DESCS: list[tuple[str, str]] = [
    ("brain", "Brain-extracted"),
    ("preproc", "Preprocessed"),
    ("T1w", "T1-weighted space"),
    ("csf", "Cerebrospinal fluid"),
    ("gm", "Gray matter"),
    ("wm", "White matter"),
    ("wmBBR", "White matter BBR mask"),
    ("motionParams", "Motion parameters"),
    ("relsDisplacement", "Relative RMS displacement"),
    ("maxDisplacement", "Max displacement"),
    ("linear", "Linear transformation"),
    ("bold", "BOLD space"),
    ("smooth", "Spatially smoothed"),
    ("smoothZstd", "Smoothed and z-standardized"),
    ("mean", "Temporal mean"),
    ("pearson", "Pearson correlation"),
    ("xcp", "XCP-D format"),
]
