"""Functional preprocessing.

This module defines functional MRI preprocessing steps for RBC datasets.

Functional MRI measures brain activity over time via the BOLD signal. Raw
fMRI contains motion artifacts, timing differences, and distortions that
must be corrected before analysis.

Steps: reorientation, TR truncation, motion correction (mcflirt), slice
timing correction (3dTshift), despiking (3dDespike), brain masking,
coregistration to T1w (FLIRT BBR), and single-step resampling to 2mm MNI
space.
"""
