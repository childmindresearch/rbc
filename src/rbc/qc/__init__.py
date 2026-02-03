"""Quality control.

This module defines quality control metrics for RBC datasets.

Computes framewise displacement (FD), DVARS, motion-DVARS correlation, and
tSNR. Metrics are evaluated against RBC-recommended thresholds from Shafiei
et al. (2024). Outputs follow XCP-style formatting.
"""
