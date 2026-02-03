"""Anatomical preprocessing.

This module defines anatomical MRI preprocessing steps for RBC pipeline.

Anatomical preprocessing prepares T1-weighted structural brain images for
analysis. The T1 provides high-resolution anatomy used to align functional
data and identify tissue types (gray matter, white matter, CSF).

Steps: reorientation, skull stripping (ANTs), N4 bias correction, tissue
segmentation (FSL FAST), and registration to MNI152 template.
"""
