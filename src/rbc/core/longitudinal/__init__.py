"""Longitudinal processing.

This module handles longitudinal processing for multi-session data.

Processing steps include the creation of unbiased within-subject anatomical template
and processes each session relative to it, improving sensitivity for detecting changes
over time.
"""

from __future__ import annotations
