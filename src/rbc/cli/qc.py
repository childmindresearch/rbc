"""CLI subcommand for quality control.

Computes QC metrics (framewise displacement, DVARS, registration overlap,
etc.) from preprocessed outputs and generates reports with pass/fail flags
based on RBC-defined thresholds. Use this after preprocessing to identify
sessions that may need exclusion.
"""

from __future__ import annotations
