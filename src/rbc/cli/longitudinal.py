"""CLI subcommand for longitudinal processing.

For subjects with multiple sessions, this builds an unbiased within-subject
anatomical template and processes each session relative to it. This improves
sensitivity for detecting changes over time by reducing session-specific
registration bias.
"""

from __future__ import annotations
