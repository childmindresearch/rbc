"""End-to-end RBC processing workflows.

Each workflow orchestrates the core processing steps (defined in ``rbc.core``)
into a complete pipeline, handling BIDS naming and output organization.
"""

from __future__ import annotations

from .anatomical import single_session

__all__ = ["single_session"]
