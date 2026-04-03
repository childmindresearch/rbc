"""Command-line interface for the RBC pipeline.

Entry point that parses arguments, validates BIDS inputs, and dispatches to
the appropriate workflow. Each subcommand (``anatomical``, ``functional``,
``derivatives``, ``longitudinal``, ``qc``) is defined in its own submodule
under ``rbc.cli``.
"""

from __future__ import annotations
