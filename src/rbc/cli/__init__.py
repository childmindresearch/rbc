"""Command-line interface for the RBC pipeline.

Entry point that parses arguments, validates BIDS inputs, and dispatches to
the appropriate workflow. Each subcommand (``anatomical``, ``functional``,
``derivatives``, ``longitudinal``, ``qc``) is defined in its own submodule
under ``rbc.cli``.
"""

from __future__ import annotations

from rbc.core import CPAC_ANTS_SEED

_DEFAULT_ENV_VARS = {
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "ANTS_RANDOM_SEED": CPAC_ANTS_SEED,
}

_SUB_SES_QUERY = ("sub", "ses")
# Suffix to split up T1w vs T2w vs something else
_ANAT_GROUP = ("run", "acq", "suffix", "part", "echo", "ce", "rec", "inv")
# Ignore suffix to keep sbref with bold
_FUNC_GROUP = ("task", "run", "acq", "dir", "echo", "part", "rec")
