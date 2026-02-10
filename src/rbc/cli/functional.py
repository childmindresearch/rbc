"""CLI subcommand for functional processing.

Parses subject/session/task arguments and delegates to
``rbc.workflows.functional.single_session``, which runs the functional
stream (reorientation -> TR truncation -> motion correction). Anatomical
preprocessing must be completed first since coregistration and template
warping depend on the anatomical outputs.
"""

from __future__ import annotations
