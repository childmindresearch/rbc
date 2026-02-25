"""End-to-end RBC processing workflows.

Each workflow orchestrates the core processing steps (defined in ``rbc.core``)
into a complete pipeline, returning output paths as named tuples.
"""

from __future__ import annotations

from .anatomical import (
    AnatomicalOutputs,
)
from .anatomical import (
    single_session_preprocess as anatomical_preprocess,
)
from .functional import (
    FunctionalOutputs,
)
from .functional import (
    single_session_preprocess as functional_preprocess,
)

__all__ = [
    "AnatomicalOutputs",
    "FunctionalOutputs",
    "anatomical_preprocess",
    "functional_preprocess",
]
