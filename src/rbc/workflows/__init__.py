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
from .metrics import (
    MetricsOutputs,
)
from .metrics import (
    single_session_metrics as metrics_pipeline,
)
from .qc import (
    QCOutputs,
)
from .qc import (
    single_session_qc as qc_pipeline,
)

__all__ = [
    "AnatomicalOutputs",
    "FunctionalOutputs",
    "MetricsOutputs",
    "QCOutputs",
    "anatomical_preprocess",
    "functional_preprocess",
    "metrics_pipeline",
    "qc_pipeline",
]
