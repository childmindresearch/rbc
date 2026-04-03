"""Consolidated BIDS package for the RBC pipeline.

Re-exports schema definitions, the :class:`~rbc.bids.schema.Bids` builder,
and bids2table query helpers so callers can write
``from rbc.bids import Bids, Datatype, ...``.
"""

from rbc.bids._schema import (
    _STANDARD_ENTITIES,
    BIDS_VERSION,
    BidsEntities,
    BIDSFile,
    Datatype,
    EntityKwargs,
    Extension,
    Modality,
    Suffix,
    TemplateSpace,
    bids_name,
    bids_name_from_entities,
    bids_path,
    bids_path_from_entities,
    bids_safe_label,
    extract_entities,
    parse_bids_name,
)
from rbc.bids.anatomical import export_anatomical
from rbc.bids.builder import Bids
from rbc.bids.functional import FunctionalInputs, export_functional, resolve_functional
from rbc.bids.metrics import MetricsInputs, export_metrics, resolve_metrics
from rbc.bids.qc import QCInputs, export_qc, resolve_qc
from rbc.bids.query import find_file, find_files, get_extra_entity, load_table
from rbc.bids.session import (
    ANAT_GROUP_ENTITIES,
    FUNC_GROUP_ENTITIES,
    SUB_SES_QUERY,
    SessionTables,
    iter_session_files,
    load_session,
)

__all__ = [
    "ANAT_GROUP_ENTITIES",
    "BIDS_VERSION",
    "FUNC_GROUP_ENTITIES",
    "SUB_SES_QUERY",
    "_STANDARD_ENTITIES",
    "BIDSFile",
    "Bids",
    "BidsEntities",
    "Datatype",
    "EntityKwargs",
    "Extension",
    "FunctionalInputs",
    "MetricsInputs",
    "Modality",
    "QCInputs",
    "SessionTables",
    "Suffix",
    "TemplateSpace",
    "bids_name",
    "bids_name_from_entities",
    "bids_path",
    "bids_path_from_entities",
    "bids_safe_label",
    "export_anatomical",
    "export_functional",
    "export_metrics",
    "export_qc",
    "extract_entities",
    "find_file",
    "find_files",
    "get_extra_entity",
    "iter_session_files",
    "load_session",
    "load_table",
    "parse_bids_name",
    "resolve_functional",
    "resolve_metrics",
    "resolve_qc",
]
