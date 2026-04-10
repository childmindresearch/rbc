"""BIDS discovery and export for the anatomical workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from rbc.bids import Suffix, TemplateSpace, extract_entities
from rbc.bids.session import ANAT_GROUP_ENTITIES, SessionTables

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rbc.bids import Bids, EntityKwargs
    from rbc.workflows.anatomical import AnatomicalOutputs


class AnatomicalRun(NamedTuple):
    """A single anatomical run discovered from a BIDS session.

    Attributes:
        path: Path to the T1w NIfTI file.
        entities: BIDS entities for this run (run, acq, rec, echo).
    """

    path: Path
    entities: EntityKwargs


def discover_anatomical(session: SessionTables) -> Iterator[AnatomicalRun]:
    """Discover T1w runs in a session's anatomical data.

    Filters for T1w files and groups by anatomical entities, yielding
    one :class:`AnatomicalRun` per group.

    Args:
        session: Session tables from :func:`~rbc.bids.session.load_session`.

    Yields:
        An :class:`AnatomicalRun` for each T1w group.
    """
    for _, anat_df in session.anat.filter(pl.col("suffix") == "T1w").group_by(
        ANAT_GROUP_ENTITIES, maintain_order=True
    ):
        row = anat_df.row(0, named=True)
        yield AnatomicalRun(
            path=Path(row["root"]) / row["path"],
            entities=extract_entities(row, ["run", "acq", "rec", "echo"]),
        )


def export_anatomical(anat: Bids, outputs: AnatomicalOutputs) -> None:
    """Export anatomical workflow outputs to BIDS-named derivatives.

    Args:
        anat: Bids builder configured with ``datatype=ANAT`` and identity
            entities (run, acq, etc.).
        outputs: Results from the anatomical preprocessing workflow.
    """
    anat.save(outputs.brain, suffix=Suffix.T1W, desc="brain")
    anat.save(outputs.brain_mask, suffix=Suffix.MASK, desc="T1w")
    anat.save(outputs.csf_mask, suffix=Suffix.MASK, desc="csf")
    anat.save(outputs.gm_mask, suffix=Suffix.MASK, desc="gm")
    anat.save(outputs.wm_mask, suffix=Suffix.MASK, desc="wm")
    anat.save(outputs.wm_bbr_mask, suffix=Suffix.MASK, desc="wmBBR")
    anat.save(
        outputs.anat_to_template_xfm,
        suffix="xfm",
        extra={
            "from": "T1w",
            "to": TemplateSpace.MNI152NLIN6ASYM,
            "mode": "image",
        },
    )
    anat.save(
        outputs.template_to_anat_xfm,
        suffix="xfm",
        extra={
            "from": TemplateSpace.MNI152NLIN6ASYM,
            "to": "T1w",
            "mode": "image",
        },
    )
    mni = anat.derive(space=TemplateSpace.MNI152NLIN6ASYM)
    mni.save(outputs.brain_tpl, suffix=Suffix.T1W, desc="brain")
