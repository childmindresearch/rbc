"""BIDS export for the anatomical workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.bids import Suffix, TemplateSpace

if TYPE_CHECKING:
    from rbc.bids import Bids
    from rbc.workflows.anatomical import AnatomicalOutputs


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
        outputs.forward_xfm,
        suffix="xfm",
        extra={
            "from": "T1w",
            "to": TemplateSpace.MNI152NLIN6ASYM,
            "mode": "image",
        },
    )
    anat.save(
        outputs.inverse_xfm,
        suffix="xfm",
        extra={
            "from": TemplateSpace.MNI152NLIN6ASYM,
            "to": "T1w",
            "mode": "image",
        },
    )
