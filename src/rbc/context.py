"""Pipeline context for BIDS-compliant derivative export.

Holds subject identity and output directory, providing a :meth:`bids` factory
that returns a :class:`~rbc.bids.Bids` builder for composing exports
and queries.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

from rbc.bids import BIDS_VERSION, Bids, EntityKwargs

if TYPE_CHECKING:
    from pathlib import Path

_RBC_VERSION = version("rbc")


@dataclass(frozen=True)
class RunContext:
    """Minimal context for a single pipeline run.

    Attributes:
        sub: Subject label (without ``sub-`` prefix).
        ses: Session label (without ``ses-`` prefix), or *None*.
        output_dir: Root output directory (e.g. ``derivatives/rbc``).
    """

    sub: str
    ses: str | None
    output_dir: Path

    def bids(
        self,
        *,
        datatype: str | None = None,
        entities: EntityKwargs | None = None,
        extra: dict[str, str | int] | None = None,
        **overrides: str | int,
    ) -> Bids:
        """Create a :class:`~rbc.bids.Bids` builder bound to this context.

        Args:
            datatype: BIDS datatype directory (e.g. ``"anat"``, ``"func"``).
            entities: Identity entities from the input file.
            extra: Non-standard entity defaults.
            **overrides: Individual entity overrides.

        Returns:
            A :class:`~rbc.bids.Bids` builder ready for
            ``.derive()`` / ``.save()`` / ``.find()`` calls.
        """
        merged: dict[str, str | int] = {**(entities or {}), **overrides}  # type: ignore[dict-item]
        return Bids(
            _sub=self.sub,
            _ses=self.ses,
            _output_dir=self.output_dir,
            _datatype=datatype,
            _entities=merged,
            _extra=extra,
        )

    def ensure_dataset_description(self) -> None:
        """Create dataset_description.json in output directory if it doesn't exist."""
        _ensure_dataset_description(output_dir=self.output_dir)


def _ensure_dataset_description(output_dir: Path) -> None:
    """Create dataset_description.json file in a directory if it doesn't exist."""
    ds_file = output_dir / "dataset_description.json"
    if ds_file.exists():
        return

    ds_file.parent.mkdir(parents=True, exist_ok=True)
    ds_data = {
        "Name": "RBC Outputs",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "ReferencesAndLinks": ["https://doi.org/10.1016/j.neuron.2025.08.026"],
        "GeneratedBy": [
            {
                "Name": "RBC",
                "Version": _RBC_VERSION,
                "CodeURL": "https://github.com/childmindresearch/rbc",
            }
        ],
    }
    with ds_file.open("w") as fpath:
        json.dump(ds_data, fpath, indent=2)
