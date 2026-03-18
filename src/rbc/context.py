"""Pipeline context for BIDS-compliant derivative export.

Holds subject identity and output directory, providing ``export()`` and
``export_dir()`` helpers that copy processing results to BIDS-named paths.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

from rbc.core.bids import BIDS_VERSION, BidsEntities, bids_path, bids_safe_label

if TYPE_CHECKING:
    from pathlib import Path

_RBC_VERSION = version("rbc")


def _sanitize_extra(
    extra: dict[str, str | int] | None,
) -> dict[str, str | int] | None:
    """Apply ``bids_safe_label`` to string values in an *extra* dict."""
    if extra is None:
        return None
    return {
        k: bids_safe_label(v) if isinstance(v, str) else v for k, v in extra.items()
    }


@dataclass
class PipelineContext:
    """Minimal context for a single pipeline run.

    Attributes:
        sub: Subject label (without ``sub-`` prefix).
        ses: Session label (without ``ses-`` prefix), or *None*.
        output_dir: Root output directory (e.g. ``derivatives/rbc``).
    """

    sub: str
    ses: str | None
    output_dir: Path

    def export(
        self,
        src: Path,
        *,
        datatype: str,
        suffix: str,
        entities: BidsEntities | None = None, 
        extension: str = ".nii.gz",
        extra: dict[str, str | int] | None = None,
    ) -> Path:
        """Copy *src* to a BIDS-named derivative path.

        Args:
            src: Source file to copy.
            datatype: BIDS datatype directory (e.g. ``"anat"``, ``"func"``).
            suffix: BIDS suffix (e.g. ``"T1w"``, ``"bold"``).
            entities: Optional BIDS entities.
            extension: File extension including leading dot.
            extra: Non-standard entities (e.g. ``{"from": "T1w"}``).

        Returns:
            Path to the copied output file.
        """
        entities = entities or {}
        rel = bids_path(
            sub=self.sub,
            ses=self.ses,
            task=entities.get("task"),
            run=entities.get("run"),
            acq=entities.get("acq"),
            dir=entities.get("dir"),
            echo=entities.get("echo"),
            part=entities.get("part"),
            rec=entities.get("rec"),
            space=entities.get("space"),
            atlas=bids_safe_label(entities["atlas"]) if "atlas" in entities else None,
            desc=bids_safe_label(entities["desc"]) if "desc" in entities else None,
            extra=_sanitize_extra(extra),
            suffix=suffix,
            extension=extension,
            datatype=datatype,
        )
        dest = self.output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def export_dir(
        self,
        src_dir: Path,
        *,
        datatype: str,
        suffix: str,
        entities: BidsEntities | None = None,
        extension: str = "",
    ) -> Path:
        """Copy a directory to a BIDS-named derivative path.

        Args:
            src_dir: Source directory to copy (e.g. motion ``.mat`` dir).
            datatype: BIDS datatype directory.
            suffix: BIDS suffix.
            entities: Optional BIDS entities.
            extension: File extension (usually empty for directories).

        Returns:
            Path to the copied output directory.
        """
        entities = entities or {}
        rel = bids_path(
            sub=self.sub,
            ses=self.ses,
            task=entities.get("task"),
            run=entities.get("run"),
            space=entities.get("space"),
            desc=bids_safe_label(entities["desc"]) if "desc" in entities else None,
            atlas=bids_safe_label(entities["atlas"]) if "atlas" in entities else None,
            suffix=suffix,
            extension=extension,
            datatype=datatype,
        )
        dest = self.output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dest, dirs_exist_ok=True)
        return dest

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
