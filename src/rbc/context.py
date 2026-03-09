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

from rbc.core.bids import BIDS_VERSION, bids_path

if TYPE_CHECKING:
    from pathlib import Path

_RBC_VERSION = version("rbc")


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
        desc: str | None = None,
        extension: str = ".nii.gz",
        task: str | None = None,
        run: int | None = None,
        space: str | None = None,
        atlas: str | None = None,
        extra: dict[str, str | int] | None = None,
    ) -> Path:
        """Copy *src* to a BIDS-named derivative path.

        Args:
            src: Source file to copy.
            datatype: BIDS datatype directory (e.g. ``"anat"``, ``"func"``).
            suffix: BIDS suffix (e.g. ``"T1w"``, ``"bold"``).
            desc: Optional ``desc-`` entity.
            extension: File extension including leading dot.
            task: Optional ``task-`` entity.
            run: Optional ``run-`` index.
            space: Optional ``space-`` entity.
            atlas: Optional ``atlas-`` entity.
            extra: Non-standard entities (e.g. ``{"from": "T1w"}``).

        Returns:
            Path to the copied output file.
        """
        rel = bids_path(
            sub=self.sub,
            ses=self.ses,
            task=task,
            run=run,
            desc=desc,
            space=space,
            atlas=atlas,
            extra=extra,
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
        desc: str | None = None,
        extension: str = "",
        task: str | None = None,
        run: int | None = None,
        space: str | None = None,
        atlas: str | None = None,
    ) -> Path:
        """Copy a directory to a BIDS-named derivative path.

        Args:
            src_dir: Source directory to copy (e.g. motion ``.mat`` dir).
            datatype: BIDS datatype directory.
            suffix: BIDS suffix.
            desc: Optional ``desc-`` entity.
            extension: File extension (usually empty for directories).
            task: Optional ``task-`` entity.
            run: Optional ``run-`` index.
            space: Optional ``space-`` entity.
            atlas: Optional ``atlas-`` entity.

        Returns:
            Path to the copied output directory.
        """
        rel = bids_path(
            sub=self.sub,
            ses=self.ses,
            task=task,
            run=run,
            desc=desc,
            space=space,
            atlas=atlas,
            suffix=suffix,
            extension=extension,
            datatype=datatype,
        )
        dest = self.output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dest, dirs_exist_ok=True)
        return dest

    def ensure_dataset_description(self) -> None:
        """Generate / append to dataset_description.json file."""
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
