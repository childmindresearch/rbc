"""Pipeline context for BIDS-compliant derivative export.

Holds subject identity and output directory, providing ``export()`` and
``export_dir()`` helpers that copy processing results to BIDS-named paths.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from filelock import BaseFileLock, FileLock, Timeout

from rbc.core.bids import BIDS_VERSION, bids_path

_RBC_VERSION = version("rbc")
_RBC_GENERATED_BY = {
    "Version": _RBC_VERSION,
    "CodeURL": "https://github.com/childmindresearch/rbc",
}

DEFAULT_DATASET_DESCRIPTION = {
    "Name": "RBC Outputs",
    "BIDSVersion": BIDS_VERSION,
    "DatasetType": "derivative",
    "ReferencesAndLinks": ["https://doi.org/10.1016/j.neuron.2025.08.026"],
    "GeneratedBy": [],
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
    _ds_lock: BaseFileLock = field(init=False)

    def __post_init__(self) -> None:
        """Post initialization of PipelineContext."""
        self._ds_lock = FileLock(self.output_dir / "dataset_description.json.lock")

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

    def generate_dataset_description(self, *, workflow: str) -> None:
        """Generate / append to dataset_description.json file.

        Args:
            workflow: specific pipeline run.
        """
        ds_file = self.output_dir / "dataset_description.json"
        ds_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._ds_lock.acquire(timeout=0):
                ds_data: dict[str, Any] = (
                    json.loads(ds_file.read_text())
                    if ds_file.exists()
                    else copy.deepcopy(DEFAULT_DATASET_DESCRIPTION)
                )

                new_entry = {"Name": f"RBC {workflow} pipeline", **_RBC_GENERATED_BY}
                generated_by: list[dict[str, str]] = ds_data.setdefault(
                    "GeneratedBy", []
                )
                if new_entry not in generated_by:
                    generated_by.append(new_entry)

                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=ds_file.parent, delete=False, suffix=".tmp"
                    ) as tmp:
                        json.dump(ds_data, tmp, indent=2)
                    tmp_path = Path(tmp.name)
                    tmp_path.replace(ds_file)
        except Timeout:
            return
