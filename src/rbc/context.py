"""Pipeline context for BIDS-compliant derivative export.

Holds subject identity and output directory, providing ``export()`` and
``export_dir()`` helpers that copy processing results to BIDS-named paths.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rbc.core.bids import bids_path

if TYPE_CHECKING:
    from pathlib import Path


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
