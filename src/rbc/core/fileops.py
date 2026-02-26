"""File-system helpers used throughout the pipeline.

Provides utilities for safely copying, renaming, and temporarily duplicating
files. The temporary-copy context manager is especially useful for AFNI tools
like ``3drefit`` that modify files in-place -- it lets us work on a throwaway
copy so the original input is never altered.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["file_copy_many", "file_rename", "file_tmp_copy"]


@contextmanager
def file_tmp_copy(
    in_file: str | Path, base_dir: str | Path | None = None
) -> Iterator[Path]:
    """Context manager that yields a temporary copy of a file.

    Useful for tools that modify images in-place (e.g. ``3drefit``). The copy
    lives in a fresh temp directory and is cleaned up automatically on exit.

    Args:
        in_file: Path to the file to copy.
        base_dir: Parent directory to create a temporary directory in.

    Yields:
        Path to the temporary copy (safe to modify in-place).
    """
    in_file = Path(in_file)
    tmp_dir = Path(tempfile.mkdtemp(dir=base_dir))
    try:
        tmp_path = tmp_dir / in_file.name
        shutil.copy2(in_file, tmp_path)
        yield tmp_path
    finally:
        shutil.rmtree(tmp_dir)


def file_rename(in_file: str | Path, new_name: str) -> Path:
    """Rename a file in-place, keeping it in the same directory.

    Raises ``FileExistsError`` if the target name already exists to prevent
    silent overwrites.
    """
    in_file = Path(in_file)
    new_path = in_file.with_name(new_name)
    if new_path.exists():
        raise FileExistsError(f"Target file already exists: {new_path}")
    return in_file.rename(new_path)


def file_copy_many(files: Iterable[str | Path], out_dir: str | Path) -> None:
    """Copy files to an output directory.

    Args:
        files: Paths to copy.
        out_dir: Destination directory (created if it doesn't exist).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, out_dir / Path(file).name)
