"""File operation helpers."""

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_tmp_copy(in_file: str | Path) -> Iterator[Path]:
    """Create a temporary copy of a file.

    Args:
        in_file: Path to file to copy.

    Yields:
        Path to the temporary copy.
    """
    in_file = Path(in_file)
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        tmp_path = tmp_dir / in_file.name
        shutil.copy2(in_file, tmp_path)
        yield tmp_path
    finally:
        shutil.rmtree(tmp_dir)


def file_rename(in_file: str | Path, new_name: str) -> Path:
    """Rename a file, keeping it in the same directory."""
    in_file = Path(in_file)
    new_path = in_file.with_name(new_name)
    if new_path.exists():
        raise FileExistsError(f"Target file already exists: {new_path}")
    return in_file.rename(new_path)


def file_save(files: Iterable[str | Path], out_dir: str | Path) -> None:
    """Copy files to an output directory.

    Args:
        files: Paths to copy.
        out_dir: Destination directory (created if it doesn't exist).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, out_dir / Path(file).name)
