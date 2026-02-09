"""Utility methods."""

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from bids2table import parse_bids_entities


@contextmanager
def create_copy(in_file: str | Path) -> Iterator[Path]:
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


def get_base_entities(
    in_file: Path, base_entities: Iterable[str] = ("sub", "ses", "run")
) -> dict[str, str]:
    """Parse base BIDS entities to be used for file naming.

    Args:
        in_file: Path to parse bids entities for.
        base_entities: List of base entities to extract (default: ['sub', 'ses' run'])

    Returns:
        A string-mapping of BIDS entities to values.
    """
    file_entities = parse_bids_entities(in_file)
    return {k: v for k, v in file_entities.items() if k in base_entities}


def rename(in_file: str | Path, new_name: str | Path) -> Path:
    """Rename a file, keeping it in the same directory."""
    target = in_file.with_name(new_name)
    if in_file == target:
        return in_file
    return Path(shutil.move(str(in_file), str(target)))

def save_directory(in_dir: str | Path, out_dir: str | Path, name: str) -> Path:
    """Save a directory by copying it to a new location.

    Args:
        in_dir: Path to directory to copy.
        out_dir: Path to destination directory.
        name: BIDS compliant name for the directory.
    """
    in_dir = Path(in_dir)
    target = Path(out_dir) / name

    if not in_dir.is_dir():
        raise ValueError(f"Input path {in_dir} is not a directory.")

    shutil.copytree(in_dir, target, dirs_exist_ok=True)
    return target
