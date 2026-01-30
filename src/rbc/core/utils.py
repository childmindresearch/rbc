"""Utility methods."""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bids2table import parse_bids_entities


@contextmanager
def create_copy(in_file: Path) -> Iterator[Path]:
    """Create a temporary copy a file.

    Args:
        in_file: Path to file for creating a copy of.

    Returns:
        Yields a file path to the copy of the temporary file.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / in_file.name
    shutil.copy2(in_file, tmp_path)

    try:
        yield tmp_path
    finally:
        shutil.rmtree(tmp_dir)


def get_base_entities(in_file: Path) -> dict[str, str]:
    """Parse base entities to be used for file naming.

    Args:
        in_file: Path to parse bids entities for.

    Returns:
        A string-mapping of BIDS entities to values.
    """
    file_entities = parse_bids_entities(in_file)
    return {k: v for k, v in file_entities.items() if k in ["sub", "ses", "run"]}


def rename(in_file: Path, new_name: str) -> Path:
    """Rename a file in place.

    Args:
        in_file: Path of file to be renamed.
        new_name: New name of file.

    Returns:
        Path to renamed file
    """
    new_path = in_file.with_name(new_name)
    in_file.rename(new_path)
    return new_path
