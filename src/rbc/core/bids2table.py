"""bids2table helpers used throughout the pipeline.

Provides utilities for working with bids2table, including flattening extra entities
for simpler querying.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import bids2table as b2t
import polars as pl

if TYPE_CHECKING:
    from rbc.core.bids import BidsEntities  # noqa: F401


def load_table(
    dataset_dir: str | Path,
    index_fpath: str | Path | None = None,
    max_workers: int | None = 0,
    verbose: bool = False,  # noqa: FBT001, FBT002 (Ignore bool arg for b2t)
) -> pl.DataFrame:
    """Get and return BIDSTable for a given dataset.

    Args:
        dataset_dir: Path to dataset directory.
        index_fpath: Path to bids2table parquet table. If provided and exists,
            will be loaded. Otherwise dataset will be indexed.
        max_workers: Number of parallel indexing processes. 0=main process only,
            None=use all CPUs.
        verbose: Show verbose messages.

    Returns:
        Polars DataFrame index for all BIDS datasets.

    Raises:
        ValueError: if no datasets found.
        TypeError: if found dataset does not return a DataFrame.
    """
    if index_fpath is not None:
        return pl.read_parquet(index_fpath)

    tables = b2t.batch_index_dataset(
        b2t.find_bids_datasets(dataset_dir),
        max_workers=max_workers,
        show_progress=verbose,
    )
    dfs: list[pl.DataFrame] = []
    for table in tables:
        result = pl.from_arrow(table)
        if not isinstance(result, pl.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(result)}")
        dfs.append(result)
    if len(dfs) == 0:
        raise ValueError(f"No datasets found in {dataset_dir}")

    return pl.concat(dfs)


def get_extra_entity(key: str) -> pl.Expr:
    """Extract a specific entity value from the extra_entities column from table.

    Args:
        key: The entity key to extract

    Returns:
        Polars expression extracting value associated with key.

    Example:
        >>> df.filter(get_extra_entity("foo") == "bar")

    """
    return (
        pl.col("extra_entities")
        .list.eval(
            pl.element()
            .filter(pl.element().struct.field("key") == key)
            .struct.field("value")
            .first()
        )
        .list.first()
    )


def get_file_path(  # noqa: C901 - handling multiple BIDS entities
    df: pl.DataFrame,
    *,
    sub: str,
    ses: str | bool | None,
    datatype: str | None = None,
    suffix: str | bool | None = None,
    desc: str | None = None,
    extension: str = "",
    task: str | bool | None = None,
    run: int | bool | None = None,
    space: str | bool | None = None,
    extra: dict[str, str | int] | None = None,
) -> Path:
    """Return existing BIDS-named path matching provided entities.

    Keyword arguments mirror :class:`~rbc.core.bids.BidsEntities`.

    Args:
        df: bids2table to filter
        sub: ``sub-`` entity
        ses: Optional ``ses-``entity
        datatype: BIDS datatype directory.
        suffix: BIDS suffix.
        desc: Optional ``desc-`` entity.
        extension: File extension (usually empty for directories).
        task: Optional ``task-`` entity.
        run: Optional ``run-`` index.
        space: Optional ``space-`` entity.
        extra: Optional non-standard entities (e.g. ``{"from": "T1w"}``).

    Returns:
        Path to the BIDS named file.

    Raises:
        FileNotFoundError: If no matching rows with provided BIDS entities
        ValueError: If multiple matches found with provided BIDS entities
    """

    def _filter(
        expr: pl.Expr,
        col: str,
        val: str | int | bool | None,  # noqa: FBT001 - bool indicator for b2t entity
    ) -> pl.Expr:
        """Helper to filter BIDS entities based on value provided."""
        if val is None or val is True:
            return expr
        if val is False:
            return expr & pl.col(col).is_null()
        return expr & (pl.col(col) == val)

    expr = pl.col("sub") == sub
    expr = _filter(expr, "ses", ses)
    if datatype is not None:
        expr &= pl.col("datatype") == datatype
    expr = _filter(expr, "suffix", suffix)
    expr = _filter(expr, "desc", desc)
    expr = _filter(expr, "task", task)
    expr = _filter(expr, "run", run)
    expr = _filter(expr, "space", space)
    if extension:
        expr &= pl.col("ext").str.contains(extension)
    if extra:
        for key, val in extra.items():
            if val is False:
                expr &= get_extra_entity(key).is_null()
            else:
                expr &= get_extra_entity(key) == val

    result = df.filter(expr)

    match len(result):
        case 0:
            raise FileNotFoundError(
                f"No BIDS file found for sub={sub!r}, ses={ses!r}, "
                f"datatype={datatype!r}, suffix={suffix!r}, desc={desc!r}, "
                f"task={task!r}, run={run!r}, space={space!r}, "
                f"extension={extension!r}, extra={extra!r}"
            )
        case 1:
            row = result.row(0, named=True)
            return Path(row["root"]) / row["path"]
        case _:
            raise ValueError(
                f"Expected 1 match but found {len(result)} for sub={sub!r}, "
                f"ses={ses!r}, datatype={datatype!r}, suffix={suffix!r}, "
                f"desc={desc!r}, task={task!r}, run={run!r}, space={space!r}, "
                f"extension={extension!r}, extra={extra!r}"
            )
