"""bids2table helpers used throughout the pipeline.

Provides utilities for working with bids2table, including flattening extra entities
for simpler querying.
"""

from pathlib import Path

import bids2table as b2t
import polars as pl


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
    ses: str | None,
    datatype: str | None = None,
    suffix: str | None = None,
    desc: str | None = None,
    extension: str = "",
    task: str | None = None,
    run: int | None = None,
    extra: dict[str, str | int] | None = None,
) -> Path:
    """Return existing BIDS-named path matching provided entities.

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
        extra: Optional non-standard entities (e.g. ``{"from": "T1w"}``).

    Returns:
        Path to the BIDS named file.

    Raises:
        FileNotFoundError: If no matching rows with provided BIDS entities
        ValueError: If multiple matches found with provided BIDS entities
    """
    expr = pl.col("sub") == sub
    if ses is not None:
        expr &= pl.col("ses") == ses
    if datatype is not None:
        expr &= pl.col("datatype") == datatype
    if suffix is not None:
        expr &= pl.col("suffix") == suffix
    if desc is not None:
        expr &= pl.col("desc") == desc
    if extension:
        expr &= pl.col("ext").str.contains(extension)
    if task is not None:
        expr &= pl.col("task") == task
    if run is not None:
        expr &= pl.col("run") == run
    if extra:
        for key, val in extra.items():
            expr &= get_extra_entity(key) == val

    result = df.filter(expr)

    match len(result):
        case 0:
            raise FileNotFoundError(
                f"No BIDS file found for sub={sub!r}, ses={ses!r}, "
                f"datatype={datatype!r}, suffix={suffix!r}, desc={desc!r}"
            )
        case 1:
            row = result.row(0, named=True)
            return Path(row["root"]) / row["path"]
        case _:
            raise ValueError(
                f"Expected 1 match but found {len(result)} for sub={sub!r}, "
                f"ses={ses!r}, datatype={datatype!r}, suffix={suffix!r}, desc={desc!r}"
            )
