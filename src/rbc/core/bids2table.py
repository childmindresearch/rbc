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
    """
    if index_fpath is not None and Path(index_fpath).exists():
        return pl.read_parquet(index_fpath)

    tables = b2t.batch_index_dataset(
        b2t.find_bids_datasets(dataset_dir),
        max_workers=max_workers,
        show_progress=verbose,
    )
    df = pl.concat([pl.from_arrow(table) for table in tables])
    if isinstance(df, pl.Series):
        df = df.to_frame()

    if index_fpath is not None:
        df.write_parquet(file=index_fpath)
    return df


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
