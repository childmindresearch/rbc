"""BIDS dataset querying utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class SessionTables(NamedTuple):
    """Subject-Session combination tables for anatomical and functional."""

    anat: pl.DataFrame
    func: pl.DataFrame | None


def load_session(df: pl.DataFrame, subject: str, session: str | None) -> SessionTables:
    """Filter for anatomical and functional data for a single subject/session.

    Args:
        df: Full bids2table dataframe.
        subject: Subject label without 'sub-' prefix (e.g. ``'01'``)
        session: Session label without 'ses-' prefix (e.g. ``'02'``)

    Returns:
        A :class:`SessionTables` containing separate anatomical and functional
            dataframes.
    """
    base: list[pl.Expr] = [
        pl.col("ext").str.contains(".nii"),
        pl.col("sub") == subject,
    ]
    if session is not None:
        base.append(pl.col("ses") == session)
    anat_df = df.filter(pl.all_horizontal([*base, pl.col("datatype") == "anat"]))
    func_df = df.filter(pl.all_horizontal([*base, pl.col("datatype") == "func"]))

    return SessionTables(anat=anat_df, func=func_df if not func_df.is_empty() else None)


def _resolve_anat(
    primary_group: pl.DataFrame,
    anat: pl.DataFrame,
    fallback_anat: pl.DataFrame,
    *,
    runs_correspond: bool,
) -> pl.DataFrame:
    """Resolve the anat subset for a given primary group."""
    if runs_correspond:
        run_vals = primary_group["run"].drop_nulls().unique()
        matched = anat.filter(pl.col("run").is_in(run_vals))
        return matched if not matched.is_empty() else fallback_anat
    return fallback_anat


def iter_session_files(
    session: SessionTables,
    groupby: Sequence[str] = ("run"),
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    """Iterate over run/task combos, paired with matching anat files.

    When functional data is present it drives iteration. For a pure anatomical
    pipeline (``session.func is None``), iteration is driven by the anat groups
    instead and each yield is ``(anat_group, anat_group)``.

    Anat matching follows this precedence:

    1. **1-to-1**: anat and func have the same number of runs — match by run label.
    2. **1-to-many**: run counts differ — use the anat for the first run.
    3. **No runs**: no run labels on either side — use available anat (e.g. single T1w).

    Args:
        session: A :class:`SessionTables` for a single subject/session.
        groupby: Sequence of BIDS entities to group the primary dataframe by.

    Yields:
        ``(primary_group, anat_subset)`` tuples. For functional pipelines
        ``primary_group`` is a func group; for anat-only pipelines both values
        are the same anat group.
    """
    has_anat_runs = (
        "run" in session.anat.columns and session.anat["run"].drop_nulls().len() > 0
    )
    anat_runs = (
        session.anat["run"].drop_nulls().unique()
        if has_anat_runs
        else pl.Series([], dtype=pl.Utf8)
    )

    if has_anat_runs:
        first_run = anat_runs.sort()[0]
        fallback_anat = session.anat.filter(pl.col("run") == first_run)
    else:
        fallback_anat = session.anat

    # Iteration from anat directly
    if session.func is None:
        for _, group in session.anat.group_by(groupby):
            yield group, group
        return

    func_runs = (
        session.func["run"].drop_nulls().unique()
        if "run" in session.func.columns
        else pl.Series([], dtype=pl.Utf8)
    )

    for _, func_group in session.func.group_by(groupby):
        anat_subset = _resolve_anat(
            func_group,
            session.anat,
            fallback_anat,
            runs_correspond=has_anat_runs and len(anat_runs) == len(func_runs),
        )
        yield func_group, anat_subset
