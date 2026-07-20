"""Unit tests for Filters.apply() -- zero mocks, real DataFrames."""

from __future__ import annotations

import polars as pl
import pytest

from rbc.orchestration import Filters


@pytest.fixture
def bids_df() -> pl.DataFrame:
    """Minimal BIDS-like table covering common filtering scenarios.

    Contains:
    - sub-01/ses-baseline: anat T1w + func bold (task-rest)
    - sub-01/ses-vis2: func bold (task-nback)
    - sub-02/ses-baseline: anat T1w + func bold (task-rest)
    - sub-01/ses-longitudinal: anat T1w (longitudinal template)
    - sub-01/ses-baseline: func bold preproc MNI (derivative)
    - sub-01/ses-baseline: func bold preproc T1w (derivative, non-MNI)
    """
    return pl.DataFrame(
        {
            "sub": ["01", "01", "01", "02", "02", "01", "01", "01", "01"],
            "ses": [
                "baseline",
                "baseline",
                "vis2",
                "baseline",
                "baseline",
                "longitudinal",
                "baseline",
                "baseline",
                None,
            ],
            "datatype": [
                "anat",
                "func",
                "func",
                "anat",
                "func",
                "anat",
                "func",
                "func",
                None,
            ],
            "suffix": [
                "T1w",
                "bold",
                "bold",
                "T1w",
                "bold",
                "T1w",
                "bold",
                "bold",
                "sessions",
            ],
            "task": [None, "rest", "nback", None, "rest", None, "rest", "rest", None],
            "space": [
                None,
                None,
                None,
                None,
                None,
                None,
                "MNI152NLin6Asym",
                "T1w",
                None,
            ],
            "desc": [None, None, None, None, None, None, "preproc", "preproc", None],
        }
    )


class TestFiltersApply:
    """Tests for Filters.apply() with real polars DataFrames."""

    def test_empty_filters_no_base_returns_all(self, bids_df: pl.DataFrame) -> None:
        """No filters and no base expressions returns the full DataFrame."""
        result = Filters().apply(bids_df)
        assert len(result) == len(bids_df)

    def test_participant_filter(self, bids_df: pl.DataFrame) -> None:
        """Participant filter keeps only matching subjects."""
        result = Filters(participant_label=["01"]).apply(bids_df)
        assert set(result["sub"].unique().to_list()) == {"01"}
        assert len(result) == 7

    def test_session_filter(self, bids_df: pl.DataFrame) -> None:
        """Session filter keeps only matching sessions."""
        result = Filters(session_label=["baseline"]).apply(bids_df)
        assert set(result["ses"].unique().to_list()) == {"baseline"}

    def test_task_filter_preserves_anat(self, bids_df: pl.DataFrame) -> None:
        """Task filter keeps matching tasks AND rows with null task (anat)."""
        result = Filters(task="rest").apply(bids_df)
        tasks = result["task"].to_list()
        assert all(t in ("rest", None) for t in tasks)
        # Anat rows (task=null) must survive
        assert result.filter(pl.col("datatype") == "anat").height > 0

    def test_combined_filters(self, bids_df: pl.DataFrame) -> None:
        """Participant + session + task filters compose correctly."""
        result = Filters(
            participant_label=["01"],
            session_label=["baseline"],
            task="rest",
        ).apply(bids_df)
        # anat T1w (task=null) + raw bold + MNI preproc + T1w preproc = 4
        assert len(result) == 4
        assert all(s == "01" for s in result["sub"].to_list())
        assert all(s == "baseline" for s in result["ses"].to_list())

    def test_no_matches_returns_empty(self, bids_df: pl.DataFrame) -> None:
        """Filters that match nothing return an empty DataFrame."""
        result = Filters(participant_label=["99"]).apply(bids_df)
        assert len(result) == 0
        assert result.columns == bids_df.columns

    def test_multi_participant(self, bids_df: pl.DataFrame) -> None:
        """Multiple participant labels are OR-combined."""
        result = Filters(participant_label=["01", "02"]).apply(bids_df)
        assert len(result) == len(bids_df)

    def test_multi_session(self, bids_df: pl.DataFrame) -> None:
        """Multiple session labels are OR-combined."""
        result = Filters(session_label=["baseline", "vis2"]).apply(bids_df)
        assert "longitudinal" not in result["ses"].to_list()

    def test_datatype_filter(self, bids_df: pl.DataFrame) -> None:
        """Datatype filter keeps only those in a datatype directory."""
        result = Filters().apply(bids_df, pl.col("datatype").is_not_null())
        assert len(result) == 8
        assert all(result["datatype"].is_not_null())


class TestFiltersWithBaseExpressions:
    """Tests for Filters.apply() with workflow-specific base expressions."""

    def test_anatomical_base_filters(self, bids_df: pl.DataFrame) -> None:
        """Anatomical base expressions: exclude longitudinal, null space/desc."""
        result = Filters().apply(
            bids_df,
            pl.col("ses") != "longitudinal",
            pl.col("space").is_null(),
            pl.col("desc").is_null(),
        )
        assert "longitudinal" not in result["ses"].to_list()
        assert all(s is None for s in result["space"].to_list())
        assert all(d is None for d in result["desc"].to_list())

    def test_derivative_base_filters(self, bids_df: pl.DataFrame) -> None:
        """Metrics/QC base expressions: func bold preproc MNI only."""
        result = Filters().apply(
            bids_df,
            pl.col("datatype") == "func",
            pl.col("suffix") == "bold",
            pl.col("desc") == "preproc",
            pl.col("space") == "MNI152NLin6Asym",
        )
        assert len(result) == 1
        assert result["space"][0] == "MNI152NLin6Asym"

    def test_base_expressions_with_user_filters(self, bids_df: pl.DataFrame) -> None:
        """Base expressions compose with user-level filters."""
        result = Filters(participant_label=["01"]).apply(
            bids_df,
            pl.col("ses") != "longitudinal",
            pl.col("space").is_null(),
        )
        assert "longitudinal" not in result["ses"].to_list()
        assert set(result["sub"].unique().to_list()) == {"01"}

    def test_longitudinal_base_filter(self, bids_df: pl.DataFrame) -> None:
        """Longitudinal base expression: exclude ses-longitudinal only."""
        result = Filters().apply(bids_df, pl.col("ses") != "longitudinal")
        assert "longitudinal" not in result["ses"].to_list()
        assert len(result) == 7
