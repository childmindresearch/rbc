"""Unit tests for BIDS dataset querying utilities."""

import polars as pl
import pytest

from rbc.cli.query import SessionTables, _resolve_anat, iter_session_files, load_session


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Full BIDS-like dataframe spanning multiple subjects, sessions, and datatypes.

    Contains:
    - sub-01/ses-baseline: anat (T1w .nii.gz + .json sidecar) + func (rest, nback)
    - sub-02/ses-baseline: anat only (no func)
    - sub-01/no-session: anat only (for session=None testing)
    """
    return pl.DataFrame(
        {
            "datatype": ["anat", "anat", "anat", "func", "func", "anat"],
            "suffix": ["T1w", "T1w", "T1w", "bold", "bold", "T1w"],
            "ext": [".nii.gz", ".json", ".nii.gz", ".nii.gz", ".nii.gz", ".nii.gz"],
            "sub": ["01", "01", "02", "01", "01", "01"],
            "ses": ["baseline", "baseline", "baseline", "baseline", "baseline", None],
            "run": [None, None, None, None, None, None],
            "task": [None, None, None, "rest", "nback", None],
            "root": ["/data"] * 6,
            "path": [
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.json",
                "sub-02/ses-baseline/anat/sub-02_ses-baseline_T1w.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                "sub-01/ses-baseline/func/sub-01_ses-baseline_task-nback_bold.nii.gz",
                "sub-01/anat/sub-01_T1w.nii.gz",
            ],
        }
    )


class TestLoadSession:
    """Tests for load_session.

    Verifies subject/session filtering, datatype separation, sidecar exclusion,
    and edge cases such as missing func data or session-less subjects.
    """

    def test_returns_session_tables(self, sample_dataframe: pl.DataFrame) -> None:
        """Returns a SessionTables namedtuple for a valid subject/session."""
        result = load_session(sample_dataframe, "01", "baseline")
        assert isinstance(result, SessionTables)

    def test_filters_correct_subject(self, sample_dataframe: pl.DataFrame) -> None:
        """Only rows matching the requested subject are included in anat."""
        result = load_session(sample_dataframe, "01", "baseline")
        assert result.anat["sub"].unique().to_list() == ["01"]

    def test_filters_correct_session(self, sample_dataframe: pl.DataFrame) -> None:
        """Only rows matching the requested session are included in anat."""
        result = load_session(sample_dataframe, "01", "baseline")
        assert result.anat["ses"].unique().to_list() == ["baseline"]

    def test_anat_and_func_separated(self, sample_dataframe: pl.DataFrame) -> None:
        """Anat and func dataframes contain only their respective datatypes."""
        result = load_session(sample_dataframe, "01", "baseline")
        assert result.anat["datatype"].unique().to_list() == ["anat"]
        assert result.func is not None
        assert result.func["datatype"].unique().to_list() == ["func"]

    def test_func_is_none_when_no_func_data(
        self, sample_dataframe: pl.DataFrame
    ) -> None:
        """'func' is None when no functional data exists for the subject/session."""
        result = load_session(sample_dataframe, "02", "baseline")
        assert result.func is None

    def test_session_none_filters_without_ses(
        self, sample_dataframe: pl.DataFrame
    ) -> None:
        """When session is None, session filtering is skipped."""
        result = load_session(sample_dataframe, "01", None)
        assert not result.anat.is_empty()

    def test_unknown_subject_returns_empty(
        self, sample_dataframe: pl.DataFrame
    ) -> None:
        """Unknown subject returns an empty anat dataframe and None func."""
        result = load_session(sample_dataframe, "99", "baseline")
        assert result.anat.is_empty()
        assert result.func is None


class TestResolveAnat:
    """Tests for _resolve_anat.

    Verifies run-matching logic and fallback behaviour when runs do not
    correspond or no match is found.
    """

    @pytest.fixture
    def anat_with_runs(self) -> pl.DataFrame:
        """Anat dataframe with two runs (01 and 02)."""
        return pl.DataFrame(
            {
                "suffix": ["T1w", "T1w"],
                "run": ["01", "02"],
                "path": ["anat_run01.nii.gz", "anat_run02.nii.gz"],
            }
        )

    @pytest.fixture
    def fallback_anat(self) -> pl.DataFrame:
        """Fallback anat dataframe containing only run 01."""
        return pl.DataFrame(
            {
                "suffix": ["T1w"],
                "run": ["01"],
                "path": ["anat_run01.nii.gz"],
            }
        )

    @pytest.fixture
    def func_group_run01(self) -> pl.DataFrame:
        """Func group for run 01, task rest."""
        return pl.DataFrame({"suffix": ["bold"], "run": ["01"], "task": ["rest"]})

    def test_runs_correspond_matches_by_run(
        self,
        func_group_run01: pl.DataFrame,
        anat_with_runs: pl.DataFrame,
        fallback_anat: pl.DataFrame,
    ) -> None:
        """When runs correspond, anat is matched by the func group's run label."""
        result = _resolve_anat(
            func_group_run01, anat_with_runs, fallback_anat, runs_correspond=True
        )
        assert result["run"].to_list() == ["01"]

    def test_runs_correspond_falls_back_on_no_match(
        self,
        anat_with_runs: pl.DataFrame,
        fallback_anat: pl.DataFrame,
    ) -> None:
        """Falls back to fallback_anat when the func run label has no anat match."""
        func_group = pl.DataFrame({"suffix": ["bold"], "run": ["99"], "task": ["rest"]})
        result = _resolve_anat(
            func_group, anat_with_runs, fallback_anat, runs_correspond=True
        )
        assert result.equals(fallback_anat)

    def test_runs_do_not_correspond_returns_fallback(
        self,
        func_group_run01: pl.DataFrame,
        anat_with_runs: pl.DataFrame,
        fallback_anat: pl.DataFrame,
    ) -> None:
        """When runs_correspond is False, fallback_anat is always returned."""
        result = _resolve_anat(
            func_group_run01, anat_with_runs, fallback_anat, runs_correspond=False
        )
        assert result.equals(fallback_anat)


class TestIterSessionFiles:
    """Tests for iter_session_files.

    Covers anat-only iteration, func-driven iteration, run correspondence
    matching, and fallback behaviour for mismatched run counts.
    """

    @pytest.fixture
    def anat_df(self) -> pl.DataFrame:
        """Single T1w anat file with no run label."""
        return pl.DataFrame(
            {
                "datatype": ["anat"],
                "suffix": ["T1w"],
                "ext": [".nii.gz"],
                "run": [None],
                "path": ["sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"],
            }
        )

    @pytest.fixture
    def func_df(self) -> pl.DataFrame:
        """Two func bold files for different tasks (rest and nback), no run label."""
        return pl.DataFrame(
            {
                "datatype": ["func", "func"],
                "suffix": ["bold", "bold"],
                "ext": [".nii.gz", ".nii.gz"],
                "run": [None, None],
                "task": ["rest", "nback"],
                "path": [
                    "sub-01/ses-baseline/func/sub-01_ses-baseline_task-rest_bold.nii.gz",
                    "sub-01/ses-baseline/func/sub-01_ses-baseline_task-nback_bold.nii.gz",
                ],
            }
        )

    @pytest.fixture
    def anat_df_with_runs(self) -> pl.DataFrame:
        """Anat dataframe with two run-labelled T1w files (run 01 and 02)."""
        return pl.DataFrame(
            {
                "datatype": ["anat", "anat"],
                "suffix": ["T1w", "T1w"],
                "ext": [".nii.gz", ".nii.gz"],
                "run": ["01", "02"],
                "path": ["anat_run01.nii.gz", "anat_run02.nii.gz"],
            }
        )

    @pytest.fixture
    def func_df_with_runs(self) -> pl.DataFrame:
        """Func dataframe with two run-labelled bold files (run 01 and 02)."""
        return pl.DataFrame(
            {
                "datatype": ["func", "func"],
                "suffix": ["bold", "bold"],
                "ext": [".nii.gz", ".nii.gz"],
                "run": ["01", "02"],
                "task": ["rest", "rest"],
                "path": ["func_run01.nii.gz", "func_run02.nii.gz"],
            }
        )

    def test_anat_only_yields_anat_groups(self, anat_df: pl.DataFrame) -> None:
        """With no func data, one group per anat group with anat as both elements."""
        session = SessionTables(anat=anat_df, func=None)
        results = list(iter_session_files(session, groupby=("run",)))
        assert len(results) == 1
        primary, anat = results[0]
        assert primary.equals(anat)

    def test_func_drives_iteration(
        self, anat_df: pl.DataFrame, func_df: pl.DataFrame
    ) -> None:
        """With func data matches number of func groups."""
        session = SessionTables(anat=anat_df, func=func_df)
        results = list(iter_session_files(session, groupby=("task",)))
        assert len(results) == 2

    def test_func_group_is_primary(
        self, anat_df: pl.DataFrame, func_df: pl.DataFrame
    ) -> None:
        """The primary element of each yielded tuple contains only func rows."""
        session = SessionTables(anat=anat_df, func=func_df)
        results = list(iter_session_files(session, groupby=("task",)))
        for primary, _ in results:
            assert primary["datatype"].unique().to_list() == ["func"]

    def test_anat_paired_with_each_func_group(
        self, anat_df: pl.DataFrame, func_df: pl.DataFrame
    ) -> None:
        """Each func group is paired with an anat dataframe."""
        session = SessionTables(anat=anat_df, func=func_df)
        results = list(iter_session_files(session, groupby=("task",)))
        for _, anat in results:
            assert anat["datatype"].unique().to_list() == ["anat"]

    def test_run_correspondence_matches_anat_by_run(
        self, anat_df_with_runs: pl.DataFrame, func_df_with_runs: pl.DataFrame
    ) -> None:
        """1-to-1 run correspondence yields matching anat run for each func group."""
        session = SessionTables(anat=anat_df_with_runs, func=func_df_with_runs)
        results = list(iter_session_files(session, groupby=("run",)))
        assert len(results) == 2
        for func_group, anat_subset in results:
            assert func_group["run"].to_list() == anat_subset["run"].to_list()

    def test_run_mismatch_uses_fallback_anat(
        self, anat_df_with_runs: pl.DataFrame
    ) -> None:
        """When func and anat run counts differ, use first anat run."""
        func_df = pl.DataFrame(
            {
                "datatype": ["func", "func", "func"],
                "suffix": ["bold"] * 3,
                "ext": [".nii.gz"] * 3,
                "run": ["01", "02", "03"],
                "task": ["rest"] * 3,
                "path": ["func_run01.nii.gz", "func_run02.nii.gz", "func_run03.nii.gz"],
            }
        )
        session = SessionTables(anat=anat_df_with_runs, func=func_df)
        results = list(iter_session_files(session, groupby=("run",)))
        first_run = anat_df_with_runs["run"].sort()[0]
        for _, anat_subset in results:
            assert anat_subset["run"].to_list() == [first_run]
