"""Unit tests for rbc.core.qc.xcp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from rbc.core.qc.dvars import DVARSQCMetrics

if TYPE_CHECKING:
    from pathlib import Path
from rbc.core.qc.motion import MotionQCMetrics
from rbc.core.qc.registration import RegistrationQCMetrics
from rbc.core.qc.xcp import (
    XCPQCMetrics,
    generate_xcp_qc,
    passes_rbc_qc,
    write_xcp_qc,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "sub",
    "ses",
    "task",
    "run",
    "desc",
    "regressors",
    "space",
    "meanFD",
    "relMeansRMSMotion",
    "relMaxRMSMotion",
    "nVolCensored",
    "meanDVInit",
    "meanDVFinal",
    "nVolsRemoved",
    "motionDVCorrInit",
    "motionDVCorrFinal",
    "coregDice",
    "coregJaccard",
    "coregCrossCorr",
    "coregCoverage",
    "normDice",
    "normJaccard",
    "normCrossCorr",
    "normCoverage",
]


def _sample_motion() -> MotionQCMetrics:
    return MotionQCMetrics(
        mean_fd=0.15,
        rel_means_rms_motion=0.12,
        rel_max_rms_motion=0.45,
        n_vol_censored=3,
    )


def _sample_dvars_init() -> DVARSQCMetrics:
    return DVARSQCMetrics(mean_dvars=25.0, motion_dvars_corr=0.6)


def _sample_dvars_final() -> DVARSQCMetrics:
    return DVARSQCMetrics(mean_dvars=10.0, motion_dvars_corr=0.3)


def _sample_coreg() -> RegistrationQCMetrics:
    return RegistrationQCMetrics(
        dice=0.90,
        jaccard=0.82,
        cross_corr=0.88,
        coverage=0.95,
    )


def _sample_norm() -> RegistrationQCMetrics:
    return RegistrationQCMetrics(
        dice=0.85,
        jaccard=0.74,
        cross_corr=0.82,
        coverage=0.91,
    )


def _sample_xcp_metrics() -> XCPQCMetrics:
    return generate_xcp_qc(
        sub="01",
        ses="001",
        task="rest",
        run=1,
        desc="preproc",
        regressors="36P",
        space="MNI152NLin2009cAsym",
        motion=_sample_motion(),
        dvars_init=_sample_dvars_init(),
        dvars_final=_sample_dvars_final(),
        n_vols_removed=5,
        coreg=_sample_coreg(),
        norm=_sample_norm(),
    )


# ===================================================================
# XCPQCMetrics
# ===================================================================
class TestXCPQCMetrics:
    """Tests for XCPQCMetrics NamedTuple structure."""

    def test_field_names_match_pq_columns(self) -> None:
        """NamedTuple field names exactly match expected Parquet columns."""
        assert list(XCPQCMetrics._fields) == EXPECTED_COLUMNS

    def test_field_count(self) -> None:
        """Exactly 24 fields."""
        assert len(XCPQCMetrics._fields) == 24

    def test_construction(self) -> None:
        """Can construct from positional args."""
        m = XCPQCMetrics(
            "01",
            "001",
            "rest",
            1,
            "preproc",
            "36P",
            "MNI",
            0.1,
            0.2,
            0.3,
            4,
            20.0,
            10.0,
            2,
            0.5,
            0.3,
            0.9,
            0.8,
            0.85,
            0.95,
            0.88,
            0.77,
            0.82,
            0.91,
        )
        assert m.sub == "01"
        assert m.run == 1
        assert m.normCoverage == 0.91


# ===================================================================
# generate_xcp_qc
# ===================================================================
class TestGenerateXcpQc:
    """Tests for the generate_xcp_qc assembly function."""

    def test_bids_entities(self) -> None:
        """BIDS entity fields are passed through correctly."""
        m = _sample_xcp_metrics()
        assert m.sub == "01"
        assert m.ses == "001"
        assert m.task == "rest"
        assert m.run == 1
        assert m.desc == "preproc"
        assert m.regressors == "36P"
        assert m.space == "MNI152NLin2009cAsym"

    def test_motion_fields(self) -> None:
        """Motion sub-metrics are mapped correctly."""
        m = _sample_xcp_metrics()
        motion = _sample_motion()
        assert m.meanFD == motion.mean_fd
        assert m.relMeansRMSMotion == motion.rel_means_rms_motion
        assert m.relMaxRMSMotion == motion.rel_max_rms_motion
        assert m.nVolCensored == motion.n_vol_censored

    def test_dvars_fields(self) -> None:
        """DVARS init/final sub-metrics are mapped correctly."""
        m = _sample_xcp_metrics()
        dvars_init = _sample_dvars_init()
        dvars_final = _sample_dvars_final()
        assert m.meanDVInit == dvars_init.mean_dvars
        assert m.meanDVFinal == dvars_final.mean_dvars
        assert m.motionDVCorrInit == dvars_init.motion_dvars_corr
        assert m.motionDVCorrFinal == dvars_final.motion_dvars_corr

    def test_n_vols_removed(self) -> None:
        """Field nVolsRemoved is passed through."""
        m = _sample_xcp_metrics()
        assert m.nVolsRemoved == 5

    def test_coreg_fields(self) -> None:
        """Coregistration sub-metrics are mapped correctly."""
        m = _sample_xcp_metrics()
        coreg = _sample_coreg()
        assert m.coregDice == coreg.dice
        assert m.coregJaccard == coreg.jaccard
        assert m.coregCrossCorr == coreg.cross_corr
        assert m.coregCoverage == coreg.coverage

    def test_norm_fields(self) -> None:
        """Normalization sub-metrics are mapped correctly."""
        m = _sample_xcp_metrics()
        norm = _sample_norm()
        assert m.normDice == norm.dice
        assert m.normJaccard == norm.jaccard
        assert m.normCrossCorr == norm.cross_corr
        assert m.normCoverage == norm.coverage

    def test_returns_named_tuple(self) -> None:
        """Result is an XCPQCMetrics instance."""
        m = _sample_xcp_metrics()
        assert isinstance(m, XCPQCMetrics)


# ===================================================================
# write_xcp_qc
# ===================================================================
class TestWriteXcpQc:
    """Tests for writing XCP QC metrics to Parquet."""

    def test_writes_file(self, tmp_path: Path) -> None:
        """Output file is created."""
        out = tmp_path / "qc.parquet"
        result = write_xcp_qc(_sample_xcp_metrics(), out)
        assert result == out
        assert out.exists()

    def test_correct_headers(self, tmp_path: Path) -> None:
        """Parquet column names match expected column names."""
        out = tmp_path / "qc.parquet"
        write_xcp_qc(_sample_xcp_metrics(), out)
        df = pl.read_parquet(out)
        assert df.columns == EXPECTED_COLUMNS

    def test_correct_values(self, tmp_path: Path) -> None:
        """Values in Parquet match the input metrics."""
        m = _sample_xcp_metrics()
        out = tmp_path / "qc.parquet"
        write_xcp_qc(m, out)
        df = pl.read_parquet(out)
        assert df["sub"][0] == "01"  # sub
        assert df["run"][0] == 1  # run
        assert df["meanFD"][0] == m.meanFD

    def test_round_trip_polars(self, tmp_path: Path) -> None:
        """Polars can read back the Parquet and recover the values."""
        m = _sample_xcp_metrics()
        out = tmp_path / "qc.parquet"
        write_xcp_qc(m, out)
        df = pl.read_parquet(out)
        assert df.shape == (1, 24)
        assert df["sub"][0] == "01"
        assert df["meanFD"][0] == m.meanFD
        assert df["normCoverage"][0] == m.normCoverage

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        out = tmp_path / "a" / "b" / "c" / "qc.parquet"
        write_xcp_qc(_sample_xcp_metrics(), out)
        assert out.exists()

    def test_single_data_row(self, tmp_path: Path) -> None:
        """Parquet has exactly one data row."""
        out = tmp_path / "qc.parquet"
        write_xcp_qc(_sample_xcp_metrics(), out)
        df = pl.read_parquet(out)
        assert df.shape[0] == 1


# ===================================================================
# passes_rbc_qc
# ===================================================================
class TestPassesRbcQc:
    """Tests for the RBC QC threshold check."""

    def test_passes_when_both_thresholds_met(self) -> None:
        """Low FD and high normCrossCorr → passes."""
        fd = np.array([0.0, 0.1, 0.15, 0.1, 0.05])
        assert passes_rbc_qc(fd, norm_cross_corr=0.9) is True

    def test_fails_when_fd_too_high(self) -> None:
        """High median FD → fails."""
        fd = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        assert passes_rbc_qc(fd, norm_cross_corr=0.9) is False

    def test_fails_when_norm_cross_corr_too_low(self) -> None:
        """Low normCrossCorr → fails."""
        fd = np.array([0.0, 0.1, 0.1, 0.05, 0.05])
        assert passes_rbc_qc(fd, norm_cross_corr=0.5) is False

    def test_fails_when_both_bad(self) -> None:
        """Both metrics out of range → fails."""
        fd = np.array([0.5, 0.6, 0.7])
        assert passes_rbc_qc(fd, norm_cross_corr=0.3) is False

    def test_edge_fd_at_threshold(self) -> None:
        """Median FD exactly 0.2 → passes (<=)."""
        fd = np.array([0.2, 0.2, 0.2])
        assert passes_rbc_qc(fd, norm_cross_corr=0.9) is True

    def test_edge_norm_at_threshold(self) -> None:
        """Exactly 0.8 normCrossCorr passes (>=)."""
        fd = np.array([0.0, 0.1, 0.1])
        assert passes_rbc_qc(fd, norm_cross_corr=0.8) is True

    def test_edge_fd_just_above_threshold(self) -> None:
        """Median FD just above 0.2 → fails."""
        fd = np.array([0.21, 0.21, 0.21])
        assert passes_rbc_qc(fd, norm_cross_corr=0.9) is False

    def test_edge_norm_just_below_threshold(self) -> None:
        """Value of normCrossCorr just below 0.8 fails."""
        fd = np.array([0.1, 0.1, 0.1])
        assert passes_rbc_qc(fd, norm_cross_corr=0.79) is False
