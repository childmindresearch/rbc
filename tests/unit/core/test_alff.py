"""Unit tests for rbc.core.metrics.alff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.metrics.alff import (
    alff,
    am_alff,
    compute_alff,
    compute_frequency_bins,
    qm_alff,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------
TR = 2.0  # seconds
N = 100  # timepoints → freq resolution 0.005 Hz, Nyquist 0.25 Hz
F_LOW = 0.01
F_HIGH = 0.1
SHAPE = (3, 3, 3, N)

# Frequencies chosen to land exactly on FFT bins (no spectral leakage):
#   0.05 Hz → bin 10 (in-band)
#   0.20 Hz → bin 40 (out-of-band)
FREQ_IN = 0.05
FREQ_OUT = 0.20


def _make_sine(freq: float, n: int = N, tr: float = TR) -> np.ndarray:
    """Pure sine at *freq* Hz sampled at 1/tr for *n* points."""
    t = np.arange(n) * tr
    return np.sin(2 * np.pi * freq * t)


def _broadcast_to_4d(
    ts: np.ndarray, shape: tuple[int, int, int, int] = SHAPE
) -> np.ndarray:
    """Tile a 1-D timeseries into a 4-D volume."""
    return np.tile(ts, (*shape[:3], 1))


# ===================================================================
# compute_frequency_bins
# ===================================================================
class TestComputeFrequencyBins:
    """Tests for FFT bin selection."""

    def test_known_bins_tr2_n100(self) -> None:
        """With TR=2, N=100 the default band [0.01, 0.1] should cover bins 2-20."""
        bins = compute_frequency_bins(100, 2.0, 0.01, 0.1)
        assert bins[0] == 2
        assert bins[-1] == 20

    def test_single_bin_band(self) -> None:
        """A band containing exactly one frequency should return one bin."""
        # bin 10 → 0.05 Hz; band [0.049, 0.051] should select only bin 10
        bins = compute_frequency_bins(100, 2.0, 0.049, 0.051)
        assert len(bins) == 1
        assert bins[0] == 10

    def test_raises_empty_band(self) -> None:
        """Band above Nyquist should raise."""
        with pytest.raises(ValueError, match="No FFT bins"):
            compute_frequency_bins(100, 2.0, 0.3, 0.4)

    def test_raises_inverted_bounds(self) -> None:
        """f_low >= f_high should raise."""
        with pytest.raises(ValueError, match="f_low must be < f_high"):
            compute_frequency_bins(100, 2.0, 0.1, 0.01)

    def test_raises_bad_tr(self) -> None:
        """Non-positive TR should raise."""
        with pytest.raises(ValueError, match="TR must be positive"):
            compute_frequency_bins(100, 0.0, 0.01, 0.1)
        with pytest.raises(ValueError, match="TR must be positive"):
            compute_frequency_bins(100, -1.0, 0.01, 0.1)

    def test_odd_n(self) -> None:
        """Odd number of timepoints should still work."""
        bins = compute_frequency_bins(101, 2.0, 0.01, 0.1)
        assert len(bins) > 0

    def test_bins_are_sorted(self) -> None:
        """Returned bins should be sorted."""
        bins = compute_frequency_bins(100, 2.0, 0.01, 0.1)
        assert np.all(np.diff(bins) > 0)


# ===================================================================
# am_alff
# ===================================================================
class TestAmAlff:
    """Tests for amplitude-mean ALFF (Zang 2007)."""

    def test_output_shapes(self) -> None:
        """Both maps should be 3-D float64."""
        data = np.random.default_rng(0).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, fa = am_alff(data, mask, TR)
        assert a.shape == SHAPE[:3]
        assert fa.shape == SHAPE[:3]
        assert a.dtype == np.float64
        assert fa.dtype == np.float64

    def test_respects_mask(self) -> None:
        """Out-of-mask voxels should be zero."""
        data = np.random.default_rng(1).standard_normal(SHAPE)
        mask = np.zeros(SHAPE[:3])
        mask[1, 1, 1] = 1
        a, fa = am_alff(data, mask, TR)
        assert a[0, 0, 0] == 0.0
        assert fa[0, 0, 0] == 0.0
        assert a[1, 1, 1] > 0.0

    def test_single_inband_sine_falff_near_one(self) -> None:
        """Pure in-band sine → fALFF ≈ 1 (all energy in band)."""
        ts = _make_sine(FREQ_IN)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = am_alff(data, mask, TR)
        np.testing.assert_allclose(fa[1, 1, 1], 1.0, atol=0.01)

    def test_outband_sine_falff_near_zero(self) -> None:
        """Pure out-of-band sine → fALFF ≈ 0."""
        ts = _make_sine(FREQ_OUT)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = am_alff(data, mask, TR)
        np.testing.assert_allclose(fa[1, 1, 1], 0.0, atol=0.01)

    def test_mixed_signal_falff_intermediate(self) -> None:
        """Equal-amplitude in-band + out-of-band → fALFF ≈ 0.5."""
        ts = _make_sine(FREQ_IN) + _make_sine(FREQ_OUT)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = am_alff(data, mask, TR)
        assert 0.3 < fa[1, 1, 1] < 0.7

    def test_zero_variance_voxel(self) -> None:
        """Constant timeseries → ALFF=0, fALFF=0."""
        data = np.ones(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, fa = am_alff(data, mask, TR)
        assert a[1, 1, 1] == 0.0
        assert fa[1, 1, 1] == 0.0

    def test_alff_positive(self) -> None:
        """ALFF should be non-negative everywhere."""
        data = np.random.default_rng(2).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, _ = am_alff(data, mask, TR)
        assert np.all(a >= 0)

    def test_falff_bounded(self) -> None:
        """Fractional ALFF should be in [0, 1]."""
        data = np.random.default_rng(3).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        _, fa = am_alff(data, mask, TR)
        assert np.all(fa >= 0)
        assert np.all(fa <= 1.0 + 1e-10)

    def test_deterministic(self) -> None:
        """Same input → identical output."""
        data = np.random.default_rng(4).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a1, fa1 = am_alff(data, mask, TR)
        a2, fa2 = am_alff(data, mask, TR)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(fa1, fa2)

    def test_rejects_non_4d(self) -> None:
        """Non-4-D data should raise."""
        with pytest.raises(ValueError, match="4D"):
            am_alff(np.zeros((5, 5, 5)), np.ones((5, 5, 5)), TR)

    def test_empty_mask(self) -> None:
        """Empty mask → all-zero output."""
        data = np.random.default_rng(5).standard_normal(SHAPE)
        mask = np.zeros(SHAPE[:3])
        a, fa = am_alff(data, mask, TR)
        assert np.all(a == 0)
        assert np.all(fa == 0)

    def test_float_mask(self) -> None:
        """Float mask > 0 treated as True."""
        data = np.random.default_rng(6).standard_normal(SHAPE)
        bool_mask = np.ones(SHAPE[:3])
        float_mask = np.ones(SHAPE[:3]) * 0.7
        a1, fa1 = am_alff(data, bool_mask, TR)
        a2, fa2 = am_alff(data, float_mask, TR)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(fa1, fa2)


# ===================================================================
# qm_alff
# ===================================================================
class TestQmAlff:
    """Tests for quadratic-mean (std-deviation) ALFF (C-PAC style)."""

    def test_output_shapes(self) -> None:
        """Both maps should be 3-D float64."""
        data = np.random.default_rng(10).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, fa = qm_alff(data, mask, TR)
        assert a.shape == SHAPE[:3]
        assert fa.shape == SHAPE[:3]
        assert a.dtype == np.float64
        assert fa.dtype == np.float64

    def test_respects_mask(self) -> None:
        """Out-of-mask voxels should be zero."""
        data = np.random.default_rng(11).standard_normal(SHAPE)
        mask = np.zeros(SHAPE[:3])
        mask[1, 1, 1] = 1
        a, fa = qm_alff(data, mask, TR)
        assert a[0, 0, 0] == 0.0
        assert fa[0, 0, 0] == 0.0
        assert a[1, 1, 1] > 0.0

    def test_single_inband_sine_alff(self) -> None:
        """Pure in-band sine → ALFF ≈ 1/sqrt(2) (std of sin)."""
        ts = _make_sine(FREQ_IN)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        a, _ = qm_alff(data, mask, TR)
        expected_std = np.std(ts, ddof=0)
        np.testing.assert_allclose(a[1, 1, 1], expected_std, atol=0.01)

    def test_single_inband_sine_falff_near_one(self) -> None:
        """Pure in-band sine → fALFF ≈ 1."""
        ts = _make_sine(FREQ_IN)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = qm_alff(data, mask, TR)
        np.testing.assert_allclose(fa[1, 1, 1], 1.0, atol=0.01)

    def test_outband_sine_alff_near_zero(self) -> None:
        """Pure out-of-band sine → ALFF ≈ 0."""
        ts = _make_sine(FREQ_OUT)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        a, _ = qm_alff(data, mask, TR)
        np.testing.assert_allclose(a[1, 1, 1], 0.0, atol=0.01)

    def test_outband_sine_falff_near_zero(self) -> None:
        """Pure out-of-band sine → fALFF ≈ 0."""
        ts = _make_sine(FREQ_OUT)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = qm_alff(data, mask, TR)
        np.testing.assert_allclose(fa[1, 1, 1], 0.0, atol=0.01)

    def test_mixed_signal_falff(self) -> None:
        """Equal in-band + out-of-band sines → fALFF ≈ 1/sqrt(2)."""
        ts = _make_sine(FREQ_IN) + _make_sine(FREQ_OUT)
        data = _broadcast_to_4d(ts)
        mask = np.ones(SHAPE[:3])
        _, fa = qm_alff(data, mask, TR)
        # std(sin) / std(sin+sin) = 1/sqrt(2) when the two sines are
        # orthogonal (non-overlapping in frequency).
        np.testing.assert_allclose(fa[1, 1, 1], 1.0 / np.sqrt(2), atol=0.02)

    def test_zero_variance_voxel(self) -> None:
        """Constant timeseries → ALFF=0, fALFF=0."""
        data = np.ones(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, fa = qm_alff(data, mask, TR)
        assert a[1, 1, 1] == 0.0
        assert fa[1, 1, 1] == 0.0

    def test_alff_positive(self) -> None:
        """ALFF should be non-negative."""
        data = np.random.default_rng(12).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a, _ = qm_alff(data, mask, TR)
        assert np.all(a >= 0)

    def test_falff_bounded(self) -> None:
        """Fractional ALFF should be in [0, 1]."""
        data = np.random.default_rng(13).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        _, fa = qm_alff(data, mask, TR)
        assert np.all(fa >= 0)
        assert np.all(fa <= 1.0 + 1e-10)

    def test_deterministic(self) -> None:
        """Same input → identical output."""
        data = np.random.default_rng(14).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        a1, fa1 = qm_alff(data, mask, TR)
        a2, fa2 = qm_alff(data, mask, TR)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(fa1, fa2)

    def test_rejects_non_4d(self) -> None:
        """Non-4-D data should raise."""
        with pytest.raises(ValueError, match="4D"):
            qm_alff(np.zeros((5, 5, 5)), np.ones((5, 5, 5)), TR)

    def test_empty_mask(self) -> None:
        """Empty mask → all-zero output."""
        data = np.random.default_rng(15).standard_normal(SHAPE)
        mask = np.zeros(SHAPE[:3])
        a, fa = qm_alff(data, mask, TR)
        assert np.all(a == 0)
        assert np.all(fa == 0)

    def test_float_mask(self) -> None:
        """Float mask > 0 treated as True."""
        data = np.random.default_rng(16).standard_normal(SHAPE)
        bool_mask = np.ones(SHAPE[:3])
        float_mask = np.ones(SHAPE[:3]) * 0.7
        a1, fa1 = qm_alff(data, bool_mask, TR)
        a2, fa2 = qm_alff(data, float_mask, TR)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(fa1, fa2)


# ===================================================================
# alff dispatcher
# ===================================================================
class TestAlffDispatcher:
    """Tests for the top-level alff() dispatcher."""

    @pytest.mark.parametrize("method", ["am", "qm"])
    def test_dispatch_matches_direct(self, method: str) -> None:
        """Dispatcher should produce identical results to direct calls."""
        data = np.random.default_rng(20).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])

        a_dispatch, fa_dispatch = alff(data, mask, TR, method=method)  # type: ignore[arg-type]

        if method == "am":
            a_direct, fa_direct = am_alff(data, mask, TR)
        else:
            a_direct, fa_direct = qm_alff(data, mask, TR)

        np.testing.assert_array_equal(a_dispatch, a_direct)
        np.testing.assert_array_equal(fa_dispatch, fa_direct)

    def test_rejects_unknown_method(self) -> None:
        """Unknown method string should raise."""
        data = np.random.default_rng(21).standard_normal(SHAPE)
        mask = np.ones(SHAPE[:3])
        with pytest.raises(ValueError, match="Unknown method"):
            alff(data, mask, TR, method="bad")  # type: ignore[arg-type]


# ===================================================================
# compute_alff (file I/O wrapper)
# ===================================================================


def _write_nifti(
    path: Path,
    data: np.ndarray,
    *,
    tr: float = TR,
    sform_code: int = 4,
) -> Path:
    """Write a minimal NIfTI with controlled metadata."""
    img = nib.Nifti1Image(data, np.eye(4))
    hdr = img.header
    hdr["sform_code"] = sform_code
    hdr["qform_code"] = sform_code
    hdr["xyzt_units"] = 2  # mm
    if data.ndim >= 4:
        pixdim = hdr["pixdim"].copy()
        pixdim[4] = tr
        hdr["pixdim"] = pixdim
    img.to_filename(str(path))
    return path


class TestComputeAlff:
    """Tests for compute_alff (Volume-based file I/O)."""

    def test_outputs_exist(self, tmp_path: Path) -> None:
        """compute_alff should write ALFF and fALFF files."""
        rng = np.random.default_rng(30)
        bold_path = _write_nifti(tmp_path / "bold.nii.gz", rng.standard_normal(SHAPE))
        mask_path = _write_nifti(
            tmp_path / "mask.nii.gz", np.ones(SHAPE[:3], dtype=np.uint8)
        )

        alff_path, falff_path = compute_alff(bold_path, mask_path)
        assert alff_path.exists()
        assert falff_path.exists()

    def test_output_is_3d(self, tmp_path: Path) -> None:
        """Output NIfTIs must have 3D shape (not 4D header from input)."""
        rng = np.random.default_rng(31)
        bold_path = _write_nifti(tmp_path / "bold.nii.gz", rng.standard_normal(SHAPE))
        mask_path = _write_nifti(
            tmp_path / "mask.nii.gz", np.ones(SHAPE[:3], dtype=np.uint8)
        )

        alff_path, falff_path = compute_alff(bold_path, mask_path)

        alff_img = nib.nifti1.load(alff_path)
        falff_img = nib.nifti1.load(falff_path)
        assert len(alff_img.shape) == 3
        assert alff_img.shape == SHAPE[:3]
        assert len(falff_img.shape) == 3
        assert falff_img.shape == SHAPE[:3]

    def test_preserves_sform(self, tmp_path: Path) -> None:
        """Output should preserve sform code from the input BOLD."""
        rng = np.random.default_rng(32)
        bold_path = _write_nifti(
            tmp_path / "bold.nii.gz", rng.standard_normal(SHAPE), sform_code=4
        )
        mask_path = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones(SHAPE[:3], dtype=np.uint8),
            sform_code=4,
        )

        alff_path, _ = compute_alff(bold_path, mask_path)
        img = nib.nifti1.load(alff_path)
        assert int(img.header["sform_code"]) == 4

    def test_tr_override(self, tmp_path: Path) -> None:
        """Explicit tr= should override the header value."""
        rng = np.random.default_rng(33)
        bold_path = _write_nifti(
            tmp_path / "bold.nii.gz", rng.standard_normal(SHAPE), tr=2.0
        )
        mask_path = _write_nifti(
            tmp_path / "mask.nii.gz", np.ones(SHAPE[:3], dtype=np.uint8)
        )

        # Header has TR=2.0. Override with TR=0.5 shifts freq resolution
        # from 0.005 Hz to 0.02 Hz, changing which bins fall in [0.01, 0.1].
        a1, _ = compute_alff(
            bold_path, mask_path, out_file=tmp_path / "a_alff.nii.gz"
        )
        a2, _ = compute_alff(
            bold_path, mask_path, tr=0.5, out_file=tmp_path / "b_alff.nii.gz"
        )

        d1 = nib.nifti1.load(a1).get_fdata()
        d2 = nib.nifti1.load(a2).get_fdata()
        assert not np.allclose(d1, d2)

    def test_matches_pure_function(self, tmp_path: Path) -> None:
        """File I/O wrapper should produce identical values to pure alff()."""
        rng = np.random.default_rng(34)
        data = rng.standard_normal(SHAPE)
        mask_data = np.ones(SHAPE[:3], dtype=np.uint8)

        bold_path = _write_nifti(tmp_path / "bold.nii.gz", data)
        mask_path = _write_nifti(tmp_path / "mask.nii.gz", mask_data)

        alff_path, falff_path = compute_alff(bold_path, mask_path)

        expected_a, expected_fa = am_alff(data, mask_data, TR)
        np.testing.assert_allclose(
            nib.nifti1.load(alff_path).get_fdata(), expected_a, atol=1e-10
        )
        np.testing.assert_allclose(
            nib.nifti1.load(falff_path).get_fdata(), expected_fa, atol=1e-10
        )
