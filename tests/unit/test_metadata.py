"""Unit tests for rbc.metadata."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from rbc.metadata import FunctionalMetadata, _resolve_tr

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveTr:
    """Tests for TR resolution logic."""

    def test_override_wins(self) -> None:
        """CLI override takes priority over sidecar and header."""
        tr = _resolve_tr(sidecar_tr=2.0, header_tr=2.0, override=1.5)
        assert tr == 1.5

    def test_override_warns_on_sidecar_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CLI override logs warning when sidecar disagrees."""
        _resolve_tr(sidecar_tr=2.0, header_tr=2.0, override=1.5)
        assert any("differs from sidecar" in msg for msg in caplog.messages)

    def test_override_warns_on_header_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CLI override logs warning when header disagrees."""
        _resolve_tr(sidecar_tr=1.5, header_tr=2.0, override=1.5)
        assert any("differs from NIfTI header" in msg for msg in caplog.messages)

    def test_sidecar_used_when_no_override(self) -> None:
        """Sidecar TR is used when no override is provided."""
        tr = _resolve_tr(sidecar_tr=2.0, header_tr=2.0, override=None)
        assert tr == 2.0

    def test_sidecar_header_mismatch_raises(self) -> None:
        """Sidecar/header mismatch without override raises ValueError."""
        with pytest.raises(ValueError, match="TR mismatch"):
            _resolve_tr(sidecar_tr=2.0, header_tr=1.5, override=None)

    def test_sidecar_header_within_tolerance(self) -> None:
        """Sidecar/header difference within tolerance is OK."""
        tr = _resolve_tr(sidecar_tr=2.0, header_tr=2.0005, override=None)
        assert tr == 2.0

    def test_header_fallback_when_no_sidecar(self) -> None:
        """Header TR is used when sidecar has no RepetitionTime."""
        tr = _resolve_tr(sidecar_tr=None, header_tr=2.0, override=None)
        assert tr == 2.0

    def test_header_fallback_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Header fallback logs a warning."""
        _resolve_tr(sidecar_tr=None, header_tr=2.0, override=None)
        assert any("falling back to NIfTI header" in msg for msg in caplog.messages)

    def test_no_tr_anywhere_raises(self) -> None:
        """Missing TR from all sources raises ValueError."""
        with pytest.raises(ValueError, match="Cannot determine TR"):
            _resolve_tr(sidecar_tr=None, header_tr=None, override=None)

    def test_zero_header_treated_as_missing(self) -> None:
        """Zero header TR is treated as missing."""
        with pytest.raises(ValueError, match="Cannot determine TR"):
            _resolve_tr(sidecar_tr=None, header_tr=0.0, override=None)

    def test_zero_sidecar_treated_as_missing(self) -> None:
        """Zero sidecar TR falls through to header."""
        tr = _resolve_tr(sidecar_tr=0.0, header_tr=2.0, override=None)
        assert tr == 2.0

    def test_negative_sidecar_treated_as_missing(self) -> None:
        """Negative sidecar TR falls through to header."""
        tr = _resolve_tr(sidecar_tr=-1.0, header_tr=2.0, override=None)
        assert tr == 2.0


class TestFunctionalMetadata:
    """Tests for FunctionalMetadata dataclass."""

    def test_frozen(self) -> None:
        """FunctionalMetadata instances are immutable."""
        meta = FunctionalMetadata(tr=2.0, slice_timing=None)
        with pytest.raises(FrozenInstanceError):
            meta.tr = 1.0  # type: ignore[misc]

    def test_load_sidecar_tr(self, tmp_path: Path) -> None:
        """load() reads TR from sidecar and slice timing."""
        bold = tmp_path / "bold.nii.gz"
        bold.touch()

        mock_hdr = MagicMock()
        mock_hdr.__getitem__ = lambda _, key: (
            [0, 1, 1, 1, 2.0] if key == "pixdim" else None
        )
        mock_img = MagicMock()
        mock_img.header = mock_hdr

        with (
            patch(
                "rbc.metadata.load_bids_metadata",
                return_value={"RepetitionTime": 2.0, "SliceTiming": [0.0, 0.5, 1.0]},
            ),
            patch("rbc.metadata.nib.nifti1.load", return_value=mock_img),
        ):
            meta = FunctionalMetadata.load(bold)

        assert meta.tr == pytest.approx(2.0)
        assert meta.slice_timing == [0.0, 0.5, 1.0]

    def test_load_with_override(self, tmp_path: Path) -> None:
        """load() uses CLI override when provided."""
        bold = tmp_path / "bold.nii.gz"
        bold.touch()

        mock_hdr = MagicMock()
        mock_hdr.__getitem__ = lambda _, key: (
            [0, 1, 1, 1, 2.0] if key == "pixdim" else None
        )
        mock_img = MagicMock()
        mock_img.header = mock_hdr

        with (
            patch(
                "rbc.metadata.load_bids_metadata",
                return_value={"RepetitionTime": 2.0},
            ),
            patch("rbc.metadata.nib.nifti1.load", return_value=mock_img),
        ):
            meta = FunctionalMetadata.load(bold, tr_override=1.5)

        assert meta.tr == pytest.approx(1.5)

    def test_load_no_slice_timing(self, tmp_path: Path) -> None:
        """load() sets slice_timing to None when absent from sidecar."""
        bold = tmp_path / "bold.nii.gz"
        bold.touch()

        mock_hdr = MagicMock()
        mock_hdr.__getitem__ = lambda _, key: (
            [0, 1, 1, 1, 2.0] if key == "pixdim" else None
        )
        mock_img = MagicMock()
        mock_img.header = mock_hdr

        with (
            patch(
                "rbc.metadata.load_bids_metadata",
                return_value={"RepetitionTime": 2.0},
            ),
            patch("rbc.metadata.nib.nifti1.load", return_value=mock_img),
        ):
            meta = FunctionalMetadata.load(bold)

        assert meta.slice_timing is None

    def test_load_missing_tr_raises(self, tmp_path: Path) -> None:
        """load() raises when TR cannot be determined."""
        bold = tmp_path / "bold.nii.gz"
        bold.touch()

        mock_hdr = MagicMock()
        mock_hdr.__getitem__ = lambda _, key: (
            [0, 1, 1, 1, 0.0] if key == "pixdim" else None
        )
        mock_img = MagicMock()
        mock_img.header = mock_hdr

        with (
            patch("rbc.metadata.load_bids_metadata", return_value={}),
            patch("rbc.metadata.nib.nifti1.load", return_value=mock_img),
            pytest.raises(ValueError, match="Cannot determine TR"),
        ):
            FunctionalMetadata.load(bold)
