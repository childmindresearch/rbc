"""Unit tests for rbc.context."""

from __future__ import annotations

import pytest

from rbc.core.bids import bids_safe_label


class TestBidsSafeLabel:
    """Tests for bids_safe_label."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("36-parameter", "36parameter"),
            ("aCompCor", "aCompCor"),
            ("preproc", "preproc"),
            ("hello world", "helloworld"),
            ("a_b-c+d", "abc+d"),
            ("MNI152", "MNI152"),
        ],
    )
    def test_strips_invalid_chars(self, raw: str, expected: str) -> None:
        """Verify that non-BIDS characters are removed."""
        assert bids_safe_label(raw) == expected
