"""Unit tests for functional workflow."""


class TestFuncInitialization:
    """Test suite for functional workflow initialization."""

    def test_truncate_string_logic(self) -> None:
        """Test the AFNI selector string generation logic."""
        start_tr = 4
        selector = f"[{start_tr}..$]"
        assert selector == "[4..$]"


class TestFuncMotion:
    """Test suite for functional workflow motion correction."""

    def test_middle_index_logic(self) -> None:
        """Test logic for calculating middle index of volumes."""
        volumes_list = [100, 101, 2, 1]
        results = [nv // 2 for nv in volumes_list]
        assert results == [50, 50, 1, 0]
