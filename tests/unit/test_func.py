"""Unit tests for functional workflow."""

import shutil
from types import SimpleNamespace

import pytest
from niwrap import afni

from rbc.core import functional

DOCKER_MISSING = shutil.which("docker") is None


class TestFuncInitialization:
    """Test suite for functional workflow initialization."""

    def test_truncate_string_logic(self) -> None:
        """Test the AFNI selector string generation logic."""
        start_tr = 4
        selector = f"[{start_tr}..$]"
        assert selector == "[4..$]"

    @pytest.mark.skipif(DOCKER_MISSING, reason="Docker is required")
    def test_truncate_to_min_volume(self, test_subject: SimpleNamespace) -> None:
        """Test truncating to minimum volume count of 1."""
        original_info = afni.v_3dinfo(dataset=[test_subject.bold], nv=True)
        original_count = int(original_info.info[0])

        start_tr = original_count - 1
        truncated_bold = functional.truncate_trs(
            in_file=test_subject.bold,
            output_fname="test_truncated_min.nii.gz",
            start_tr=start_tr,
        )
        # Test truncated BOLD file volume count is 1
        new_info = afni.v_3dinfo(dataset=[truncated_bold.output_file], nv=True)
        assert int(new_info.info[0]) == 1


class TestFuncMotion:
    """Test suite for functional workflow motion correction."""

    def test_middle_index_logic(self) -> None:
        """Test logic for calculating middle index of volumes."""
        volumes_list = [100, 101, 2, 1]
        results = [nv // 2 for nv in volumes_list]
        assert results == [50, 50, 1, 0]

    @pytest.mark.skipif(DOCKER_MISSING, reason="Docker is required")
    def test_motion_reference_volume_count(self, test_subject: SimpleNamespace) -> None:
        """Test motion reference volume count is 1."""
        reference = functional.generate_motion_reference(
            in_file=test_subject.bold, output_fname="test_motion_ref.nii.gz"
        )
        # Test motion reference file has 1 volume
        ref_info = afni.v_3dinfo(dataset=[reference.output_file], nv=True)
        assert int(ref_info.info[0]) == 1
