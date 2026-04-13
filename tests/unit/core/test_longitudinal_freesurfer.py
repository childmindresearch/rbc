"""Unit tests for ``rbc.core.longitudinal.freesurfer``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbc.core.longitudinal.freesurfer import (
    fs_to_itk_xfm,
    generate_robust_template,
    itk_filename,
    lta_filename,
    template_filename,
)


class TestFilenameMapping:
    """Filename helpers should follow the BIDS xfm naming convention."""

    def test_lta_filename(self) -> None:
        """LTA filename matches the BIDS xfm convention."""
        assert (
            lta_filename("01", "baseline")
            == "sub-01_ses-baseline_from-baseline_to-longitudinal_mode-image_xfm.lta"
        )

    def test_itk_filename(self) -> None:
        """ITK filename matches the BIDS xfm convention."""
        assert (
            itk_filename("01", "baseline")
            == "sub-01_ses-baseline_from-baseline_to-longitudinal_mode-image_xfm.txt"
        )

    def test_template_filename(self) -> None:
        """Template filename uses ses-longitudinal."""
        assert template_filename("01") == "sub-01_ses-longitudinal_T1w.nii.gz"


class TestGenerateRobustTemplate:
    """Tests for :func:`generate_robust_template`."""

    @pytest.fixture
    def two_volumes(self, tmp_path: Path) -> list[Path]:
        """Two empty input volumes for happy-path tests."""
        files = [
            tmp_path / "sub-01_ses-baseline_T1w.nii.gz",
            tmp_path / "sub-01_ses-vis2_T1w.nii.gz",
        ]
        for f in files:
            f.touch()
        return files

    def test_single_volume_raises(self, tmp_path: Path) -> None:
        """Per-subject single-session case must raise (bug #19 regression)."""
        single = tmp_path / "sub-01_ses-baseline_T1w.nii.gz"
        single.touch()
        with pytest.raises(ValueError, match="At least 2 input volumes"):
            generate_robust_template(sub="01", sessions=["baseline"], in_files=[single])

    def test_session_file_length_mismatch(self, two_volumes: list[Path]) -> None:
        """Sessions and in_files must match in length."""
        with pytest.raises(ValueError, match="must have the same length"):
            generate_robust_template(
                sub="01", sessions=["baseline"], in_files=two_volumes
            )

    def test_missing_input_file(self, two_volumes: list[Path]) -> None:
        """Missing inputs raise FileNotFoundError before invoking FreeSurfer."""
        two_volumes[0].unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            generate_robust_template(
                sub="01",
                sessions=["baseline", "vis2"],
                in_files=two_volumes,
            )

    @patch("rbc.core.longitudinal.freesurfer.freesurfer")
    def test_invokes_mri_robust_template(
        self, mock_fs: MagicMock, two_volumes: list[Path]
    ) -> None:
        """mri_robust_template is called with fmriprep-parity defaults."""
        mock_fs.mri_robust_template.return_value.template_output = Path(
            "/work/sub-01_ses-longitudinal_T1w.nii.gz"
        )
        mock_fs.mri_robust_template.return_value.root = Path("/work")

        result = generate_robust_template(
            sub="01",
            sessions=["baseline", "vis2"],
            in_files=two_volumes,
        )

        call_kwargs = mock_fs.mri_robust_template.call_args.kwargs
        assert call_kwargs["template"] == "sub-01_ses-longitudinal_T1w.nii.gz"
        assert call_kwargs["lta"] == [
            lta_filename("01", "baseline"),
            lta_filename("01", "vis2"),
        ]
        # Defaults from fmriprep parity.
        assert call_kwargs["inittp"] == 1
        assert call_kwargs["fixtp"] is True
        assert call_kwargs["iscale"] is True
        assert call_kwargs["noit"] is True
        assert call_kwargs["satit"] is True
        assert call_kwargs["subsample"] == 200

        assert result.template == Path("/work/sub-01_ses-longitudinal_T1w.nii.gz")
        assert result.transforms == [
            Path("/work") / lta_filename("01", "baseline"),
            Path("/work") / lta_filename("01", "vis2"),
        ]


class TestFsToItkXfm:
    """Tests for :func:`fs_to_itk_xfm`."""

    @patch("rbc.core.longitudinal.freesurfer.mat_to_itk")
    @patch("rbc.core.longitudinal.freesurfer.freesurfer")
    def test_per_session_itk_naming(
        self,
        mock_fs: MagicMock,
        mock_mat_to_itk: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Per-session ITK files are named with from-<ses>_to-longitudinal."""
        sources = [tmp_path / "src1.nii.gz", tmp_path / "src2.nii.gz"]
        in_xfms = [
            tmp_path / lta_filename("01", "baseline"),
            tmp_path / lta_filename("01", "vis2"),
        ]
        ref = tmp_path / "tpl.nii.gz"

        mock_fs.lta_convert.return_value.root = tmp_path
        mock_mat_to_itk.side_effect = lambda **kwargs: kwargs["output"]

        result = fs_to_itk_xfm(
            sub="01",
            sessions=["baseline", "vis2"],
            reference=ref,
            sources=sources,
            in_xfms=in_xfms,
        )

        assert result == [
            tmp_path / itk_filename("01", "baseline"),
            tmp_path / itk_filename("01", "vis2"),
        ]
        assert mock_fs.lta_convert.call_count == 2
        assert mock_mat_to_itk.call_count == 2

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        """Sessions, sources, and in_xfms must agree in length."""
        with pytest.raises(ValueError, match="same length"):
            fs_to_itk_xfm(
                sub="01",
                sessions=["baseline"],
                reference=tmp_path / "tpl.nii.gz",
                sources=[tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"],
                in_xfms=[tmp_path / "a.lta", tmp_path / "b.lta"],
            )
