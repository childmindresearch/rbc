"""Unit tests for longitudinal modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbc.core.longitudinal.transform import anat_transform


@pytest.fixture
def tmp_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create temporary input files for testing."""
    in_file = tmp_path / "brain.nii.gz"
    template = tmp_path / "template.nii.gz"
    xfm = tmp_path / "subj_to_template.mat"
    in_file.touch()
    template.touch()
    xfm.touch()
    return in_file, template, xfm


class TestAnatomicalLongitudinalTranforms:
    """Test suite for transformation between anatomical and longitudinal templates."""

    def test_missing_in_file(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if input file to transform does not exist."""
        in_file, template, xfm = tmp_files
        in_file.unlink()
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            anat_transform(in_file=in_file, template=template, xfm=xfm)

    def test_missing_xfm(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if transformation does not exist."""
        in_file, template, xfm = tmp_files
        xfm.unlink()
        with pytest.raises(FileNotFoundError, match="Transformation not found"):
            anat_transform(in_file=in_file, template=template, xfm=xfm)

    @patch("rbc.core.longitudinal.transform.ants")
    def test_returns_output_path(
        self, mock_ants: MagicMock, tmp_files: tuple[Path, ...]
    ) -> None:
        """Successful transformation to template."""
        in_file, template, xfm = tmp_files
        expected = Path("/out/subject_to_template.nii.gz")

        mock_ants.ants_apply_transforms.return_value.output.output_image_outfile = (
            expected
        )
        mock_ants.ants_apply_transforms_warped_output.return_value = MagicMock()
        mock_ants.ants_apply_transforms_linear.return_value = MagicMock()

        result = anat_transform(in_file=in_file, template=template, xfm=xfm)

        assert result == expected
