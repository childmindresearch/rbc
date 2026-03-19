"""Unit tests for longitudinal modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbc.core.longitudinal.transform import (
    anat_transform,
    compose_transform,
    func_transform,
)


@pytest.fixture
def tmp_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create temporary input files for testing."""
    in_file = tmp_path / "brain.nii.gz"
    template = tmp_path / "template.nii.gz"
    xfm = tmp_path / "subj_to_template.txt"
    in_file.touch()
    template.touch()
    xfm.touch()
    return in_file, template, xfm


@pytest.fixture
def tmp_compose_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create temporary input files for compose transform testing."""
    ref = tmp_path / "ref.nii.gz"
    bold_to_anat = tmp_path / "bold_to_anat.txt"
    anat_to_tpl = tmp_path / "anat_to_tpl.txt"
    ref.touch()
    bold_to_anat.touch()
    anat_to_tpl.touch()
    return ref, bold_to_anat, anat_to_tpl


class TestAnatomicalLongitudinalTransforms:
    """Test suite for transformation between anatomical and longitudinal templates."""

    def test_missing_in_file(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if input file to transform does not exist."""
        in_file, template, xfm = tmp_files
        in_file.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            anat_transform(in_file=in_file, template=template, xfm=xfm)

    def test_missing_xfm(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if transformation does not exist."""
        in_file, template, xfm = tmp_files
        xfm.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
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


class TestFunctionalLongitudinalTransforms:
    """Test suite for transformation between functional and longitudinal templates."""

    def test_missing_in_file(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if input file to transform does not exist."""
        in_file, template, xfm = tmp_files
        in_file.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            func_transform(in_file=in_file, template=template, xfm=xfm)

    def test_missing_xfm(self, tmp_files: tuple[Path, ...]) -> None:
        """Raises error if transformation does not exist."""
        in_file, template, xfm = tmp_files
        xfm.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            func_transform(in_file=in_file, template=template, xfm=xfm)

    @patch("rbc.core.longitudinal.transform.ants")
    @patch("rbc.core.longitudinal.transform.split_4d")
    @patch("rbc.core.longitudinal.transform.merge_3d_to_4d")
    @patch("rbc.core.longitudinal.transform._restore_tr")
    @pytest.mark.parametrize("strategy", ["chunked", "single"])
    def test_returns_output_path(
        self,
        mock_restore_tr: MagicMock,
        mock_merge_3d_to_4d: MagicMock,
        mock_split_4d: MagicMock,
        mock_ants: MagicMock,
        tmp_files: tuple[Path, ...],
        strategy: str,
    ) -> None:
        """Successful functional transformation to template."""
        in_file, template, xfm = tmp_files
        expected = Path("/out/subj_bold_to_template.nii.gz")

        mock_ants.ants_apply_transforms.return_value.output.output_image_outfile = (
            expected
        )
        mock_ants.ants_apply_transforms_warped_output.return_value = MagicMock()
        mock_ants.ants_apply_transforms_linear.return_value = MagicMock()
        mock_ants.ants_apply_transforms_transform_file_name.return_value = MagicMock()

        if strategy == "chunked":
            mock_split_4d.return_value = [in_file]
            mock_merge_3d_to_4d.return_value = expected
            mock_restore_tr.return_value = None

        result = func_transform(
            in_file=in_file,
            template=template,
            xfm=xfm,
            strategy=strategy,  # type: ignore [arg-type]
        )
        assert result == expected


class TestComposeTransform:
    """Test suite for composing bold-to-template transformations."""

    def test_missing_ref(self, tmp_compose_files: tuple[Path, ...]) -> None:
        """Raises error if reference image does not exist."""
        ref, bold_to_anat, anat_to_tpl = tmp_compose_files
        ref.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            compose_transform(
                ref=ref, bold_to_anat_xfm=bold_to_anat, anat_to_tpl_xfm=anat_to_tpl
            )

    def test_missing_bold_to_anat_xfm(
        self, tmp_compose_files: tuple[Path, ...]
    ) -> None:
        """Raises error if bold-to-anatomical transformation does not exist."""
        ref, bold_to_anat, anat_to_tpl = tmp_compose_files
        bold_to_anat.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            compose_transform(
                ref=ref, bold_to_anat_xfm=bold_to_anat, anat_to_tpl_xfm=anat_to_tpl
            )

    def test_missing_anat_to_tpl_xfm(self, tmp_compose_files: tuple[Path, ...]) -> None:
        """Raises error if anatomical-to-template transformation does not exist."""
        ref, bold_to_anat, anat_to_tpl = tmp_compose_files
        anat_to_tpl.unlink()
        with pytest.raises(FileNotFoundError, match="not found"):
            compose_transform(
                ref=ref, bold_to_anat_xfm=bold_to_anat, anat_to_tpl_xfm=anat_to_tpl
            )

    @patch("rbc.core.longitudinal.transform.ants")
    def test_returns_output_path(
        self, mock_ants: MagicMock, tmp_compose_files: tuple[Path, ...]
    ) -> None:
        """Successful composition of bold-to-template warp field."""
        ref, bold_to_anat, anat_to_tpl = tmp_compose_files
        expected = Path("/out/bold_to_tpl_xfm.nii.gz")

        mock_ants.ants_apply_transforms.return_value.output.output_image_outfile = (
            expected
        )
        # Renamed for ruff
        comp_out = mock_ants.ants_apply_transforms_composite_displacement_field_output
        comp_out.return_value = MagicMock()
        mock_ants.ants_apply_transforms_transform_file_name.return_value = MagicMock()

        result = compose_transform(
            ref=ref, bold_to_anat_xfm=bold_to_anat, anat_to_tpl_xfm=anat_to_tpl
        )
        assert result == expected
