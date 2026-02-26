"""Unit tests for BOLD resampling utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.functional.resampling import merge_3d_to_4d

if TYPE_CHECKING:
    from pathlib import Path


def _make_3d_nifti(path: Path, shape: tuple = (4, 5, 6)) -> Path:
    """Create a minimal 3D NIfTI image with random data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, path)
    return path


class TestMerge3dTo4d:
    """Tests for merge_3d_to_4d."""

    def test_produces_correct_4d_shape(self, tmp_path: Path) -> None:
        """Merging N 3D volumes produces a 4D image with shape (*spatial, N)."""
        n_vols = 5
        shape = (4, 5, 6)
        vols = [
            _make_3d_nifti(tmp_path / f"vol_{i}.nii.gz", shape) for i in range(n_vols)
        ]
        out = merge_3d_to_4d(vols, tmp_path / "merged.nii.gz")
        merged = nib.nifti1.load(out)
        assert merged.shape == (*shape, n_vols)

    def test_preserves_voxel_data(self, tmp_path: Path) -> None:
        """Merged 4D image contains the same voxel data as the input volumes."""
        shape = (3, 4, 5)
        vols = [_make_3d_nifti(tmp_path / f"vol_{i}.nii.gz", shape) for i in range(3)]
        out = merge_3d_to_4d(vols, tmp_path / "merged.nii.gz")
        merged_data = np.asanyarray(nib.nifti1.load(out).dataobj)
        for i, vol_path in enumerate(vols):
            vol_data = np.asanyarray(nib.nifti1.load(vol_path).dataobj)
            np.testing.assert_array_equal(merged_data[..., i], vol_data)

    def test_preserves_affine(self, tmp_path: Path) -> None:
        """Merged image retains the affine of the first input volume."""
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        vol_path = tmp_path / "vol_0.nii.gz"
        data = np.zeros((3, 3, 3), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, affine=affine), vol_path)
        out = merge_3d_to_4d([vol_path], tmp_path / "merged.nii.gz")
        np.testing.assert_array_equal(nib.nifti1.load(out).affine, affine)

    def test_single_volume(self, tmp_path: Path) -> None:
        """Merging a single 3D volume produces a 4D image with last dim = 1."""
        shape = (3, 4, 5)
        vols = [_make_3d_nifti(tmp_path / "vol_0.nii.gz", shape)]
        out = merge_3d_to_4d(vols, tmp_path / "merged.nii.gz")
        assert nib.nifti1.load(out).shape == (*shape, 1)

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        """Passing an empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            merge_3d_to_4d([], tmp_path / "merged.nii.gz")
