"""Unit tests for BOLD resampling utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.common import merge_3d_to_4d
from rbc.core.functional.resampling import (
    apply_motion_transforms,
    resample_bold_to_template,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_identity_warp(
    path: Path, shape: tuple[int, int, int], affine: np.ndarray
) -> Path:
    """Write a zero-displacement ANTs/ITK warp at the given grid (5D format)."""
    warp = np.zeros((*shape, 1, 3), dtype=np.float32)
    img = nib.Nifti1Image(warp, affine)
    img.header.set_intent(1007)  # NIFTI_INTENT_VECTOR
    nib.save(img, path)
    return path


def _write_fsl_mat(path: Path, mat: np.ndarray) -> Path:
    """Write a 4x4 matrix to disk in FSL .mat (whitespace-delimited) format."""
    np.savetxt(path, mat)
    return path


def _make_bold(
    path: Path, shape: tuple[int, int, int], n_vols: int, affine: np.ndarray, tr: float
) -> Path:
    """Create a 4D BOLD NIfTI with reproducible random voxel values."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((*shape, n_vols)).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((*affine[:3, :3].diagonal().tolist(), tr))
    nib.save(img, path)
    return path


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


class TestApplyMotionTransforms:
    """Tests for apply_motion_transforms (resamples in BOLD native space)."""

    def test_identity_motion_is_roundtrip(self, tmp_path: Path) -> None:
        """Identity motion mats reproduce the input STC data within interp tolerance."""
        shape = (8, 9, 7)
        n_vols = 3
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bold_path = _make_bold(tmp_path / "stc.nii.gz", shape, n_vols, affine, tr=1.5)

        ref_path = tmp_path / "bold_ref.nii.gz"
        src_data = np.asanyarray(nib.nifti1.load(bold_path).dataobj)
        nib.save(nib.Nifti1Image(src_data[..., 1], affine), ref_path)

        mat_dir = tmp_path / "mats"
        mat_dir.mkdir()
        for i in range(n_vols):
            _write_fsl_mat(mat_dir / f"MAT_{i:04d}", np.eye(4))

        out = apply_motion_transforms(
            stc_img=bold_path, motion_mat_dir=mat_dir, bold_ref=ref_path
        )
        out_img = nib.nifti1.load(out)
        assert out_img.shape == (*shape, n_vols)
        np.testing.assert_allclose(out_img.get_fdata(), src_data, atol=1e-3)

    def test_preserves_tr(self, tmp_path: Path) -> None:
        """Output keeps the source BOLD's TR in pixdim[4]."""
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bold_path = _make_bold(tmp_path / "stc.nii.gz", (4, 5, 4), 2, affine, tr=2.7)
        ref_path = tmp_path / "bold_ref.nii.gz"
        ref_data = np.zeros((4, 5, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(ref_data, affine), ref_path)
        mat_dir = tmp_path / "mats"
        mat_dir.mkdir()
        for i in range(2):
            _write_fsl_mat(mat_dir / f"MAT_{i:04d}", np.eye(4))

        out = apply_motion_transforms(
            stc_img=bold_path, motion_mat_dir=mat_dir, bold_ref=ref_path
        )
        assert nib.nifti1.load(out).header.get_zooms()[3] == pytest.approx(2.7)

    def test_missing_mats_raises(self, tmp_path: Path) -> None:
        """Empty motion directory raises FileNotFoundError."""
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bold_path = _make_bold(tmp_path / "stc.nii.gz", (4, 5, 4), 2, affine, tr=1.0)
        ref_path = tmp_path / "bold_ref.nii.gz"
        ref_data = np.zeros((4, 5, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(ref_data, affine), ref_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match=r"No motion \.mat files"):
            apply_motion_transforms(
                stc_img=bold_path, motion_mat_dir=empty, bold_ref=ref_path
            )

    def test_mat_count_mismatch_raises(self, tmp_path: Path) -> None:
        """Mismatched motion-mat / volume counts raise ValueError."""
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bold_path = _make_bold(tmp_path / "stc.nii.gz", (4, 5, 4), 3, affine, tr=1.0)
        ref_path = tmp_path / "bold_ref.nii.gz"
        ref_data = np.zeros((4, 5, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(ref_data, affine), ref_path)
        mat_dir = tmp_path / "mats"
        mat_dir.mkdir()
        for i in range(2):  # one too few
            _write_fsl_mat(mat_dir / f"MAT_{i:04d}", np.eye(4))
        with pytest.raises(ValueError, match="Count mismatch"):
            apply_motion_transforms(
                stc_img=bold_path, motion_mat_dir=mat_dir, bold_ref=ref_path
            )


class TestResampleBoldToTemplate:
    """Tests for resample_bold_to_template (single-step BOLD->template)."""

    def _set_up(
        self, tmp_path: Path, n_vols: int = 3, tr: float = 2.0
    ) -> dict[str, Path]:
        """Build a minimal synthetic input set."""
        bold_shape = (8, 9, 7)
        bold_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bold_path = _make_bold(
            tmp_path / "stc.nii.gz", bold_shape, n_vols, bold_affine, tr=tr
        )

        ref_path = tmp_path / "bold_ref.nii.gz"
        src_data = np.asanyarray(nib.nifti1.load(bold_path).dataobj)
        nib.save(nib.Nifti1Image(src_data[..., 0], bold_affine), ref_path)

        t1w_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        t1w_path = tmp_path / "t1w.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((10, 12, 9), dtype=np.float32), t1w_affine),
            t1w_path,
        )

        tpl_shape = (6, 7, 5)
        tpl_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        tpl_path = tmp_path / "tpl.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros(tpl_shape, dtype=np.float32), tpl_affine),
            tpl_path,
        )

        bold2anat = _write_fsl_mat(tmp_path / "bold2anat.mat", np.eye(4))
        anat2tpl = _write_identity_warp(
            tmp_path / "anat2tpl.nii.gz", tpl_shape, tpl_affine
        )

        mat_dir = tmp_path / "mats"
        mat_dir.mkdir()
        for i in range(n_vols):
            _write_fsl_mat(mat_dir / f"MAT_{i:04d}", np.eye(4))

        return {
            "stc_bold": bold_path,
            "motion_mat_dir": mat_dir,
            "bold_to_anat": bold2anat,
            "anat_to_template": anat2tpl,
            "bold_ref": ref_path,
            "template": tpl_path,
            "t1w_brain": t1w_path,
        }

    def test_output_shape_and_voxel_size_match_template(self, tmp_path: Path) -> None:
        """Output spatial grid matches the template, time dim matches BOLD."""
        kwargs = self._set_up(tmp_path, n_vols=4)
        out = resample_bold_to_template(**kwargs)
        out_img = nib.nifti1.load(out)
        tpl_img = nib.nifti1.load(kwargs["template"])
        assert out_img.shape == (*tpl_img.shape, 4)
        np.testing.assert_array_equal(out_img.affine, tpl_img.affine)
        assert out_img.header.get_zooms()[:3] == tpl_img.header.get_zooms()[:3]

    def test_preserves_tr_from_source(self, tmp_path: Path) -> None:
        """Output pixdim[4] equals the BOLD source's TR, not the template's."""
        kwargs = self._set_up(tmp_path, tr=2.7)
        out = resample_bold_to_template(**kwargs)
        assert nib.nifti1.load(out).header.get_zooms()[3] == pytest.approx(2.7)

    def test_output_is_finite(self, tmp_path: Path) -> None:
        """Resampled output contains no NaN or Inf voxels."""
        kwargs = self._set_up(tmp_path)
        out = resample_bold_to_template(**kwargs)
        data = nib.nifti1.load(out).get_fdata()
        assert np.isfinite(data).all()

    def test_distortion_warp_accepted(self, tmp_path: Path) -> None:
        """Optional distortion warp is composed in without crashing."""
        # A zero-displacement warp does NOT exactly reproduce the no-distortion
        # result: DenseFieldTransform.map cubic-spline-samples the stored RAS
        # coord field, which rings near the grid boundary. So we only check
        # shape + finiteness here; numerical agreement is in integration.
        kwargs = self._set_up(tmp_path)
        bold_affine = nib.nifti1.load(kwargs["bold_ref"]).affine
        distortion = _write_identity_warp(
            tmp_path / "distortion.nii.gz", (8, 9, 7), bold_affine
        )
        out = resample_bold_to_template(distortion_warp=distortion, **kwargs)
        out_img = nib.nifti1.load(out)
        tpl_img = nib.nifti1.load(kwargs["template"])
        assert out_img.shape == (*tpl_img.shape, 3)
        assert np.isfinite(out_img.get_fdata()).all()

    def test_missing_mats_raises(self, tmp_path: Path) -> None:
        """Empty motion directory raises FileNotFoundError."""
        kwargs = self._set_up(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        kwargs["motion_mat_dir"] = empty
        with pytest.raises(FileNotFoundError, match=r"No motion \.mat files"):
            resample_bold_to_template(**kwargs)

    def test_mat_count_mismatch_raises(self, tmp_path: Path) -> None:
        """Mismatched motion-mat / volume counts raise ValueError."""
        kwargs = self._set_up(tmp_path, n_vols=3)
        # Drop one mat so the count no longer matches.
        for m in sorted(kwargs["motion_mat_dir"].glob("MAT_*"))[-1:]:
            m.unlink()
        with pytest.raises(ValueError, match="Count mismatch"):
            resample_bold_to_template(**kwargs)
