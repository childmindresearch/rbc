"""Tests for rbc.core.nifti — Volume abstraction and metadata queries."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from rbc.core.nifti import (
    Space,
    Units,
    Volume,
    nifti_num_slices,
    nifti_num_volumes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_nifti(
    tmp_path: Path,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: np.dtype | type = np.float64,
    sform_code: int = 4,
    qform_code: int = 4,
    xyzt_units: int = 2,  # mm
    tr: float = 2.0,
    affine: np.ndarray | None = None,
) -> Path:
    """Write a minimal NIfTI file with controlled metadata."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(dtype)
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    hdr = img.header
    hdr["sform_code"] = sform_code
    hdr["qform_code"] = qform_code
    hdr.set_sform(affine, code=sform_code)
    hdr.set_qform(affine, code=qform_code)
    hdr["xyzt_units"] = xyzt_units
    if len(shape) >= 4:
        pixdim = hdr["pixdim"].copy()
        pixdim[4] = tr
        hdr["pixdim"] = pixdim
    path = tmp_path / name
    img.to_filename(str(path))
    return path


@pytest.fixture
def nifti_4d(tmp_path: Path) -> Path:
    """4D NIfTI: (5, 6, 7, 10), sform=MNI, units=mm, TR=2.0."""
    return _make_nifti(tmp_path, "bold.nii.gz", (5, 6, 7, 10))


@pytest.fixture
def nifti_3d(tmp_path: Path) -> Path:
    """3D NIfTI: (5, 6, 7), sform=MNI, units=mm."""
    return _make_nifti(tmp_path, "anat.nii.gz", (5, 6, 7))


@pytest.fixture
def mask_3d(tmp_path: Path) -> Path:
    """3D binary mask matching the 4D fixture's spatial dims."""
    data = np.ones((5, 6, 7), dtype=np.uint8)
    img = nib.Nifti1Image(data, np.eye(4))
    hdr = img.header
    hdr["sform_code"] = 4
    hdr["qform_code"] = 4
    hdr["xyzt_units"] = 2
    path = tmp_path / "mask.nii.gz"
    img.to_filename(str(path))
    return path


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestSpace:
    def test_values_match_nifti_standard(self) -> None:
        assert Space.UNKNOWN == 0
        assert Space.SCANNER == 1
        assert Space.ALIGNED == 2
        assert Space.TALAIRACH == 3
        assert Space.MNI == 4


class TestUnits:
    def test_values(self) -> None:
        assert Units.UNKNOWN.value == "unknown"
        assert Units.MM.value == "mm"
        assert Units.M.value == "m"
        assert Units.MICRON.value == "um"


# ---------------------------------------------------------------------------
# Volume.load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_basic_load_4d(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float32)
        assert vol.data.shape == (5, 6, 7, 10)
        assert vol.data.dtype == np.float32
        assert vol.sform == Space.MNI
        assert vol.spatial_units == Units.MM
        assert vol.tr == pytest.approx(2.0)
        assert vol.source_path == nifti_4d

    def test_basic_load_3d(self, nifti_3d: Path) -> None:
        vol = Volume.load(nifti_3d, dtype=np.float64)
        assert vol.data.shape == (5, 6, 7)
        assert vol.tr is None

    def test_dtype_any(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        assert vol.data.dtype == np.float64  # nibabel default

    def test_dtype_uint8(self, mask_3d: Path) -> None:
        vol = Volume.load(mask_3d, dtype=np.uint8)
        assert vol.data.dtype == np.uint8

    def test_expected_ndim_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any", expected_ndim=4)
        assert vol.data.ndim == 4

    def test_expected_ndim_fail(self, nifti_4d: Path) -> None:
        with pytest.raises(ValueError, match="Expected 3D"):
            Volume.load(nifti_4d, dtype="any", expected_ndim=3)

    def test_source_path_stored(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        assert vol.source_path == nifti_4d


# ---------------------------------------------------------------------------
# Volume.from_array
# ---------------------------------------------------------------------------


class TestFromArray:
    def test_basic(self) -> None:
        data = np.zeros((3, 4, 5), dtype=np.float32)
        vol = Volume.from_array(data, np.eye(4))
        assert vol.spatial_shape == (3, 4, 5)
        assert vol.source_path is None

    def test_bad_affine_shape(self) -> None:
        with pytest.raises(ValueError, match="4x4"):
            Volume.from_array(np.zeros((3, 4, 5)), np.eye(3))

    def test_defaults(self) -> None:
        vol = Volume.from_array(np.zeros((3, 4, 5)), np.eye(4))
        assert vol.spatial_units == Units.MM
        assert vol.sform == Space.UNKNOWN
        assert vol.qform == Space.UNKNOWN

    def test_custom_metadata(self) -> None:
        vol = Volume.from_array(
            np.zeros((3, 4, 5)),
            np.eye(4),
            spatial_units=Units.M,
            sform=Space.MNI,
        )
        assert vol.spatial_units == Units.M
        assert vol.sform == Space.MNI
        assert vol.qform == Space.MNI


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_tr_4d(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        assert vol.tr == pytest.approx(2.0)

    def test_tr_3d_is_none(self, nifti_3d: Path) -> None:
        vol = Volume.load(nifti_3d, dtype="any")
        assert vol.tr is None

    def test_voxel_sizes_identity(self, nifti_3d: Path) -> None:
        vol = Volume.load(nifti_3d, dtype="any")
        assert vol.voxel_sizes == pytest.approx((1.0, 1.0, 1.0))

    def test_voxel_sizes_scaled(self, tmp_path: Path) -> None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        path = _make_nifti(tmp_path, "scaled.nii.gz", (3, 4, 5), affine=affine)
        vol = Volume.load(path, dtype="any")
        assert vol.voxel_sizes == pytest.approx((2.0, 3.0, 4.0))

    def test_voxel_sizes_rotated(self) -> None:
        """Voxel sizes are column norms, not just diagonal."""
        theta = np.pi / 4
        affine = np.eye(4)
        affine[0, 0] = np.cos(theta) * 2.0
        affine[1, 0] = np.sin(theta) * 2.0
        affine[0, 1] = -np.sin(theta) * 2.0
        affine[1, 1] = np.cos(theta) * 2.0
        affine[2, 2] = 3.0
        vol = Volume.from_array(np.zeros((3, 4, 5)), affine)
        assert vol.voxel_sizes == pytest.approx((2.0, 2.0, 3.0))

    def test_spatial_shape(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        assert vol.spatial_shape == (5, 6, 7)


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


class TestCheck:
    def test_ndim_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        assert vol.check(ndim=4) is vol  # chainable

    def test_ndim_fail(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="3D"):
            vol.check(ndim=3)

    def test_dtype_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float64)
        vol.check(dtype=np.float64)

    def test_dtype_fail(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            vol.check(dtype=np.int32)

    def test_spatial_units_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        vol.check(spatial_units=Units.MM)

    def test_spatial_units_fail(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="spatial_units"):
            vol.check(spatial_units=Units.M)

    def test_sform_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        vol.check(sform=Space.MNI)

    def test_sform_fail(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="sform"):
            vol.check(sform=Space.SCANNER)

    def test_qform(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        vol.check(qform=Space.MNI)
        with pytest.raises(ValueError, match="qform"):
            vol.check(qform=Space.SCANNER)

    def test_min_volumes_pass(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        vol.check(min_volumes=10)

    def test_min_volumes_fail(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="volumes"):
            vol.check(min_volumes=20)

    def test_error_includes_path(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="bold.nii.gz"):
            vol.check(ndim=3)

    def test_chainable(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float64)
        result = vol.check(ndim=4, sform=Space.MNI, spatial_units=Units.MM)
        assert result is vol


# ---------------------------------------------------------------------------
# check_compatible()
# ---------------------------------------------------------------------------


class TestCheckCompatible:
    def test_matching_grids(self, nifti_4d: Path, mask_3d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        mask = Volume.load(mask_3d, dtype="any")
        bold.check_compatible(mask)  # should not raise

    def test_shape_mismatch(self, nifti_4d: Path, tmp_path: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        other_path = _make_nifti(tmp_path, "other.nii.gz", (8, 9, 10))
        other = Volume.load(other_path, dtype="any")
        with pytest.raises(ValueError, match="Spatial shape"):
            bold.check_compatible(other)

    def test_affine_mismatch(self, nifti_4d: Path, tmp_path: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        shifted_affine = np.eye(4)
        shifted_affine[0, 3] = 100.0
        path = _make_nifti(
            tmp_path, "shifted.nii.gz", (5, 6, 7), affine=shifted_affine
        )
        other = Volume.load(path, dtype="any")
        with pytest.raises(ValueError, match="Affine"):
            bold.check_compatible(other)

    def test_atol(self, tmp_path: Path) -> None:
        aff1 = np.eye(4)
        aff2 = np.eye(4)
        aff2[0, 3] = 1e-5  # tiny offset
        p1 = _make_nifti(tmp_path, "a.nii.gz", (3, 4, 5), affine=aff1)
        p2 = _make_nifti(tmp_path, "b.nii.gz", (3, 4, 5), affine=aff2)
        v1 = Volume.load(p1, dtype="any")
        v2 = Volume.load(p2, dtype="any")
        v1.check_compatible(v2, atol=1e-4)  # passes
        with pytest.raises(ValueError, match="Affine"):
            v1.check_compatible(v2, atol=1e-6)

    def test_sform_mismatch_default(self, tmp_path: Path) -> None:
        p1 = _make_nifti(tmp_path, "mni.nii.gz", (3, 4, 5), sform_code=4)
        p2 = _make_nifti(tmp_path, "scan.nii.gz", (3, 4, 5), sform_code=1)
        v1 = Volume.load(p1, dtype="any")
        v2 = Volume.load(p2, dtype="any")
        with pytest.raises(ValueError, match="sform"):
            v1.check_compatible(v2)

    def test_sform_mismatch_disabled(self, tmp_path: Path) -> None:
        p1 = _make_nifti(tmp_path, "mni.nii.gz", (3, 4, 5), sform_code=4)
        p2 = _make_nifti(tmp_path, "scan.nii.gz", (3, 4, 5), sform_code=1)
        v1 = Volume.load(p1, dtype="any")
        v2 = Volume.load(p2, dtype="any")
        v1.check_compatible(v2, check_sform=False)  # should not raise

    def test_units_mismatch_default(self, tmp_path: Path) -> None:
        p1 = _make_nifti(tmp_path, "mm.nii.gz", (3, 4, 5), xyzt_units=2)
        p2 = _make_nifti(tmp_path, "m.nii.gz", (3, 4, 5), xyzt_units=1)
        v1 = Volume.load(p1, dtype="any")
        v2 = Volume.load(p2, dtype="any")
        with pytest.raises(ValueError, match="units"):
            v1.check_compatible(v2)

    def test_units_mismatch_disabled(self, tmp_path: Path) -> None:
        p1 = _make_nifti(tmp_path, "mm.nii.gz", (3, 4, 5), xyzt_units=2)
        p2 = _make_nifti(tmp_path, "m.nii.gz", (3, 4, 5), xyzt_units=1)
        v1 = Volume.load(p1, dtype="any")
        v2 = Volume.load(p2, dtype="any")
        v1.check_compatible(v2, check_units=False)  # should not raise


# ---------------------------------------------------------------------------
# derive()
# ---------------------------------------------------------------------------


class TestDerive:
    def test_4d_to_3d(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype=np.float64)
        map_3d = np.zeros((5, 6, 7), dtype=np.float64)
        derived = bold.derive(map_3d)
        assert derived.data.shape == (5, 6, 7)
        assert derived.tr is None
        assert derived.sform == Space.MNI
        assert derived.spatial_units == Units.MM
        np.testing.assert_array_equal(derived.affine, bold.affine)

    def test_4d_to_4d(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype=np.float64)
        new_data = np.zeros((5, 6, 7, 10), dtype=np.float32)
        derived = bold.derive(new_data)
        assert derived.data.shape == (5, 6, 7, 10)
        assert derived.tr == pytest.approx(2.0)

    def test_3d_to_3d(self, nifti_3d: Path) -> None:
        vol = Volume.load(nifti_3d, dtype=np.float64)
        new_data = np.ones((5, 6, 7), dtype=np.float32)
        derived = vol.derive(new_data)
        assert derived.data.shape == (5, 6, 7)

    def test_3d_to_4d_rejected(self, nifti_3d: Path) -> None:
        vol = Volume.load(nifti_3d, dtype=np.float64)
        with pytest.raises(ValueError, match="3D source"):
            vol.derive(np.zeros((5, 6, 7, 4)))

    def test_spatial_shape_mismatch(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="Spatial shape"):
            bold.derive(np.zeros((8, 9, 10)))

    def test_too_few_dims(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        with pytest.raises(ValueError, match="3D"):
            bold.derive(np.zeros((5, 6)))

    def test_metadata_preserved(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype="any")
        derived = bold.derive(np.zeros((5, 6, 7)))
        assert derived.sform == bold.sform
        assert derived.qform == bold.qform
        assert derived.spatial_units == bold.spatial_units

    def test_derived_dtype_preserved(self, nifti_4d: Path) -> None:
        bold = Volume.load(nifti_4d, dtype=np.float64)
        int_data = np.zeros((5, 6, 7), dtype=np.int16)
        derived = bold.derive(int_data)
        assert derived.data.dtype == np.int16


# ---------------------------------------------------------------------------
# replace()
# ---------------------------------------------------------------------------


class TestReplace:
    def test_sform_override(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(sform=Space.SCANNER)
        assert replaced.sform == Space.SCANNER
        assert replaced.qform == Space.MNI  # unchanged

    def test_units_override(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(spatial_units=Units.M)
        assert replaced.spatial_units == Units.M

    def test_data_shared(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(sform=Space.SCANNER)
        assert replaced.data is vol.data

    def test_unspecified_preserved(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(sform=Space.SCANNER)
        assert replaced.tr == vol.tr
        assert replaced.spatial_units == vol.spatial_units
        np.testing.assert_array_equal(replaced.affine, vol.affine)

    def test_tr_override(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(tr=3.0)
        assert replaced.tr == pytest.approx(3.0)

    def test_tr_clear(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        replaced = vol.replace(tr=None)
        assert replaced.tr is None


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_roundtrip(self, nifti_4d: Path, tmp_path: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float64)
        out = tmp_path / "roundtrip.nii.gz"
        vol.save(out)
        reloaded = Volume.load(out, dtype=np.float64)
        np.testing.assert_array_almost_equal(reloaded.data, vol.data)
        np.testing.assert_array_almost_equal(reloaded.affine, vol.affine)
        assert reloaded.sform == vol.sform
        assert reloaded.spatial_units == vol.spatial_units

    def test_3d_output_has_3d_header(self, nifti_4d: Path, tmp_path: Path) -> None:
        """THE critical bug test: 4D->3D derivation must produce correct shape."""
        bold = Volume.load(nifti_4d, dtype=np.float64)
        map_3d = np.zeros((5, 6, 7), dtype=np.float64)
        out = tmp_path / "map3d.nii.gz"
        bold.derive(map_3d).save(out)

        img = nib.nifti1.load(out)
        assert img.shape == (5, 6, 7)
        assert len(img.shape) == 3

    def test_nan_warn(self, tmp_path: Path) -> None:
        data = np.array([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        vol = Volume.from_array(data, np.eye(4))
        out = tmp_path / "nan.nii.gz"
        with pytest.warns(match="NaN"):
            vol.save(out, nan="warn")

    def test_nan_raise(self, tmp_path: Path) -> None:
        data = np.array([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        vol = Volume.from_array(data, np.eye(4))
        out = tmp_path / "nan.nii.gz"
        with pytest.raises(ValueError, match="NaN"):
            vol.save(out, nan="raise")

    def test_nan_ignore(self, tmp_path: Path) -> None:
        data = np.array([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        vol = Volume.from_array(data, np.eye(4))
        out = tmp_path / "nan.nii.gz"
        vol.save(out, nan="ignore")  # should not raise or warn
        assert out.exists()

    def test_dtype_preserved(self, tmp_path: Path) -> None:
        data = np.ones((3, 4, 5), dtype=np.float32)
        vol = Volume.from_array(data, np.eye(4))
        out = tmp_path / "f32.nii.gz"
        vol.save(out)
        reloaded = nib.nifti1.load(out)
        assert reloaded.get_data_dtype() == np.float32

    def test_returns_path(self, tmp_path: Path) -> None:
        vol = Volume.from_array(np.zeros((3, 4, 5)), np.eye(4))
        out = tmp_path / "out.nii.gz"
        result = vol.save(out)
        assert result == out

    def test_sform_qform_written(self, tmp_path: Path) -> None:
        vol = Volume.from_array(
            np.zeros((3, 4, 5)), np.eye(4), sform=Space.MNI
        )
        out = tmp_path / "sform.nii.gz"
        vol.save(out)
        img = nib.nifti1.load(out)
        assert int(img.header["sform_code"]) == 4
        assert int(img.header["qform_code"]) == 4

    def test_tr_written(self, nifti_4d: Path, tmp_path: Path) -> None:
        vol = Volume.load(nifti_4d, dtype="any")
        out = tmp_path / "tr.nii.gz"
        vol.save(out)
        img = nib.nifti1.load(out)
        assert float(img.header["pixdim"][4]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_loaded(self, nifti_4d: Path) -> None:
        vol = Volume.load(nifti_4d, dtype=np.float32)
        r = repr(vol)
        assert "5, 6, 7, 10" in r
        assert "float32" in r
        assert "bold.nii.gz" in r

    def test_repr_memory(self) -> None:
        vol = Volume.from_array(np.zeros((3, 4, 5)), np.eye(4))
        assert "memory" in repr(vol)


# ---------------------------------------------------------------------------
# Existing functions (regression)
# ---------------------------------------------------------------------------


class TestExistingFunctions:
    def test_nifti_num_volumes_4d(self, nifti_4d: Path) -> None:
        assert nifti_num_volumes(nifti_4d) == 10

    def test_nifti_num_volumes_3d(self, nifti_3d: Path) -> None:
        assert nifti_num_volumes(nifti_3d) == 1

    def test_nifti_num_slices_3d(self, nifti_3d: Path) -> None:
        assert nifti_num_slices(nifti_3d) == 7
