"""Unit tests for rbc.core.fsl2itk."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.fsl2itk import _fsl_swap_matrix, _fsl_to_ras, mat_to_itk

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Inlined test data from a real c3d_affine_tool run.
#
# Reference image: masked_ref_bold.nii.gz (64x64x33, radiological convention)
# The FSL MAT files come from MCFLIRT motion correction where ref == src.
# Expected ITK outputs were produced by c3d_affine_tool -fsl2ras -oitk.
# ---------------------------------------------------------------------------

_REF_AFFINE = np.array(
    [
        [-3.125, -0.0, -0.0, 97.93151855],
        [-0.0, 3.125, -0.0, -68.01921082],
        [0.0, 0.0, 4.0, -106.6505661],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_REF_SHAPE = (64, 64, 33)

_CASES: list[tuple[str, np.ndarray, np.ndarray]] = [
    # (label, FSL 4x4 matrix, expected ITK 12-parameter vector)
    (
        "MAT_0000",
        np.array(
            [
                [1.000000, 0.000000, 0.000000, 0.006818],
                [0.000000, 0.999999, -0.001456, 0.068589],
                [0.000000, 0.001456, 0.999999, -0.064242],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        np.array(
            [
                1,
                0,
                0,
                0,
                0.9999988800631344,
                -0.0014559998253717492,
                0,
                0.0014559998253717492,
                0.9999988800631344,
                -0.00681799999999555,
                -0.086711641553336,
                -0.03481360734485861,
            ]
        ),
    ),
    (
        "MAT_0100",
        np.array(
            [
                [1.000000, 0.000000, 0.000525, -0.048494],
                [-0.000001, 0.999997, 0.002279, -0.147670],
                [-0.000525, -0.002279, 0.999997, 0.056052],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        np.array(
            [
                0.9999997243744843,
                0.0000011964756347597939,
                -0.0005249987035247431,
                1.9647810423647357e-7,
                0.9999978061484665,
                0.002279001734066553,
                0.0005250009825259522,
                -0.002279001209066274,
                0.9999975305233202,
                -0.007576547509401621,
                0.09542788142147174,
                0.1504767739889843,
            ]
        ),
    ),
    (
        "MAT_0180",
        np.array(
            [
                [1.000000, -0.000756, -0.000499, 0.088891],
                [0.000757, 0.999998, 0.001896, -0.138337],
                [0.000498, -0.001896, 0.999998, -0.083145],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        np.array(
            [
                0.9999991792056033,
                -0.0007569442774028026,
                0.0004975654192044786,
                0.00075605396377283,
                0.9999978328881628,
                0.0018963769548377889,
                -0.0004994340684278406,
                -0.001895622724151258,
                0.9999981566780283,
                0.015727160807514906,
                0.13819236145373281,
                0.16328331032126187,
            ]
        ),
    ),
]


def _make_ref_image() -> nib.Nifti1Image:
    """Create a minimal NIfTI matching the real reference image geometry."""
    return nib.Nifti1Image(np.zeros(_REF_SHAPE, dtype=np.float32), _REF_AFFINE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "fsl_mat", "expected_params"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_mat_to_itk_matches_c3d(
    label: str,
    fsl_mat: np.ndarray,
    expected_params: np.ndarray,
    tmp_path: Path,
) -> None:
    """Pure-Python mat_to_itk matches c3d_affine_tool output."""
    mat_file = tmp_path / f"{label}.mat"
    np.savetxt(mat_file, fsl_mat)

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(_make_ref_image(), ref_path)

    output = tmp_path / f"{label}.txt"
    result = mat_to_itk(mat_file, ref_path, ref_path, str(output))

    assert result == output
    assert output.exists()

    # Parse the written parameters
    for line in output.read_text().splitlines():
        if line.startswith("Parameters:"):
            computed = np.array([float(v) for v in line.split(": ", 1)[1].split()])
            break

    np.testing.assert_allclose(computed, expected_params, atol=1e-10)


def test_itk_output_format(tmp_path: Path) -> None:
    """Output file has the expected ITK transform header structure."""
    mat_file = tmp_path / "identity.mat"
    np.savetxt(mat_file, np.eye(4))

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(_make_ref_image(), ref_path)

    output = tmp_path / "test.txt"
    mat_to_itk(mat_file, ref_path, ref_path, str(output))

    lines = output.read_text().splitlines()
    assert lines[0] == "#Insight Transform File V1.0"
    assert lines[1] == "#Transform 0"
    assert lines[2] == "Transform: MatrixOffsetTransformBase_double_3_3"
    assert lines[3].startswith("Parameters:")
    assert lines[4] == "FixedParameters: 0 0 0"

    params = lines[3].split(": ", 1)[1].split()
    assert len(params) == 12


def test_identity_mat_gives_identity_transform(tmp_path: Path) -> None:
    """An identity FSL matrix should produce a near-identity ITK transform."""
    mat_file = tmp_path / "identity.mat"
    np.savetxt(mat_file, np.eye(4))

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(_make_ref_image(), ref_path)

    output = tmp_path / "identity.txt"
    mat_to_itk(mat_file, ref_path, ref_path, str(output))

    for line in output.read_text().splitlines():
        if line.startswith("Parameters:"):
            params = np.array([float(v) for v in line.split(": ", 1)[1].split()])
            break

    rotation = params[:9].reshape(3, 3)
    translation = params[9:]
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(translation, np.zeros(3), atol=1e-10)


def test_rejects_non_4x4_matrix(tmp_path: Path) -> None:
    """mat_to_itk raises ValueError for a non-4x4 matrix."""
    mat_file = tmp_path / "bad.mat"
    np.savetxt(mat_file, np.eye(3))

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(_make_ref_image(), ref_path)

    with pytest.raises(ValueError, match="Expected 4x4"):
        mat_to_itk(mat_file, ref_path, ref_path, str(tmp_path / "out.txt"))


def test_rejects_bad_last_row(tmp_path: Path) -> None:
    """mat_to_itk raises ValueError when the last row is not [0,0,0,1]."""
    bad = np.eye(4)
    bad[3, 3] = 2.0
    mat_file = tmp_path / "bad.mat"
    np.savetxt(mat_file, bad)

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(_make_ref_image(), ref_path)

    with pytest.raises(ValueError, match="Last row"):
        mat_to_itk(mat_file, ref_path, ref_path, str(tmp_path / "out.txt"))


def test_fsl_swap_matrix_radiological() -> None:
    """Swap matrix is identity for radiological (negative det) images."""
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10, 10, 10)), affine)
    swap = _fsl_swap_matrix(img)
    np.testing.assert_array_equal(swap, np.eye(4))


def test_fsl_swap_matrix_neurological() -> None:
    """Swap matrix flips x-axis for neurological (positive det) images."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10, 10, 10)), affine)
    swap = _fsl_swap_matrix(img)

    assert swap[0, 0] == -1.0
    assert swap[0, 3] == (10 - 1) * 2.0
    assert swap[1, 1] == 1.0
    assert swap[2, 2] == 1.0


def test_fsl_to_ras_identity() -> None:
    """fsl_to_ras with identity FSL mat and same ref/src gives identity."""
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10, 10, 10)), affine)
    result = _fsl_to_ras(np.eye(4), img, img)
    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)
