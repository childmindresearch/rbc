"""Unit tests for rbc.core.functional.distortion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.functional.distortion import (
    _PE_DIR_VECTORS,
    _PE_TO_FUGUE,
    _fieldmap_hz_to_itk_warp,
    _shiftmap_to_itk_warp,
    _validate_pe_direction,
    _write_acqparams,
)


# ===================================================================
# PE direction mappings
# ===================================================================
class TestPeDirectionMappings:
    """Tests for phase-encoding direction lookup tables."""

    @pytest.mark.parametrize(
        ("bids_dir", "fugue_dir"),
        [("i", "x"), ("i-", "x-"), ("j", "y"), ("j-", "y-"), ("k", "z"), ("k-", "z-")],
    )
    def test_pe_to_fugue(self, bids_dir: str, fugue_dir: str) -> None:
        """BIDS PE directions map to correct FUGUE unwarp codes."""
        assert _PE_TO_FUGUE[bids_dir] == fugue_dir

    @pytest.mark.parametrize(
        ("bids_dir", "expected"),
        [
            ("i", (1, 0, 0)),
            ("i-", (-1, 0, 0)),
            ("j", (0, 1, 0)),
            ("j-", (0, -1, 0)),
            ("k", (0, 0, 1)),
            ("k-", (0, 0, -1)),
        ],
    )
    def test_pe_dir_vectors(
        self, bids_dir: str, expected: tuple[int, int, int]
    ) -> None:
        """BIDS PE directions map to correct unit direction vectors."""
        assert _PE_DIR_VECTORS[bids_dir] == expected

    def test_invalid_direction_raises_keyerror(self) -> None:
        """Invalid PE direction raises KeyError on lookup."""
        with pytest.raises(KeyError):
            _PE_TO_FUGUE["q"]

    def test_validate_pe_direction_valid(self) -> None:
        """Valid PE directions do not raise."""
        for d in ("i", "i-", "j", "j-", "k", "k-"):
            _validate_pe_direction(d)

    def test_validate_pe_direction_invalid(self) -> None:
        """Invalid PE direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pe_direction"):
            _validate_pe_direction("q")


# ===================================================================
# _write_acqparams
# ===================================================================
class TestWriteAcqparams:
    """Tests for TOPUP acquisition parameters file writer."""

    @staticmethod
    def _make_nifti(shape: tuple[int, ...], tmp_path: Path, name: str) -> Path:
        """Create a minimal NIfTI with given shape."""
        data = np.zeros(shape, dtype=np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        path = tmp_path / name
        nib.save(img, path)
        return path

    def test_basic_j_direction(self, tmp_path: Path) -> None:
        """AP (j) + PA (j-) with known readout times."""
        epi_ap = self._make_nifti((4, 4, 4, 1), tmp_path, "ap.nii.gz")
        epi_pa = self._make_nifti((4, 4, 4, 2), tmp_path, "pa.nii.gz")
        output = tmp_path / "acqparams.txt"

        _write_acqparams(
            pe_dirs=["j", "j-"],
            readout_times=[0.05, 0.05],
            epi_files=[epi_ap, epi_pa],
            output=output,
        )

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3  # 1 AP volume + 2 PA volumes
        assert lines[0] == "0 1 0 0.05"
        assert lines[1] == "0 -1 0 0.05"
        assert lines[2] == "0 -1 0 0.05"

    def test_i_direction(self, tmp_path: Path) -> None:
        """RL (i) direction produces correct x-axis encoding."""
        epi = self._make_nifti((4, 4, 4, 1), tmp_path, "rl.nii.gz")
        output = tmp_path / "acqparams.txt"

        _write_acqparams(
            pe_dirs=["i"],
            readout_times=[0.03],
            epi_files=[epi],
            output=output,
        )

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        assert lines[0] == "1 0 0 0.03"

    def test_3d_epi_treated_as_single_volume(self, tmp_path: Path) -> None:
        """A 3D EPI (no time dimension) is treated as 1 volume."""
        epi = self._make_nifti((4, 4, 4), tmp_path, "epi_3d.nii.gz")
        output = tmp_path / "acqparams.txt"

        _write_acqparams(
            pe_dirs=["j"],
            readout_times=[0.05],
            epi_files=[epi],
            output=output,
        )

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_invalid_pe_direction_raises(self, tmp_path: Path) -> None:
        """Invalid PE direction raises ValueError."""
        epi = self._make_nifti((4, 4, 4, 1), tmp_path, "epi.nii.gz")
        with pytest.raises(ValueError, match="Invalid pe_direction"):
            _write_acqparams(
                pe_dirs=["q"],
                readout_times=[0.05],
                epi_files=[epi],
                output=tmp_path / "acqparams.txt",
            )


# ===================================================================
# _shiftmap_to_itk_warp
# ===================================================================
class TestShiftmapToItkWarp:
    """Tests for FUGUE shift map to ITK warp conversion."""

    def test_j_direction_displacement(self, tmp_path: Path) -> None:
        """Shift of 2 voxels along j with 2mm voxels = 4mm displacement."""
        shift = np.full((3, 3, 3), 2.0, dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(
            nib.Nifti1Image(shift, affine),
            tmp_path / "shift.nii.gz",
        )
        result = _shiftmap_to_itk_warp(
            tmp_path / "shift.nii.gz", "j", tmp_path / "warp.nii.gz"
        )
        warp = np.asarray(nib.nifti1.load(result).dataobj)
        # j-axis → index 1; LPS negates y → expect -4.0
        np.testing.assert_allclose(warp[..., 0], 0.0)
        np.testing.assert_allclose(warp[..., 1], -4.0)
        np.testing.assert_allclose(warp[..., 2], 0.0)

    def test_i_direction_displacement(self, tmp_path: Path) -> None:
        """Shift along i axis lands in x component (negated for LPS)."""
        shift = np.full((3, 3, 3), 1.0, dtype=np.float32)
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        nib.save(
            nib.Nifti1Image(shift, affine),
            tmp_path / "shift.nii.gz",
        )
        result = _shiftmap_to_itk_warp(
            tmp_path / "shift.nii.gz", "i", tmp_path / "warp.nii.gz"
        )
        warp = np.asarray(nib.nifti1.load(result).dataobj)
        # i-axis → index 0; LPS negates x → expect -3.0
        np.testing.assert_allclose(warp[..., 0], -3.0)
        np.testing.assert_allclose(warp[..., 1], 0.0)
        np.testing.assert_allclose(warp[..., 2], 0.0)

    def test_intent_code_and_dtype(self, tmp_path: Path) -> None:
        """Output has vector intent and float32 dtype."""
        shift = np.zeros((3, 3, 3), dtype=np.float32)
        nib.save(
            nib.Nifti1Image(shift, np.eye(4)),
            tmp_path / "shift.nii.gz",
        )
        result = _shiftmap_to_itk_warp(
            tmp_path / "shift.nii.gz", "j", tmp_path / "warp.nii.gz"
        )
        img = nib.nifti1.load(result)
        assert img.header["intent_code"] == 1007
        assert img.get_data_dtype() == np.float32


# ===================================================================
# _fieldmap_hz_to_itk_warp
# ===================================================================
class TestFieldmapHzToItkWarp:
    """Tests for TOPUP Hz fieldmap to ITK warp conversion."""

    def test_j_direction_displacement(self, tmp_path: Path) -> None:
        """10 Hz field, 0.05s readout, 2mm voxels = 1mm displacement."""
        fmap = np.full((3, 3, 3), 10.0, dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(
            nib.Nifti1Image(fmap, affine),
            tmp_path / "fmap.nii.gz",
        )
        result = _fieldmap_hz_to_itk_warp(
            tmp_path / "fmap.nii.gz", "j", 0.05,
            tmp_path / "warp.nii.gz",
        )
        warp = np.asarray(nib.nifti1.load(result).dataobj)
        # 10 * 0.05 * 2.0 = 1.0 mm; LPS negates y → -1.0
        np.testing.assert_allclose(warp[..., 0], 0.0)
        np.testing.assert_allclose(warp[..., 1], -1.0)
        np.testing.assert_allclose(warp[..., 2], 0.0)

    def test_4d_fieldmap_uses_first_volume(self, tmp_path: Path) -> None:
        """4D fieldmap (multi-volume) uses first volume only."""
        fmap = np.zeros((3, 3, 3, 2), dtype=np.float32)
        fmap[..., 0] = 5.0
        fmap[..., 1] = 99.0  # should be ignored
        nib.save(
            nib.Nifti1Image(fmap, np.diag([2.0, 2.0, 2.0, 1.0])),
            tmp_path / "fmap.nii.gz",
        )
        result = _fieldmap_hz_to_itk_warp(
            tmp_path / "fmap.nii.gz", "j", 0.1,
            tmp_path / "warp.nii.gz",
        )
        warp = np.asarray(nib.nifti1.load(result).dataobj)
        # 5 * 0.1 * 2.0 = 1.0; negated for LPS y
        np.testing.assert_allclose(warp[..., 1], -1.0)

    def test_negative_pe_flips_sign(self, tmp_path: Path) -> None:
        """j- direction produces opposite displacement to j."""
        fmap = np.full((3, 3, 3), 10.0, dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(
            nib.Nifti1Image(fmap, affine),
            tmp_path / "fmap.nii.gz",
        )
        result = _fieldmap_hz_to_itk_warp(
            tmp_path / "fmap.nii.gz", "j-", 0.05,
            tmp_path / "warp.nii.gz",
        )
        warp = np.asarray(nib.nifti1.load(result).dataobj)
        # sign=-1, so 10*0.05*2.0*(-1) = -1.0; LPS negates y → +1.0
        np.testing.assert_allclose(warp[..., 1], 1.0)


# ===================================================================
# Input validation for main correction functions
# ===================================================================
class TestCorrectDistortionPhasediffValidation:
    """Input validation tests for correct_distortion_phasediff."""

    def test_no_phase_inputs_raises(self, tmp_path: Path) -> None:
        """Neither phasediff nor phase1+phase2 raises ValueError."""
        from rbc.core.functional.distortion import correct_distortion_phasediff

        dummy = tmp_path / "dummy.nii.gz"
        img = nib.Nifti1Image(
            np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)
        )
        nib.save(img, dummy)

        with pytest.raises(ValueError, match="Must provide either"):
            correct_distortion_phasediff(
                bold_ref=dummy,
                magnitude=dummy,
                delta_te=2.46,
                effective_echo_spacing=0.00068,
                pe_direction="j",
            )

    def test_phase1_without_phase2_raises(self, tmp_path: Path) -> None:
        """Providing phase1 without phase2 raises ValueError."""
        from rbc.core.functional.distortion import correct_distortion_phasediff

        dummy = tmp_path / "dummy.nii.gz"
        img = nib.Nifti1Image(
            np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)
        )
        nib.save(img, dummy)

        with pytest.raises(ValueError, match="Must provide either"):
            correct_distortion_phasediff(
                bold_ref=dummy,
                magnitude=dummy,
                delta_te=2.46,
                effective_echo_spacing=0.00068,
                pe_direction="j",
                phase1=dummy,
            )

    def test_invalid_pe_direction_raises(self, tmp_path: Path) -> None:
        """Invalid pe_direction raises ValueError."""
        from rbc.core.functional.distortion import correct_distortion_phasediff

        dummy = tmp_path / "dummy.nii.gz"
        img = nib.Nifti1Image(
            np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)
        )
        nib.save(img, dummy)

        with pytest.raises(ValueError, match="Invalid pe_direction"):
            correct_distortion_phasediff(
                bold_ref=dummy,
                magnitude=dummy,
                delta_te=2.46,
                effective_echo_spacing=0.00068,
                pe_direction="bad",
                phasediff=dummy,
            )


class TestCorrectDistortionPepolarValidation:
    """Input validation tests for correct_distortion_pepolar."""

    def test_invalid_pe_direction_raises(self, tmp_path: Path) -> None:
        """Invalid pe_direction raises ValueError."""
        from rbc.core.functional.distortion import correct_distortion_pepolar

        dummy = tmp_path / "dummy.nii.gz"
        img = nib.Nifti1Image(
            np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)
        )
        nib.save(img, dummy)

        with pytest.raises(ValueError, match="Invalid pe_direction"):
            correct_distortion_pepolar(
                bold_ref=dummy,
                epi_ap=dummy,
                epi_pa=dummy,
                readout_time_ap=0.05,
                readout_time_pa=0.05,
                pe_direction="bad",
            )
