"""Unit tests for rbc.core.functional.mask_utils."""

from __future__ import annotations

import numpy as np
import pytest

from rbc.core.functional.mask_utils import (
    erode_mask_by_distance,
    erode_mask_to_proportion,
)

SHAPE = (30, 30, 30)


def _make_sphere(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    """Create a binary sphere mask centred in a volume."""
    centre = np.array(shape) / 2
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(float)
    dist = np.sqrt(sum((coords[i] - centre[i]) ** 2 for i in range(3)))
    return (dist <= radius).astype(np.uint8)


# ===================================================================
# erode_mask_to_proportion
# ===================================================================
class TestErodeMaskToProportion:
    """Tests for iterative proportion-based erosion."""

    def test_shape_preserved(self) -> None:
        """Output shape should match input shape."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_to_proportion(mask, 0.6)
        assert result.shape == SHAPE

    def test_volume_reduced(self) -> None:
        """Eroded mask should have fewer voxels than the original."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_to_proportion(mask, 0.6)
        assert result.sum() < mask.sum()

    def test_close_to_target(self) -> None:
        """Eroded volume should be close to the target proportion.

        The overshoot guard picks whichever erosion step is closest to
        the target, so the result may be slightly above or below.
        """
        mask = _make_sphere(SHAPE, 12)
        target = 0.6
        result = erode_mask_to_proportion(mask, target)
        actual_proportion = result.sum() / mask.sum()
        # Should be within one erosion layer of the target
        assert abs(actual_proportion - target) < 0.15

    def test_subset_of_original(self) -> None:
        """Eroded mask should be a subset of the original."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_to_proportion(mask, 0.6)
        assert np.all(result[mask == 0] == 0)

    def test_high_proportion_preserves_more(self) -> None:
        """Higher target proportion should retain more voxels."""
        mask = _make_sphere(SHAPE, 12)
        result_90 = erode_mask_to_proportion(mask, 0.9)
        result_60 = erode_mask_to_proportion(mask, 0.6)
        assert result_90.sum() >= result_60.sum()

    def test_empty_mask(self) -> None:
        """Empty mask should return empty mask."""
        mask = np.zeros(SHAPE, dtype=np.uint8)
        result = erode_mask_to_proportion(mask, 0.6)
        assert result.sum() == 0

    def test_rejects_non_3d(self) -> None:
        """Non-3D mask should raise."""
        with pytest.raises(ValueError, match="3D"):
            erode_mask_to_proportion(np.zeros((5, 5)), 0.6)

    def test_rejects_bad_proportion_zero(self) -> None:
        """Proportion of 0 should raise."""
        with pytest.raises(ValueError, match="target_proportion"):
            erode_mask_to_proportion(np.ones(SHAPE), 0.0)

    def test_rejects_bad_proportion_one(self) -> None:
        """Proportion of 1 should raise."""
        with pytest.raises(ValueError, match="target_proportion"):
            erode_mask_to_proportion(np.ones(SHAPE), 1.0)

    def test_rejects_bad_proportion_negative(self) -> None:
        """Negative proportion should raise."""
        with pytest.raises(ValueError, match="target_proportion"):
            erode_mask_to_proportion(np.ones(SHAPE), -0.5)


# ===================================================================
# erode_mask_by_distance
# ===================================================================
class TestErodeMaskByDistance:
    """Tests for fixed-distance erosion."""

    def test_shape_preserved(self) -> None:
        """Output shape should match input shape."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        assert result.shape == SHAPE

    def test_volume_reduced(self) -> None:
        """Eroded mask should have fewer voxels than the original."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        assert result.sum() < mask.sum()

    def test_larger_distance_more_erosion(self) -> None:
        """Larger distance should erode more."""
        mask = _make_sphere(SHAPE, 12)
        result_3 = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        result_6 = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 6.0)
        assert result_6.sum() <= result_3.sum()

    def test_subset_of_original(self) -> None:
        """Eroded mask should be a subset of the original."""
        mask = _make_sphere(SHAPE, 12)
        result = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        assert np.all(result[mask == 0] == 0)

    def test_voxel_size_affects_erosion(self) -> None:
        """Check if voxel size affects erosion.

        Larger voxels → larger physical sphere → more
        voxels survive same mm erosion.
        """
        mask = _make_sphere(SHAPE, 12)
        # Same voxel-space sphere, but 2mm voxels = 24mm physical radius
        result_2mm = erode_mask_by_distance(mask, (2.0, 2.0, 2.0), 3.0)
        # 1mm voxels = 12mm physical radius
        result_1mm = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        assert result_2mm.sum() >= result_1mm.sum()

    def test_anisotropic_voxels(self) -> None:
        """Anisotropic voxels should erode more along the coarse axis."""
        mask = _make_sphere(SHAPE, 12)
        # Isotropic 1mm
        result_iso = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        # Anisotropic: 3mm along z means 3mm erosion removes ~1 voxel in z
        # but 3 voxels in x/y → less total erosion than isotropic
        result_aniso = erode_mask_by_distance(mask, (1.0, 1.0, 3.0), 3.0)
        # With EDT, anisotropic correctly accounts for physical distance,
        # so the result should differ from isotropic
        assert result_iso.sum() != result_aniso.sum()

    def test_empty_mask(self) -> None:
        """Empty mask should return empty mask."""
        mask = np.zeros(SHAPE, dtype=np.uint8)
        result = erode_mask_by_distance(mask, (1.0, 1.0, 1.0), 3.0)
        assert result.sum() == 0

    def test_rejects_non_3d(self) -> None:
        """Non-3D mask should raise."""
        with pytest.raises(ValueError, match="3D"):
            erode_mask_by_distance(np.zeros((5, 5)), (1.0, 1.0, 1.0), 3.0)

    def test_rejects_non_positive_distance(self) -> None:
        """Non-positive distance should raise."""
        with pytest.raises(ValueError, match="distance_mm"):
            erode_mask_by_distance(np.ones(SHAPE), (1.0, 1.0, 1.0), 0.0)
        with pytest.raises(ValueError, match="distance_mm"):
            erode_mask_by_distance(np.ones(SHAPE), (1.0, 1.0, 1.0), -1.0)
