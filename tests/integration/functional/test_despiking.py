"""Tests for BOLD timeseries despiking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import pytest

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import despike_bold
from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_despike(test_subject: TestSubjectData) -> None:
    """Test that despike runs successfully and produces output."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    despiked = despike_bold(in_file=reoriented.out_file)
    assert despiked.exists()


@pytest.mark.slow
def test_despike_reduces_single_outlier(test_subject: TestSubjectData) -> None:
    """Test that despike attenuates an artificial spike in a single voxel."""
    img = nib.nifti1.load(test_subject.bold)
    data = img.get_fdata()

    center_x, center_y, center_z = [s // 2 for s in data.shape[:3]]

    original_val = data[center_x, center_y, center_z, 5]

    # Inject spike into single voxel
    spike_multiplier = 100
    data[center_x, center_y, center_z, 5] = data.mean() * spike_multiplier
    spiked_val = data[center_x, center_y, center_z, 5]

    out_dir = generate_exec_folder("spiked_bold")
    spiked_path = out_dir / "spiked_input.nii.gz"
    nib.save(nib.Nifti1Image(data, img.affine, img.header), spiked_path)

    despiked = despike_bold(in_file=spiked_path)

    despiked_data = nib.nifti1.load(despiked).get_fdata()
    despiked_val = despiked_data[center_x, center_y, center_z, 5]

    assert despiked_val < spiked_val, (
        f"Despike failed to reduce spike: {despiked_val} >= {spiked_val}"
    )
    assert abs(despiked_val - original_val) < abs(spiked_val - original_val) * 0.5, (
        f"Despiked value {despiked_val} too far from original {original_val}"
    )


@pytest.mark.slow
def test_despike_multiple_spikes(test_subject: TestSubjectData) -> None:
    """Test that despike attenuates multiple scattered spikes across volumes."""
    img = nib.nifti1.load(test_subject.bold)
    data = img.get_fdata()

    center_x, center_y, center_z = [s // 2 for s in data.shape[:3]]

    # Inject spikes into multiple voxels
    spike_multiplier = 100
    spike_coords = [
        (center_x, center_y, center_z, 5),
        (center_x + 2, center_y - 2, center_z, 10),
        (center_x + 4, center_y - 4, center_z, 15),
    ]

    for coord in spike_coords:
        data[coord] = data.mean() * spike_multiplier

    out_dir = generate_exec_folder("multi_spiked_bold")
    spiked_path = out_dir / "multi_spiked.nii.gz"
    nib.save(nib.Nifti1Image(data, img.affine, img.header), spiked_path)

    despiked = despike_bold(in_file=spiked_path)
    despiked_data = nib.nifti1.load(despiked).get_fdata()

    for coord in spike_coords:
        spiked_val = data[coord]
        despiked_val = despiked_data[coord]
        assert despiked_val < spiked_val, (
            f"Despike failed to reduce spike at {coord}: {despiked_val} >= {spiked_val}"
        )
