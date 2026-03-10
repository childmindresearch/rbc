"""Integration tests for functional distortion correction.

These tests require a Docker runner and create synthetic fieldmap data from
the existing test BOLD, since the default test dataset (ds000001) does not
include fieldmaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from niwrap import ants

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import extract_motion_reference
from rbc.core.functional.distortion import (
    correct_distortion_pepolar,
    correct_distortion_phasediff,
)
from rbc.core.nifti import nifti_num_volumes

if TYPE_CHECKING:
    from conftest import TestSubjectData


def _create_synthetic_magnitude(bold_ref: nib.Nifti1Image, path: str) -> None:
    """Save the BOLD ref data as a fake magnitude image."""
    nib.save(bold_ref, path)


def _create_synthetic_phasediff(
    bold_ref: nib.Nifti1Image, path: str, seed: int = 42
) -> None:
    """Create a synthetic phase-difference image with random phase values."""
    rng = np.random.default_rng(seed)
    shape = bold_ref.shape[:3]
    # Phase values in 0-4096 range (typical Siemens 12-bit phase)
    phase_data = rng.integers(0, 4096, size=shape).astype(np.float32)
    phase_img = nib.Nifti1Image(
        phase_data, affine=bold_ref.affine, header=bold_ref.header
    )
    nib.save(phase_img, path)


def _create_synthetic_epi_pair(
    bold_ref: nib.Nifti1Image, ap_path: str, pa_path: str
) -> None:
    """Create synthetic AP/PA EPI pair from a BOLD reference."""
    data = np.asarray(bold_ref.dataobj, dtype=np.float32)
    # AP: use ref data as-is (single volume as 4D)
    ap_data = data[..., np.newaxis]
    ap_img = nib.Nifti1Image(ap_data, affine=bold_ref.affine, header=bold_ref.header)
    nib.save(ap_img, ap_path)

    # PA: flip along j-axis as a crude simulation of reversed PE distortion
    pa_data = ap_data[:, ::-1, :, :]
    pa_img = nib.Nifti1Image(pa_data, affine=bold_ref.affine, header=bold_ref.header)
    nib.save(pa_img, pa_path)


@pytest.mark.slow
def test_correct_distortion_phasediff(test_subject: TestSubjectData) -> None:
    """Phase-difference correction produces valid corrected ref and warp."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    motion_ref = extract_motion_reference(in_file=reoriented.out_file)

    ref_img = nib.nifti1.load(motion_ref)
    from rbc.core.niwrap import generate_exec_folder

    out_dir = generate_exec_folder(suffix="test_phasediff_synth")

    mag_path = out_dir / "magnitude.nii.gz"
    phasediff_path = out_dir / "phasediff.nii.gz"
    _create_synthetic_magnitude(ref_img, str(mag_path))
    _create_synthetic_phasediff(ref_img, str(phasediff_path))

    result = correct_distortion_phasediff(
        bold_ref=motion_ref,
        magnitude=mag_path,
        delta_te=2.46,
        effective_echo_spacing=0.00068,
        pe_direction="j",
        phasediff=phasediff_path,
    )

    assert result.corrected_ref.exists()
    assert result.warp_field.exists()
    assert nifti_num_volumes(result.corrected_ref) == 1

    corrected = nib.nifti1.load(result.corrected_ref)
    assert corrected.shape[:3] == ref_img.shape[:3]


@pytest.mark.slow
def test_correct_distortion_pepolar(test_subject: TestSubjectData) -> None:
    """PEPOLAR correction produces valid corrected ref and warp."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    motion_ref = extract_motion_reference(in_file=reoriented.out_file)

    ref_img = nib.nifti1.load(motion_ref)
    from rbc.core.niwrap import generate_exec_folder

    out_dir = generate_exec_folder(suffix="test_pepolar_synth")

    ap_path = out_dir / "epi_ap.nii.gz"
    pa_path = out_dir / "epi_pa.nii.gz"
    _create_synthetic_epi_pair(ref_img, str(ap_path), str(pa_path))

    result = correct_distortion_pepolar(
        bold_ref=motion_ref,
        epi_ap=ap_path,
        epi_pa=pa_path,
        readout_time_ap=0.05,
        readout_time_pa=0.05,
        pe_direction="j",
    )

    assert result.corrected_ref.exists()
    assert result.warp_field.exists()
    assert nifti_num_volumes(result.corrected_ref) == 1

    corrected = nib.nifti1.load(result.corrected_ref)
    assert corrected.shape[:3] == ref_img.shape[:3]


@pytest.mark.slow
def test_warp_field_compatible_with_ants(test_subject: TestSubjectData) -> None:
    """The ITK warp field can be used by ants.ants_apply_transforms."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    motion_ref = extract_motion_reference(in_file=reoriented.out_file)

    ref_img = nib.nifti1.load(motion_ref)
    from rbc.core.niwrap import generate_exec_folder

    out_dir = generate_exec_folder(suffix="test_ants_compat")

    mag_path = out_dir / "magnitude.nii.gz"
    phasediff_path = out_dir / "phasediff.nii.gz"
    _create_synthetic_magnitude(ref_img, str(mag_path))
    _create_synthetic_phasediff(ref_img, str(phasediff_path))

    result = correct_distortion_phasediff(
        bold_ref=motion_ref,
        magnitude=mag_path,
        delta_te=2.46,
        effective_echo_spacing=0.00068,
        pe_direction="j",
        phasediff=phasediff_path,
    )

    # Apply the warp field using ANTs to verify format compatibility
    ants_result = ants.ants_apply_transforms(
        input_image=motion_ref,
        reference_image=motion_ref,
        transform=[ants.ants_apply_transforms_transform_file_name(result.warp_field)],
        dimensionality=3,
        output=ants.ants_apply_transforms_warped_output("ants_test_output.nii.gz"),
    )

    assert ants_result.output.output_image_outfile.exists()
