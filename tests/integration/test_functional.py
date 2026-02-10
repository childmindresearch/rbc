"""Integration tests for functional workflow."""

from types import SimpleNamespace

import nibabel as nib
import pytest
from niwrap import afni

from rbc.core.common import reorient
from rbc.core.functional import (
    generate_motion_reference,
    motion_correction,
    truncate_trs,
)


def test_truncate_trs(test_subject: SimpleNamespace) -> None:
    """Test truncating initial TRs from BOLD timeseries."""
    original_count = nib.load(test_subject.bold).shape[3]

    start_tr = 4
    reoriented = reorient(
        in_file=test_subject.bold, output_fname="test_reoriented.nii.gz"
    )
    truncated_bold = truncate_trs(
        in_file=reoriented.out_file,
        output_fname="test_truncated.nii.gz",
        start_tr=start_tr,
    )
    # Test truncated BOLD file exists & volume count is reduced
    assert truncated_bold.output_file.exists()
    new_shape = nib.load(truncated_bold.output_file).shape
    assert new_shape[3] == original_count - start_tr


def test_truncate_to_min_volume(test_subject: SimpleNamespace) -> None:
    """Test truncating to minimum volume count of 1."""
    original_count = nib.load(test_subject.bold).shape[3]

    start_tr = original_count - 1
    truncated_bold = truncate_trs(
        in_file=test_subject.bold,
        output_fname="test_truncated_min.nii.gz",
        start_tr=start_tr,
    )
    # Test truncated BOLD file volume count is 1
    new_shape = nib.load(truncated_bold.output_file).shape
    nvols = new_shape[3] if len(new_shape) > 3 else 1
    assert nvols == 1


def test_motion_reference_volume_count(test_subject: SimpleNamespace) -> None:
    """Test motion reference volume count is 1."""
    reference = generate_motion_reference(
        in_file=test_subject.bold, output_fname="test_motion_ref.nii.gz"
    )
    # Test motion reference file has 1 volume
    ref_shape = nib.load(reference.output_file).shape
    nvols = ref_shape[3] if len(ref_shape) > 3 else 1
    assert nvols == 1


@pytest.mark.slow
def test_motion_correction_10vols(test_subject: SimpleNamespace) -> None:
    """Test motion correction on 10 volumes of BOLD timeseries."""
    reoriented = reorient(
        in_file=test_subject.bold, output_fname="test_reoriented.nii.gz"
    )
    truncated_10 = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="test_10vols.nii.gz",
    )
    motion_reference = generate_motion_reference(
        in_file=truncated_10.output_file, output_fname="test_ref_10v.nii.gz"
    )
    motion_corrected = motion_correction(
        in_file=truncated_10.output_file,
        ref_file=motion_reference.output_file,
        output_prefix="test_mc_10v",
    )
    assert motion_corrected.bold.with_suffix(".nii.gz").exists()
    mc_shape = nib.load(motion_corrected.bold.with_suffix(".nii.gz")).shape
    assert mc_shape[3] == 10

    assert motion_corrected.par.exists()
    par_data = motion_corrected.par.read_text().splitlines()
    assert len(par_data) == 10


@pytest.mark.slow
def test_motion_correction(test_subject: SimpleNamespace) -> None:
    """Test motion correction on full BOLD timeseries."""
    reoriented = reorient(
        in_file=test_subject.bold, output_fname="test_reoriented.nii.gz"
    )
    truncated = truncate_trs(
        in_file=reoriented.out_file, output_fname="test_truncated.nii.gz", start_tr=2
    )
    motion_reference = generate_motion_reference(
        in_file=truncated.output_file, output_fname="test_full_ref.nii.gz"
    )
    motion_corrected = motion_correction(
        in_file=truncated.output_file,
        ref_file=motion_reference.output_file,
        output_prefix="test_full_mc",
    )
    # Test motion corrected BOLD files exists
    assert motion_corrected.bold.with_suffix(".nii.gz").exists()
    assert motion_corrected.par.exists()
    assert motion_corrected.rms_rel.exists()
