"""Integration tests for AFNI methods used across modalities."""

from types import SimpleNamespace

from niwrap import afni

from rbc.core.common import reorient


def test_reorient(test_subject: SimpleNamespace) -> None:
    """Test deobliqueing and reorientation."""
    reoriented_file = reorient(test_subject.t1w, output_fname="test.nii.gz")
    assert (
        afni.v_3dinfo(dataset=[reoriented_file.out_file], orient=True).info[0] == "RPI"
    )
