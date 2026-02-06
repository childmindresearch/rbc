"""General functions useful across modalities."""

from pathlib import Path

from niwrap import afni

from rbc.core.utils import create_copy


def reorient(in_file: Path, output_fname: str) -> afni.V3dresampleOutputs:
    """AFNI deobliquing and reorientation to RPI.

    Sets image into a cardinal orientation if it was acquired obliquely from scanner
    and standardize orientation of images ('RPI' is internal assumption from AFNI).

    Args:
        in_file: Input T1w to reorient
        output_fname: Output filename

    Returns:
        An object representing the outputs from AFNI's 3D resample.
    """
    with create_copy(in_file) as tmp_file:
        afni.v_3drefit(in_file=tmp_file, deoblique=True)
        reorient = afni.v_3dresample(
            in_file=tmp_file, prefix=output_fname, orientation="RPI"
        )
    return reorient
