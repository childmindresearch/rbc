"""Integration tests for XCP-style QC output using real data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import polars as pl
import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
)
from rbc.core.qc.dvars import dvars_qc_metrics
from rbc.core.qc.motion import framewise_displacement_jenkinson, motion_qc_metrics
from rbc.core.qc.registration import registration_qc_metrics
from rbc.core.qc.xcp import (
    XCPQCMetrics,
    generate_xcp_qc,
    passes_rbc_qc,
    write_xcp_qc,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestSubjectData


@pytest.mark.slow
def test_xcp_qc_from_bold(
    test_subject: TestSubjectData,
    tmp_path: Path,
) -> None:
    """Compute all sub-metrics from real data, generate XCP TSV, and verify."""
    # Preprocess: deoblique, truncate to 10 volumes, motion correct
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="xcp_qc_test_10vols.nii.gz",
    )

    ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=ref.output_file,
    )

    # Load MCFLIRT outputs
    rms_values = np.loadtxt(mc.rms_rel)
    motion_params = np.loadtxt(mc.par)
    fd = framewise_displacement_jenkinson(rms_values)

    # Load motion-corrected BOLD and create a brain mask
    bold_img = nib.nifti1.load(mc.bold.with_suffix(".nii.gz"))
    bold_data = bold_img.get_fdata()
    mask = np.mean(bold_data, axis=3) > 0

    # Compute sub-metrics
    motion = motion_qc_metrics(rms_values, motion_params)
    dvars_init = dvars_qc_metrics(bold_data, mask, fd)
    dvars_final = dvars_qc_metrics(bold_data, mask, fd)  # same data for test
    coreg = registration_qc_metrics(mask, mask)  # self-comparison for test
    norm = registration_qc_metrics(mask, mask)  # self-comparison for test

    # Generate XCP QC row
    metrics = generate_xcp_qc(
        sub=test_subject.subject_id,
        ses="001",
        task="balloonanalogrisktask",
        run=1,
        desc="preproc",
        regressors="36P",
        space="MNI152NLin2009cAsym",
        motion=motion,
        dvars_init=dvars_init,
        dvars_final=dvars_final,
        n_vols_removed=0,
        coreg=coreg,
        norm=norm,
    )

    assert isinstance(metrics, XCPQCMetrics)
    assert metrics.sub == test_subject.subject_id
    assert metrics.meanFD >= 0
    assert metrics.meanDVInit >= 0

    # Write TSV and verify
    out_path = tmp_path / "xcp_qc.tsv"
    write_xcp_qc(metrics, out_path)
    assert out_path.exists()

    df = pl.read_csv(out_path, separator="\t")
    assert df.shape == (1, 24)
    assert df["sub"][0] == test_subject.subject_id
    assert df["meanFD"][0] == metrics.meanFD

    # Verify passes_rbc_qc runs without error
    result = passes_rbc_qc(fd, metrics.normCrossCorr)
    assert isinstance(result, bool)
