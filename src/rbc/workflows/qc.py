"""QC metrics workflow.

Orchestrates quality control metrics from functional and anatomical
preprocessing outputs, returning all QC data as a :class:`QCOutputs`
named tuple.  No BIDS naming or file copying is performed here -- that
responsibility belongs to the CLI layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from niwrap import fsl

from rbc.bids import TemplateSpace
from rbc.core.qc.dvars import dvars_qc_metrics
from rbc.core.qc.motion import framewise_displacement_jenkinson, motion_qc_metrics
from rbc.core.qc.registration import registration_qc_metrics
from rbc.core.qc.xcp import XCPQCMetrics, generate_xcp_qc, passes_rbc_qc, write_xcp_qc
from rbc_resources import REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_logger = logging.getLogger("rbc")


@dataclass
class QCOutputs:
    """QC outputs from the metrics workflow.

    Attributes:
        metrics: All 24 XCP-style QC fields for the run.
        qc_file: Path to the written single-row TSV.
        passed: Whether the run passes RBC QC thresholds.
    """

    metrics: dict[str, XCPQCMetrics] = field(default_factory=dict)
    qc_file: dict[str, Path] = field(default_factory=dict)
    passed: bool = field(default=False)  # Set default to fail


def single_session_qc(
    template_bold: Path,
    cleaned_bold: Mapping[str, Path],
    motion_params: Path,
    rms_rel: Path,
    bold_mask: Path,
    brain_mask: Path,
    bold_to_anat_matrix: Path,
    template_brain_mask: Path,
    sub: str,
    ses: str,
    task: str,
    run: int,
    start_tr: int = 2,
    regressor_set: Sequence[str] = ("36-parameter",),
    mni_brain_mask_2mm: Path = REGISTRATION_TEMPLATES.brain_mask_2mm,
) -> QCOutputs:
    """Compute all QC metrics for a single functional run.

    Args:
        template_bold: Pre-denoising BOLD in template space.
        cleaned_bold: Post-denoising (nuisance-regressed) BOLD.
        motion_params: ``.1D`` file (six-column).
        rms_rel: ``_rel.rms`` file from MCFLIRT.
        bold_mask: Native BOLD brain mask.
        brain_mask: Anatomical brain mask (native space).
        bold_to_anat_matrix: BOLD-to-T1w affine matrix (from BBR).
        template_brain_mask: Brain mask warped to template space.
        sub: Subject ID.
        ses: Session label.
        task: Task label.
        run: Run number.
        start_tr: Number of initial TRs that were discarded.
        regressor_set: Nuisance regressor strategy name.
        mni_brain_mask_2mm: Brain mask for normalization QC (default: MNI152 2 mm).

    Returns:
        All QC outputs bundled in a :class:`QCOutputs` tuple.
    """
    _logger.info("Computing QC metrics")
    # 1. Load motion data
    rms_values = np.loadtxt(rms_rel)
    motion_data = np.loadtxt(motion_params)

    # 2. Motion QC metrics
    motion = motion_qc_metrics(rms_values, motion_data)

    # 3. Framewise displacement (needed for DVARS + pass/fail)
    fd = framewise_displacement_jenkinson(rms_values)

    # 4. Pre-denoising DVARS
    pre_img = nib.nifti1.load(template_bold)
    tmpl_mask_img = nib.nifti1.load(template_brain_mask)
    pre_data = pre_img.get_fdata()
    tmpl_mask = tmpl_mask_img.get_fdata()
    dvars_init = dvars_qc_metrics(pre_data, tmpl_mask, fd)
    del pre_img, pre_data, tmpl_mask_img

    # 5. Coregistration overlap (BOLD mask warped to anat space vs anat brain mask)
    bold_mask_in_anat = fsl.flirt(
        in_file=bold_mask,
        reference=brain_mask,
        out_file="bold_mask_in_anat.nii.gz",
        out_matrix_file="identity.mat",
        in_matrix_file=bold_to_anat_matrix,
        apply_xfm=True,
        interp="nearestneighbour",
    )
    bold_mask_arr = nib.nifti1.load(bold_mask_in_anat.out_file).get_fdata()
    brain_mask_arr = nib.nifti1.load(brain_mask).get_fdata()
    coreg = registration_qc_metrics(bold_mask_arr, brain_mask_arr)

    # 6. Normalization overlap (template brain mask vs MNI brain mask)
    #    Resample MNI mask to template grid if shapes differ.
    tmpl_brain_img = nib.nifti1.load(template_brain_mask)
    mni_mask_img = nib.nifti1.load(mni_brain_mask_2mm)
    if mni_mask_img.shape[:3] != tmpl_brain_img.shape[:3]:
        from nibabel.processing import resample_from_to

        mni_mask_img = resample_from_to(mni_mask_img, tmpl_brain_img, order=0)
    mni_mask_arr = mni_mask_img.get_fdata()
    tmpl_brain_arr = tmpl_brain_img.get_fdata()
    del tmpl_brain_img, mni_mask_img
    norm = registration_qc_metrics(tmpl_brain_arr, mni_mask_arr)

    qc_outputs = QCOutputs()
    for regressor in regressor_set:
        # 7. Post-denoising DVARS
        post_img = nib.nifti1.load(cleaned_bold[regressor])
        post_data = post_img.get_fdata()
        dvars_final = dvars_qc_metrics(post_data, tmpl_mask, fd)

        # 8. Assemble XCP QC row
        qc_outputs.metrics[regressor] = generate_xcp_qc(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            desc="RBC",
            regressors=regressor,
            space=TemplateSpace.MNI152NLIN2009CASYM,
            motion=motion,
            dvars_init=dvars_init,
            dvars_final=dvars_final,
            n_vols_removed=start_tr,
            coreg=coreg,
            norm=norm,
        )

        # 9. Write QC TSV
        qc_outputs.qc_file[regressor] = write_xcp_qc(
            qc_outputs.metrics[regressor],
            template_bold.parent
            / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_reg-{regressor}_qc.tsv",
        )

    # 10. RBC pass/fail
    qc_outputs.passed = passes_rbc_qc(fd, norm.cross_corr)

    return qc_outputs
