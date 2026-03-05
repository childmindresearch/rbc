# Data Dictionary

This document describes every output file the RBC pipeline produces. All outputs are written to a BIDS derivatives directory (`derivatives/rbc/`) organized by subject, session, and data type.

## BIDS filename conventions

RBC output filenames follow the [BIDS derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) naming scheme. Each filename is built from key-value pairs called **entities**, separated by underscores:

| Entity | Meaning | Example |
|--------|---------|---------|
| `sub-` | Subject identifier | `sub-01` |
| `ses-` | Session label (optional) | `ses-baseline` |
| `task-` | Task performed during fMRI | `task-rest` |
| `run-` | Run index when a scan is repeated | `run-1` |
| `space-` | Coordinate space the image is aligned to | `space-MNI152NLin6ASym` |
| `desc-` | Free-text description of the file content | `desc-brain` |
| `reg-` | Nuisance regression strategy applied | `reg-36-parameter` |
| `atlas-` | Brain parcellation used | `atlas-schaefer200` |
| `from-` / `to-` | Source and target spaces of a transform | `from-T1w_to-template` |

Entities in square brackets (e.g., `[_ses-{ses}]`) are included only when the dataset contains that information.

---

## Anatomical outputs

Produced by `rbc anatomical`. These are structural (T1-weighted) processing results.

| File | Suffix | Description | Created by | Format |
|------|--------|-------------|------------|--------|
| `*_desc-brain_T1w.nii.gz` | `T1w` | Skull-stripped, bias-corrected brain image | ANTs brain extraction + N4 bias correction | 3D NIfTI |
| `*_desc-T1w_mask.nii.gz` | `mask` | Binary mask delineating brain tissue | ANTs brain extraction | 3D NIfTI, binary mask |
| `*_desc-csf_mask.nii.gz` | `mask` | Cerebrospinal fluid partial volume map | FSL FAST tissue segmentation | 3D NIfTI, binary mask |
| `*_desc-gm_mask.nii.gz` | `mask` | Gray matter partial volume map | FSL FAST tissue segmentation | 3D NIfTI, binary mask |
| `*_desc-wm_mask.nii.gz` | `mask` | White matter partial volume map | FSL FAST tissue segmentation | 3D NIfTI, binary mask |
| `*_desc-wmBBR_mask.nii.gz` | `mask` | White matter boundary used for boundary-based registration (BBR) of functional data to anatomy | FSL WM boundary extraction | 3D NIfTI, binary mask |
| `*_from-T1w_to-template_mode-image_xfm.nii.gz` | `xfm` | Nonlinear warp field mapping subject anatomy to MNI152NLin6ASym template space | ANTs registration | 3D NIfTI, displacement field |
| `*_from-template_to-T1w_mode-image_xfm.nii.gz` | `xfm` | Inverse warp field mapping MNI152NLin6ASym template space back to subject anatomy | ANTs registration | 3D NIfTI, displacement field |

---

## Functional outputs

Produced by `rbc functional`. These are functional MRI (BOLD) processing results. Requires anatomical outputs.

| File | Suffix | Description | Created by | Format |
|------|--------|-------------|------------|--------|
| `*_sbref.nii.gz` | `sbref` | Single-volume reference image used for motion estimation | Extracted from BOLD timeseries | 3D NIfTI |
| `*_desc-preproc_bold.nii.gz` | `bold` | Motion-corrected BOLD timeseries in native subject space (after despiking and slice-timing correction) | FSL MCFLIRT | 4D NIfTI |
| `*_desc-motionParams_motion.1D` | `motion` | Six rigid-body motion parameters (3 translations, 3 rotations) estimated during motion correction | FSL MCFLIRT | Text, 6-column 1D file |
| `*_desc-relsDisplacement_motion.rms` | `motion` | Frame-to-frame relative root-mean-square displacement over time | FSL MCFLIRT | Text, single-column RMS file |
| `*_desc-maxDisplacement_motion.rms` | `motion` | Absolute root-mean-square displacement of each volume relative to the reference | FSL MCFLIRT | Text, single-column RMS file |
| `*_desc-brain_mask.nii.gz` | `mask` | Binary brain mask in native BOLD space | FSL BET | 3D NIfTI, binary mask |
| `*_desc-linear_from-bold_to-T1w_mode-image_xfm.mat` | `xfm` | Affine transformation matrix aligning BOLD to the T1w anatomical image | ANTs boundary-based registration | 4x4 affine matrix |
| `*_space-MNI152NLin6ASym_desc-preproc_bold.nii.gz` | `bold` | BOLD timeseries resampled to MNI152NLin6ASym template space in a single interpolation step (before denoising) | ANTs resampling | 4D NIfTI |
| `*_space-MNI152NLin6ASym_desc-bold_mask.nii.gz` | `mask` | Brain mask warped to template space at the BOLD resolution | ANTs resampling | 3D NIfTI, binary mask |
| `*_space-MNI152NLin6ASym_desc-preproc_reg-{regressor}_bold.nii.gz` | `bold` | Denoised BOLD timeseries in template space after nuisance regression and bandpass filtering. `{regressor}` is `36-parameter` or `aCompCor` | Nuisance regression | 4D NIfTI |
| `*_desc-{regressor}_regressors.1D` | `regressors` | Nuisance regressor matrix used for denoising. `{regressor}` is `36-parameter` or `aCompCor` | Computed from motion parameters and tissue masks | Text, multi-column 1D file |

---

## Metrics outputs

Produced by `rbc metrics`. Voxel-wise and region-wise summary measures computed from the denoised BOLD data. All are in MNI152NLin6ASym template space and tagged with the regression strategy used.

| File | Suffix | Description | Created by | Format |
|------|--------|-------------|------------|--------|
| `*_reg-{regressor}_alff.nii.gz` | `alff` | Amplitude of Low-Frequency Fluctuations — sum of spectral power in the 0.01–0.1 Hz band at each voxel | Frequency-domain analysis | 3D NIfTI |
| `*_desc-smooth_reg-{regressor}_alff.nii.gz` | `alff` | Spatially smoothed ALFF map | Gaussian kernel smoothing | 3D NIfTI |
| `*_desc-smoothZstd_reg-{regressor}_alff.nii.gz` | `alff` | Z-scored (standardized) smoothed ALFF, normalized within the brain mask | Z-score normalization | 3D NIfTI |
| `*_reg-{regressor}_falff.nii.gz` | `falff` | Fractional ALFF — ratio of low-frequency power (0.01–0.1 Hz) to total power at each voxel | Frequency-domain analysis | 3D NIfTI |
| `*_desc-smooth_reg-{regressor}_falff.nii.gz` | `falff` | Spatially smoothed fALFF map | Gaussian kernel smoothing | 3D NIfTI |
| `*_desc-smoothZstd_reg-{regressor}_falff.nii.gz` | `falff` | Z-scored smoothed fALFF, normalized within the brain mask | Z-score normalization | 3D NIfTI |
| `*_reg-{regressor}_reho.nii.gz` | `reho` | Regional Homogeneity — Kendall's W measuring local synchronization in a 26-voxel neighborhood | AFNI 3dReHo | 3D NIfTI |
| `*_desc-smooth_reg-{regressor}_reho.nii.gz` | `reho` | Spatially smoothed ReHo map | Gaussian kernel smoothing | 3D NIfTI |
| `*_desc-smoothZstd_reg-{regressor}_reho.nii.gz` | `reho` | Z-scored smoothed ReHo, normalized within the brain mask | Z-score normalization | 3D NIfTI |
| `*_atlas-{atlas}_desc-mean_reg-{regressor}_timeseries.tsv` | `timeseries` | Mean BOLD signal averaged within each region of the specified atlas. Atlases: `schaefer_200`, `schaefer_300`, `schaefer_400`, `schaefer_1000`, `aal` | Region-wise averaging | TSV, regions x timepoints |
| `*_atlas-{atlas}_desc-pearson_reg-{regressor}_correlations.tsv` | `correlations` | Pairwise Pearson correlation matrix between all atlas region timeseries | Pearson correlation | TSV, regions x regions |

All metrics files include `space-MNI152NLin6ASym` in the filename (omitted from the table for readability).

---

## QC outputs

Produced by `rbc qc`. A single summary file per functional run containing quality control metrics in [XCP-D](https://xcp-d.readthedocs.io/) format.

| File | Suffix | Description | Created by | Format |
|------|--------|-------------|------------|--------|
| `*_space-MNI152NLin6ASym_desc-xcp_reg-{regressor}_quality.tsv` | `quality` | Single-row TSV with 24 quality control metrics (see columns below) | QC aggregation | TSV, 1 row |

### QC columns

| Column | Type | Description |
|--------|------|-------------|
| `sub` | string | Subject identifier |
| `ses` | string | Session label |
| `task` | string | Task label |
| `run` | integer | Run number |
| `desc` | string | Pipeline variant description |
| `regressors` | string | Nuisance regression strategy name |
| `space` | string | Coordinate space (`MNI152NLin6ASym`) |
| `meanFD` | float | Mean framewise displacement in mm (Jenkinson method) |
| `relMeansRMSMotion` | float | Mean relative RMS of translation parameters |
| `relMaxRMSMotion` | float | Maximum relative RMS of translation parameters |
| `nVolCensored` | integer | Number of volumes exceeding the FD threshold (0.2 mm) |
| `meanDVInit` | float | Mean DVARS before denoising |
| `meanDVFinal` | float | Mean DVARS after denoising |
| `nVolsRemoved` | integer | Number of non-steady-state volumes removed |
| `motionDVCorrInit` | float | Correlation between motion and DVARS before denoising |
| `motionDVCorrFinal` | float | Correlation between motion and DVARS after denoising |
| `coregDice` | float | Dice coefficient for BOLD-to-T1w coregistration overlap |
| `coregJaccard` | float | Jaccard index for BOLD-to-T1w coregistration overlap |
| `coregCrossCorr` | float | Cross-correlation for BOLD-to-T1w coregistration |
| `coregCoverage` | float | Coverage metric for BOLD-to-T1w coregistration |
| `normDice` | float | Dice coefficient for T1w-to-MNI normalization overlap |
| `normJaccard` | float | Jaccard index for T1w-to-MNI normalization overlap |
| `normCrossCorr` | float | Cross-correlation for T1w-to-MNI normalization |
| `normCoverage` | float | Coverage metric for T1w-to-MNI normalization |

**Pass/fail criteria:** A run passes RBC quality thresholds when both `median(FD) <= 0.2 mm` and `normCrossCorr >= 0.8`.

---

## Longitudinal outputs

Produced by `rbc longitudinal`. Anatomical outputs aligned to a subject-specific longitudinal template built from multiple sessions. Note: longitudinal functional processing is not yet implemented.

| File | Suffix | Description | Created by | Format |
|------|--------|-------------|------------|--------|
| `*_space-longitudinal_desc-brain_T1w.nii.gz` | `T1w` | Skull-stripped brain image aligned to the subject's longitudinal template | ANTs registration to longitudinal template | 3D NIfTI |
| `*_space-longitudinal_desc-T1w_mask.nii.gz` | `mask` | Brain mask in longitudinal template space | ANTs registration to longitudinal template | 3D NIfTI, binary mask |
| `*_space-longitudinal_desc-csf_mask.nii.gz` | `mask` | CSF mask in longitudinal template space | ANTs registration to longitudinal template | 3D NIfTI, binary mask |
| `*_space-longitudinal_desc-gm_mask.nii.gz` | `mask` | Gray matter mask in longitudinal template space | ANTs registration to longitudinal template | 3D NIfTI, binary mask |
| `*_space-longitudinal_desc-wm_mask.nii.gz` | `mask` | White matter mask in longitudinal template space | ANTs registration to longitudinal template | 3D NIfTI, binary mask |
| `*_from-T1w_to-longitudinal_mode-image_xfm.nii.gz` | `xfm` | Warp field mapping subject anatomy to the longitudinal template | ANTs registration | 3D NIfTI, displacement field |
| `*_from-longitudinal_to-T1w_mode-image_xfm.nii.gz` | `xfm` | Inverse warp field mapping longitudinal template back to subject anatomy | ANTs registration | 3D NIfTI, displacement field |
