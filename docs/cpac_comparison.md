# C-PAC RBC Pipeline: Implementation Notes

Findings from cross-referencing C-PAC source code and actual RBC pipeline run logs.

These notes document differences observed between RBC's implementation and the C-PAC reference. Items under **Bugs** are numerical errors or spec violations that affect output correctness regardless of intent. Items under **Divergences** are differences that may reflect intentional design choices in C-PAC (final classification is the RBC team's interpretation based on code reading).

**Sources:**
- C-PAC code: `CPAC/anat_preproc/anat_preproc.py`, `CPAC/anat_preproc/ants.py`, `CPAC/pipeline/cpac_pipeline.py`, `CPAC/func_preproc/func_preproc.py`, `CPAC/alff/alff.py`, `CPAC/qc/xcp.py`
- C-PAC RBC run logs (ds000001, sub-01_ses-1)

---

## Bugs

### 4. Bandpass filtering uses wrong TR from resampled NIfTI header

C-PAC's `frequency_filter` node calls `bandpass_voxels` (in `CPAC/nuisance/bandpass.py`) which reads the TR from the NIfTI header of the template-space residuals image. After ANTs resampling to template space, the pixdim[4] (TR) in the NIfTI header is set to 1.0 instead of the original repetition time (2.0 for ds000001).

Verified from the working directory:
```
tests/data/cpac_outputs/ds000001/working/.../nuisance_regression_template_36-parameter_212/
    .../frequency_filter/_inputs.pklz
    -> realigned_file: .../residuals.nii.gz   (header has TR=1.0)
    -> bandpass_freqs: [0.01, 0.1]
    -> sample_period: (not connected, reads from nifti)
```

The `ideal_bandpass` function uses `sample_period` to compute FFT frequency bin indices. With TR=1.0 instead of 2.0, the effective passband is 0.005-0.05 Hz (half the intended 0.01-0.1 Hz range). This affects both the BOLD bandpass filtering and the regressor bandpass filtering.

Confirmed by reproducing the filter: applying `ideal_bandpass` with TR=1.0 to C-PAC's raw regressors produces r=1.000 correlation with C-PAC's exported filtered regressors, while TR=2.0 (correct) produces r=0.758.

This bug affects all outputs downstream of the frequency filter: cleaned BOLD, ReHo, timeseries extraction, and connectivity matrices. ALFF/fALFF are also affected since they are computed from the bandpassed BOLD.

---

### 5. DVFinal metrics computed from pre-regression BOLD (same as DVInit)

The XCP-style QC file reports `meanDVFinal` and `motionDVCorrFinal` as post-regression DVARS metrics (per the XCP dictionary: "correlation of RMS and DVARS after regression"). However, inspecting the C-PAC source code (`xcp.py`) reveals that both `DVInit` and `DVFinal` are computed from the same pre-nuisance regression BOLD input.

This was confirmed empirically: `meanDVInit = meanDVFinal` and `motionDVCorrInit = motionDVCorrFinal` for every subject across all three datasets (HBN, PNC, NKI) during the small benchmarking study.

In contrast, RBC correctly computes `DVFinal` from the post-regression bandpass-filtered BOLD (`cleaned_bold`), as intended.

---

## Divergences

### 1. N4 bias correction output discarded (may be unintentional)

ANTs brain extraction (`antsBrainExtraction.sh`) runs N4 bias field correction internally (both an initial `inu_n4` and a refined `inu_n4_final`), producing a bias-corrected T1w. However, the downstream `brain_extraction` node (`anat_preproc.py:1925-1936`) applies the brain mask to the *original* (pre-N4) T1w via `3dcalc`, discarding the corrected output entirely.

From the logs:
```
3dcalc -a .../anat_reorient_0/sub-01_T1w_resample.nii.gz
       -b .../anat_skullstrip_ants/atropos_wf/.../mask_xform.nii.gz
       -expr "a*step(b)"
```

The `-a` input is the reoriented T1w, not the `inu_n4_final` output. The brain mask itself is valid (derived from N4-corrected data), but all downstream steps (segmentation, registration, etc.) operate on uncorrected intensities.

The standalone `n4_bias_field_correction` config option is also `Off` by default and not overridden by the RBC config, so no bias correction is applied at any stage.

---

### 2. Despiking runs in template space

The RBC config sets `despike.space: template`. The logs confirm `3dDespike` ran on already-resampled template-space data, after single-step resampling and merging. Motion estimation and slice-timing correction both operate on non-despiked data, which negates the purpose of despiking (removing outlier timepoints that bias motion estimation).

Despiking should also happen before resampling for signal processing reasons: in native space, each voxel's timeseries reflects a consistent tissue location, so outlier detection operates on clean per-voxel signals. After resampling, each voxel is an interpolated blend of neighboring native voxels, which smooths out the sharp spikes that despiking is designed to catch and mixes signals across tissue boundaries.

This also breaks the `calculate_motion_first` flag. The config inherits `calculate_motion_first: On` from `fmriprep-options`, which should estimate motion on raw (pre-STC) data. In `cpac_pipeline.py:1199-1210`, when this flag is true, the block order is `func_init + func_motion + func_preproc + [func_motion_correct_only] + ...`. But `func_preproc_blocks` includes both despike and STC, and since despike was deferred to template space, the block effectively only contained STC. The logs show motion estimation (`3dvolreg` for reference, `mcflirt` for correction) ran on post-STC data instead of raw data.

---

### 3. ALFF uses temporal std instead of mean FFT amplitude

C-PAC computes ALFF as the temporal standard deviation of a bandpass-filtered signal (`3dBandpass` followed by `3dTstat -stdev`), and fALFF as the ratio of filtered to unfiltered standard deviation. This differs from the original Zang et al. 2007 definition, which uses the arithmetic mean of FFT amplitude coefficients within the frequency band. The two approaches are related (both measure low-frequency power) but produce numerically different maps.

---

### 6. Motion correction runs twice (pre-STC for regressors, post-STC for spatial application)

C-PAC's logs show two separate mcflirt runs:
- One on **pre-STC** data, used to produce motion parameter regressors and derivative-space outputs.
- One on **post-STC** data, whose per-volume `.mat` transforms are applied alongside BBR and anat-to-template warps in the single-step template resampling, and whose corrected BOLD is also used for masking and BBR estimation.

RBC runs motion correction once: motion is estimated on pre-STC (despiked) data, and those `.mat` transforms are applied to STC'd BOLD in the single-step resample (see `src/rbc/core/functional/resampling.py:resample_bold_to_template`). The same motion parameters drive both the regressors and the spatial application.

C-PAC's duplication is not numerically incorrect — the single-step resampling still uses one interpolation pass per volume — but estimating motion twice on differently-processed data risks subtle inconsistencies between the regressor motion params and the spatial transforms actually applied.
