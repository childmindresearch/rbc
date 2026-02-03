# RBC Quality Control Metrics: Reimplementation Guide

This guide documents the QC metrics computed by C-PAC for the RBC pipeline and how to reimplement them independently.

**C-PAC Reference**: `CPAC/qc/xcp.py`, `CPAC/qc/qcmetrics.py`, `CPAC/generate_motion_statistics/generate_motion_statistics.py`

---

## Output Format

C-PAC generates a TSV file (`space-template_desc-xcp_quality.tsv`) with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| sub | str | Subject ID |
| ses | str | Session ID |
| task | str | Task ID |
| run | int | Run index |
| desc | str | Description |
| regressors | str | Nuisance regressor set name |
| space | str | Template space |
| meanFD | float | Mean framewise displacement (Jenkinson) |
| relMeansRMSMotion | float | Mean RMS of translation |
| relMaxRMSMotion | float | Max RMS of translation |
| meanDVInit | float | Mean DVARS before nuisance regression |
| meanDVFinal | float | Mean DVARS after nuisance regression |
| nVolCensored | int | Number of censored volumes |
| nVolsRemoved | int | Volumes removed (original - final) |
| motionDVCorrInit | float | Correlation(DVARS, FD) before regression |
| motionDVCorrFinal | float | Correlation(DVARS, FD) after regression |
| coregDice | float | BOLD→T1w coregistration Dice coefficient |
| coregJaccard | float | BOLD→T1w coregistration Jaccard index |
| coregCrossCorr | float | BOLD→T1w coregistration cross-correlation |
| coregCoverage | float | BOLD→T1w coregistration coverage |
| normDice | float | Template normalization Dice coefficient |
| normJaccard | float | Template normalization Jaccard index |
| normCrossCorr | float | Template normalization cross-correlation |
| normCoverage | float | Template normalization coverage |

---

## Motion Metrics

### 1. Framewise Displacement - Jenkinson (meanFD)

**Reference**: Jenkinson et al. 2002; `CPAC/generate_motion_statistics/generate_motion_statistics.py` lines 373-467

FD-J is calculated from either affine transformation matrices or RMS displacement files from MCFLIRT.

```python
import numpy as np

def calculate_FD_J_from_rms(rms_file):
    """
    Calculate FD-Jenkinson from MCFLIRT *_rel.rms output.

    Parameters
    ----------
    rms_file : str
        Path to *_rel.rms file from MCFLIRT

    Returns
    -------
    fdj : np.ndarray
        Framewise displacement array (first value is 0)
    """
    fdj = np.genfromtxt(rms_file)
    fdj = np.insert(fdj, 0, 0)  # Prepend 0 for first volume
    return fdj

# Mean FD for QC
mean_fd = np.mean(fdj)
```

### 2. Framewise Displacement - Power (alternative)

**Reference**: Power et al. 2012; `CPAC/generate_motion_statistics/generate_motion_statistics.py` lines 332-363

```python
import numpy as np

def calculate_FD_P(motion_params_file):
    """
    Calculate FD-Power from 6-parameter motion file.

    Parameters
    ----------
    motion_params_file : str
        Path to motion parameters file (6 columns: 3 rotation, 3 translation)
        Format: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z] per row

    Returns
    -------
    fd : np.ndarray
        Framewise displacement array
    """
    motion_params = np.genfromtxt(motion_params_file).T

    # Rotations (first 3 params) - convert to mm using 50mm sphere radius
    rotations = np.transpose(np.abs(np.diff(motion_params[0:3, :])))
    # Translations (last 3 params)
    translations = np.transpose(np.abs(np.diff(motion_params[3:6, :])))

    # FD = sum of translations + (50mm * pi/180) * sum of rotations
    fd = np.sum(translations, axis=1) + (50 * np.pi / 180) * np.sum(rotations, axis=1)
    fd = np.insert(fd, 0, 0)  # Prepend 0 for first volume

    return fd
```

### 3. RMS Motion (relMeansRMSMotion, relMaxRMSMotion)

**Reference**: `CPAC/qc/xcp.py` lines 262-269

```python
import numpy as np

def calculate_rms_motion(motion_params_file):
    """
    Calculate RMS of translation parameters.

    Parameters
    ----------
    motion_params_file : str
        Path to motion parameters (6 columns)

    Returns
    -------
    mean_rms : float
        Mean RMS motion
    max_rms : float
        Maximum RMS motion
    """
    mot = np.genfromtxt(motion_params_file).T

    # RMS of translation parameters (columns 3, 4, 5 for x, y, z translation)
    rms = np.sqrt(mot[3]**2 + mot[4]**2 + mot[5]**2)

    return np.mean(rms), np.max(rms)
```

---

## DVARS Metrics

### 4. DVARS Calculation (meanDVInit, meanDVFinal)

**Reference**: Power et al. 2012; `CPAC/generate_motion_statistics/generate_motion_statistics.py` lines 718-755

DVARS = Derivative of RMS VARiance over voxelS

```python
import numpy as np
import nibabel as nib

def calculate_DVARS(func_file, mask_file):
    """
    Calculate DVARS (temporal derivative of RMS variance).

    Parameters
    ----------
    func_file : str
        Path to 4D functional image
    mask_file : str
        Path to brain mask

    Returns
    -------
    dvars : np.ndarray
        DVARS timeseries (first value is 0)
    mean_dvars : float
        Mean DVARS
    """
    func_data = nib.load(func_file).get_fdata().astype(np.float32)
    mask_data = nib.load(mask_file).get_fdata().astype(bool)

    # Temporal difference (derivative)
    diff_data = np.diff(func_data, axis=3)

    # Square of differences
    squared_diff = np.square(diff_data)

    # Apply mask
    masked_data = squared_diff[mask_data]

    # RMS across voxels for each timepoint
    dvars = np.sqrt(np.mean(masked_data, axis=0))

    # Prepend 0 for first volume
    dvars = np.insert(dvars, 0, 0)

    return dvars, np.mean(dvars)
```

**Alternative using AFNI** (as C-PAC does):
```bash
3dTto1D -input func.nii.gz -mask mask.nii.gz -method dvars -prefix dvars.1D
```

### 5. Motion-DVARS Correlation (motionDVCorrInit, motionDVCorrFinal)

**Reference**: `CPAC/qc/xcp.py` lines 135-144

```python
import numpy as np

def calculate_motion_dvars_correlation(dvars_file, fdj_file):
    """
    Calculate correlation between DVARS and FD-Jenkinson.

    Note: DVARS has N-1 values (no derivative for first volume),
    so we correlate with FD[1:] (skipping first FD value).

    Parameters
    ----------
    dvars_file : str
        Path to DVARS 1D file
    fdj_file : str
        Path to FD-Jenkinson 1D file

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    """
    dvars = np.loadtxt(dvars_file)
    fdj = np.loadtxt(fdj_file)

    # DVARS is 1 shorter than FD (or both have leading 0)
    # If both have leading 0, skip it
    if len(dvars) == len(fdj):
        dvars = dvars[1:]
        fdj = fdj[1:]
    elif len(dvars) == len(fdj) - 1:
        fdj = fdj[1:]

    return np.corrcoef(dvars, fdj)[0, 1]
```

---

## Registration Quality Metrics

These metrics compare mask overlap to assess registration quality.

**Reference**: `CPAC/qc/qcmetrics.py`

### Required Inputs

| Metric Type | Input 1 | Input 2 |
|-------------|---------|---------|
| Coregistration | BOLD mask transformed to T1w space | T1w brain mask |
| Normalization | BOLD mask in template space | Template brain mask |

### 6. Dice Coefficient (coregDice, normDice)

```python
import numpy as np
import nibabel as nib

def dice_coefficient(mask1_file, mask2_file):
    """
    Calculate Dice coefficient between two binary masks.

    DC = 2|A ∩ B| / (|A| + |B|)

    Range: 0 (no overlap) to 1 (perfect overlap)
    """
    mask1 = nib.load(mask1_file).get_fdata().astype(bool)
    mask2 = nib.load(mask2_file).get_fdata().astype(bool)

    intersection = np.count_nonzero(mask1 & mask2)
    size1 = np.count_nonzero(mask1)
    size2 = np.count_nonzero(mask2)

    if size1 + size2 == 0:
        return 0.0

    return 2.0 * intersection / (size1 + size2)
```

### 7. Jaccard Index (coregJaccard, normJaccard)

```python
def jaccard_coefficient(mask1_file, mask2_file):
    """
    Calculate Jaccard coefficient between two binary masks.

    JC = |A ∩ B| / |A ∪ B|

    Range: 0 (no overlap) to 1 (perfect overlap)
    """
    mask1 = nib.load(mask1_file).get_fdata().astype(bool)
    mask2 = nib.load(mask2_file).get_fdata().astype(bool)

    intersection = np.count_nonzero(mask1 & mask2)
    union = np.count_nonzero(mask1 | mask2)

    if union == 0:
        return 0.0

    return intersection / union
```

### 8. Cross-Correlation (coregCrossCorr, normCrossCorr)

```python
def cross_correlation(mask1_file, mask2_file):
    """
    Calculate Pearson correlation between flattened binary masks.
    """
    mask1 = nib.load(mask1_file).get_fdata().astype(bool).flatten()
    mask2 = nib.load(mask2_file).get_fdata().astype(bool).flatten()

    return np.corrcoef(mask1, mask2)[0, 1]
```

### 9. Coverage Index (coregCoverage, normCoverage)

```python
def coverage(mask1_file, mask2_file):
    """
    Calculate coverage: intersection / smaller mask.

    Measures how much of the smaller mask is covered by the overlap.
    """
    mask1 = nib.load(mask1_file).get_fdata().astype(bool)
    mask2 = nib.load(mask2_file).get_fdata().astype(bool)

    intersection = np.count_nonzero(mask1 & mask2)
    smaller_size = min(np.count_nonzero(mask1), np.count_nonzero(mask2))

    if smaller_size == 0:
        return 0.0

    return intersection / smaller_size
```

---

## Volume Metrics

### 10. Censored Volumes (nVolCensored)

Number of volumes flagged for censoring based on motion/DVARS thresholds.

```python
def count_censored_volumes(fd, dvars, fd_threshold=0.2, dvars_threshold=None):
    """
    Count volumes exceeding motion thresholds.

    Parameters
    ----------
    fd : np.ndarray
        Framewise displacement timeseries
    dvars : np.ndarray
        DVARS timeseries
    fd_threshold : float
        FD threshold (RBC uses 0.2 mm)
    dvars_threshold : float, optional
        DVARS threshold

    Returns
    -------
    n_censored : int
        Number of volumes exceeding thresholds
    censor_indices : list
        Indices of censored volumes
    """
    censor_mask = fd > fd_threshold

    if dvars_threshold is not None:
        censor_mask |= dvars > dvars_threshold

    return np.sum(censor_mask), np.where(censor_mask)[0].tolist()
```

### 11. Volumes Removed (nVolsRemoved)

```python
def volumes_removed(original_func, final_func):
    """
    Calculate difference in volume count between original and processed data.
    """
    orig_nvols = nib.load(original_func).shape[3]
    final_nvols = nib.load(final_func).shape[3]
    return orig_nvols - final_nvols
```

---

## Complete QC Function

```python
import numpy as np
import nibabel as nib
import pandas as pd

def generate_xcp_qc(
    sub, ses, task, run, desc, regressors, space,
    original_func, final_func,
    motion_params_file, fd_jenkinson_file,
    dvars_before_file, dvars_after_file,
    bold2t1w_mask, t1w_mask,
    bold2template_mask, template_mask,
    censor_indices=None
):
    """
    Generate XCP-style QC metrics TSV.

    Returns
    -------
    qc_dict : dict
        Dictionary of all QC metrics
    """
    # Motion metrics
    fdj = np.loadtxt(fd_jenkinson_file)
    mean_fd = np.mean(fdj)

    mot = np.genfromtxt(motion_params_file).T
    rms = np.sqrt(mot[3]**2 + mot[4]**2 + mot[5]**2)

    # DVARS
    dvars_init = np.loadtxt(dvars_before_file)
    dvars_final = np.loadtxt(dvars_after_file)

    # Motion-DVARS correlation
    def dvcorr(dvars, fd):
        if len(dvars) == len(fd):
            return np.corrcoef(dvars[1:], fd[1:])[0, 1]
        return np.corrcoef(dvars, fd[1:])[0, 1]

    # Registration metrics
    def load_mask(f):
        return nib.load(f).get_fdata().astype(bool)

    def dice(m1, m2):
        inter = np.count_nonzero(m1 & m2)
        return 2.0 * inter / (np.count_nonzero(m1) + np.count_nonzero(m2))

    def jaccard(m1, m2):
        return np.count_nonzero(m1 & m2) / np.count_nonzero(m1 | m2)

    def crosscorr(m1, m2):
        return np.corrcoef(m1.flatten(), m2.flatten())[0, 1]

    def coverage(m1, m2):
        inter = np.count_nonzero(m1 & m2)
        return inter / min(np.count_nonzero(m1), np.count_nonzero(m2))

    # Load masks
    coreg_m1, coreg_m2 = load_mask(bold2t1w_mask), load_mask(t1w_mask)
    norm_m1, norm_m2 = load_mask(bold2template_mask), load_mask(template_mask)

    # Volume counts
    orig_nvols = nib.load(original_func).shape[3]
    final_nvols = nib.load(final_func).shape[3]

    return {
        'sub': sub, 'ses': ses, 'task': task, 'run': run,
        'desc': desc, 'regressors': regressors, 'space': space,
        'meanFD': mean_fd,
        'relMeansRMSMotion': np.mean(rms),
        'relMaxRMSMotion': np.max(rms),
        'meanDVInit': np.mean(dvars_init),
        'meanDVFinal': np.mean(dvars_final),
        'nVolCensored': len(censor_indices) if censor_indices else 0,
        'nVolsRemoved': orig_nvols - final_nvols,
        'motionDVCorrInit': dvcorr(dvars_init, fdj),
        'motionDVCorrFinal': dvcorr(dvars_final, fdj),
        'coregDice': dice(coreg_m1, coreg_m2),
        'coregJaccard': jaccard(coreg_m1, coreg_m2),
        'coregCrossCorr': crosscorr(coreg_m1, coreg_m2),
        'coregCoverage': coverage(coreg_m1, coreg_m2),
        'normDice': dice(norm_m1, norm_m2),
        'normJaccard': jaccard(norm_m1, norm_m2),
        'normCrossCorr': crosscorr(norm_m1, norm_m2),
        'normCoverage': coverage(norm_m1, norm_m2),
    }
```

---

## RBC Recommended Thresholds

From the RBC paper (Shafiei et al., Neuron 2025):

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **Median FD** | ≤ 0.2 mm | Pass |
| **normCrossCorr** | ≥ 0.8 | Adequate registration |

**Combined QC**: A scan passes functional QC if **both** conditions are met.

```python
def passes_rbc_qc(qc_dict, fd_timeseries):
    """
    Apply RBC recommended QC thresholds.

    Parameters
    ----------
    qc_dict : dict
        Output from generate_xcp_qc
    fd_timeseries : np.ndarray
        Full FD timeseries (for median calculation)

    Returns
    -------
    passes : bool
        True if scan passes QC
    """
    median_fd = np.median(fd_timeseries)
    norm_cc = qc_dict['normCrossCorr']

    return (median_fd <= 0.2) and (norm_cc >= 0.8)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Array operations |
| nibabel | ≥3.0 | NIfTI I/O |
| pandas | ≥1.0 | TSV output |

**Optional AFNI tools**:
- `3dTto1D` - DVARS calculation (alternative to Python implementation)

---

## Notes

1. **FD-Jenkinson vs FD-Power**: RBC uses FD-Jenkinson (from MCFLIRT) for QC. FD-Power is an alternative method.

2. **DVARS timing**: DVARS has N-1 values for N volumes (temporal derivative). Both C-PAC implementations prepend 0 to align with FD.

3. **Registration masks**: For coregistration metrics, you need the BOLD brain mask transformed to T1w space, not just the native BOLD mask.

4. **Median vs Mean FD**: RBC recommends median FD ≤ 0.2 for exclusion, but `meanFD` is stored in the QC file. Calculate median from the full timeseries for exclusion decisions.
