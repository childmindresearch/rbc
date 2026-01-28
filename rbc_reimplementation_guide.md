# RBC Pipeline Reimplementation Guide

A step-by-step guide for reimplementing the C-PAC RBC pipeline in native Python without Nipype.

## Overview

The RBC pipeline processes data in this order:
1. Anatomical preprocessing → brain extraction → segmentation → registration
2. Functional preprocessing → motion correction → masking → coregistration
3. Distortion correction (if fieldmaps available)
4. Transform to template space (single-step resampling)
5. Nuisance regression
6. Derivatives (ALFF, ReHo, Centrality)
7. Timeseries extraction

---

## Step 1: Anatomical Initialization

**Purpose**: Reorient T1w to RPI orientation

**Tool**: AFNI `3drefit` + `3dresample`

**C-PAC Reference**: `CPAC/anat_preproc/anat_preproc.py` lines 1273-1295

```python
# What C-PAC does:
anat_deoblique = afni.Refit(deoblique=True)
anat_reorient = afni.Resample(orientation='RPI', outputtype='NIFTI_GZ')
```

**Native Implementation**:
```bash
3drefit -deoblique input.nii.gz
3dresample -orient RPI -prefix output.nii.gz -input input.nii.gz
```

**Outputs**: 
- `desc-reorient_T1w` (reoriented T1w)

---

## Step 2: Brain Extraction (niworkflows-ANTs)

**Purpose**: Skull-strip the T1w image

**Tool**: ANTs `antsBrainExtraction.sh` (via niworkflows implementation)

**C-PAC Reference**: `CPAC/anat_preproc/ants.py` lines 62-350

**Key Steps**:
1. Truncate intensity (0.01-0.999 percentile)
2. N4 bias correction (initial)
3. Resample to 4mm for initial registration
4. Affine initialization with `antsAI`
5. Full registration (Rigid → Affine → SyN) to OASIS template
6. Map brain probability mask to subject space
7. Threshold at 0.5, dilate, get largest component
8. Final N4 bias correction with brain mask
9. Apply mask

**Parameters**:
```python
# Template paths
template = '/ants_template/oasis/T_template0.nii.gz'
mask = '/ants_template/oasis/T_template0_BrainCerebellumProbabilityMask.nii.gz'
regmask = '/ants_template/oasis/T_template0_BrainCerebellumRegistrationMask.nii.gz'

# N4 parameters
n4_iterations = [50, 50, 50, 50]
n4_convergence = 1e-7
n4_shrink = 4
n4_bspline_distance = 200

# Registration (see ants.py lines 231-238 for full config)
# Uses: CPAC/anat_preproc/data/antsBrainExtraction_precise.json
```

**Native Implementation**:
```bash
# Option 1: Use antsBrainExtraction.sh directly
antsBrainExtraction.sh -d 3 -a input.nii.gz -e template.nii.gz \
    -m probability_mask.nii.gz -o output_prefix

# Option 2: Use ANTsPy
import ants
img = ants.image_read('input.nii.gz')
result = ants.brain_extraction(img, template, ...)
```

**Outputs**:
- `space-T1w_desc-brain_mask` (brain mask)
- `bias_corrected` (N4-corrected full head) ← **USE THIS, not the original**

**⚠️ BUG IN C-PAC**: The subsequent `brain_extraction` step (line 2041) uses the original uncorrected image instead of the N4-corrected output. Your reimplementation should use the bias-corrected image.

---

## Step 3: Brain Extraction (Apply Mask)

**Purpose**: Apply brain mask to get skull-stripped brain

**C-PAC Reference**: `CPAC/anat_preproc/anat_preproc.py` lines 2016-2054

```python
# C-PAC (buggy - uses uncorrected input):
anat_skullstrip = afni.Calc(expr='a*step(b)')
# input a = desc-head_T1w (original, uncorrected)
# input b = brain_mask

# CORRECT implementation should use:
# input a = bias_corrected (from N4)
```

**Native Implementation**:
```bash
3dcalc -a bias_corrected.nii.gz -b brain_mask.nii.gz -expr 'a*step(b)' -prefix brain.nii.gz
```

**Outputs**:
- `desc-brain_T1w` (skull-stripped brain)

---

## Step 4: Tissue Segmentation (FSL-FAST)

**Purpose**: Segment brain into CSF, GM, WM

**Tool**: FSL `fast`

**C-PAC Reference**: `CPAC/seg_preproc/seg_preproc.py` lines 461-600

```python
segment = fsl.FAST(
    img_type=1,           # T1
    segments=True,        # output binary masks
    probability_maps=True # output probability maps
)
```

**Parameters**:
```yaml
# Thresholding
CSF_threshold: 0.95
WM_threshold: 0.95
GM_threshold: 0.95
use_priors: Off
```

**Native Implementation**:
```bash
fast -t 1 -n 3 -g -p -o output_prefix brain.nii.gz

# Then threshold probability maps:
fslmaths output_pve_0.nii.gz -thr 0.95 -bin csf_mask.nii.gz  # CSF
fslmaths output_pve_1.nii.gz -thr 0.95 -bin gm_mask.nii.gz   # GM
fslmaths output_pve_2.nii.gz -thr 0.95 -bin wm_mask.nii.gz   # WM
```

**Outputs**:
- `label-CSF_mask`, `label-GM_mask`, `label-WM_mask` (binary masks)
- `label-CSF_probseg`, `label-GM_probseg`, `label-WM_probseg` (probability maps)

---

## Step 5: Anatomical Registration (ANTs)

**Purpose**: Register T1w brain to MNI template

**Tool**: ANTs `antsRegistration`

**C-PAC Reference**: `CPAC/registration/registration.py` lines 1188-1300

**Parameters** (from resolved config):
```yaml
registration:
  - Rigid:
      iterations: 100x100
      metric: MI (32 bins, 25% sampling, Regular)
      shrink_factors: 2x1
      smoothing_sigmas: 2.0x1.0vox
      gradient_step: 0.05
  - Affine:
      iterations: 100x100
      metric: MI
      shrink_factors: 2x1
      smoothing_sigmas: 1.0x0.0vox
      gradient_step: 0.08
  - SyN:
      iterations: 100x70x50x20
      metric: CC (radius=4)
      shrink_factors: 8x4x2x1
      smoothing_sigmas: 3.0x2.0x1.0x0.0vox
      gradient_step: 0.1
      update_field_variance: 3.0
      total_field_variance: 0.0

# Template
fixed = '$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz'
interpolation = 'LanczosWindowedSinc'
```

**Native Implementation**:
```bash
antsRegistration --dimensionality 3 --float 1 \
    --output [output_prefix,outputWarped.nii.gz] \
    --interpolation LanczosWindowedSinc \
    --initial-moving-transform [fixed.nii.gz,moving.nii.gz,1] \
    --transform Rigid[0.05] \
    --metric MI[fixed.nii.gz,moving.nii.gz,1,32,Regular,0.25] \
    --convergence [100x100,1e-6,20] \
    --shrink-factors 2x1 \
    --smoothing-sigmas 2.0x1.0vox \
    --transform Affine[0.08] \
    --metric MI[fixed.nii.gz,moving.nii.gz,1,32,Regular,0.25] \
    --convergence [100x100,1e-6,20] \
    --shrink-factors 2x1 \
    --smoothing-sigmas 1.0x0.0vox \
    --transform SyN[0.1,3.0,0.0] \
    --metric CC[fixed.nii.gz,moving.nii.gz,1,4] \
    --convergence [100x70x50x20,1e-6,10] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3.0x2.0x1.0x0.0vox
```

**Outputs**:
- `from-T1w_to-template_mode-image_xfm` (forward transforms: affine + warp)
- `from-template_to-T1w_mode-image_xfm` (inverse transforms)

---

## Step 6: Functional Initialization

**Purpose**: Reorient, scale, truncate BOLD

**C-PAC Reference**: `CPAC/func_preproc/func_preproc.py` lines 526-400

### 6a. Reorient
```python
func_deoblique = afni.Refit(deoblique=True)
func_reorient = afni.Resample(orientation='RPI')
```

### 6b. Scale
```python
# Divide voxel dimensions by 10 (scaling factor)
func_scale = afni.Refit()  # With scaling
```

### 6c. Truncate (Remove first 2 TRs)
```python
func_drop_trs = afni.Calc(expr='a', start_idx=2)  # start_tr=2
```

**Native Implementation**:
```bash
# Reorient
3drefit -deoblique func.nii.gz
3dresample -orient RPI -prefix func_rpi.nii.gz -input func.nii.gz

# Truncate first 2 TRs
3dcalc -a 'func_rpi.nii.gz[2..$]' -expr 'a' -prefix func_trunc.nii.gz
```

**Outputs**:
- `desc-reorient_bold` (reoriented)
- `bold` (truncated)

---

## Step 7: Motion Reference Generation

**Purpose**: Create reference volume for motion correction (fmriprep-style)

**C-PAC Reference**: `CPAC/func_preproc/func_motion.py` lines 249-330

**Method** (`fmriprep_reference`):
1. Extract middle volume
2. Run preliminary motion correction
3. Compute weighted average favoring low-motion volumes

```python
# Get middle volume
func_get_RPI = afni.TStat(options='-median', outputtype='NIFTI_GZ')

# Or specific volume
func_get_RPI = afni.Calc(expr='a', single_idx=middle_vol)
```

**Native Implementation**:
```bash
# Get number of volumes
nvols=$(3dinfo -nv func.nii.gz)
mid=$((nvols / 2))

# Extract middle volume
3dcalc -a "func.nii.gz[$mid]" -expr 'a' -prefix ref.nii.gz
```

**Outputs**:
- `sbref` (reference volume)

---

## Step 8: Motion Estimation and Correction (mcflirt)

**Purpose**: Estimate and correct head motion

**Tool**: FSL `mcflirt`

**C-PAC Reference**: `CPAC/func_preproc/func_motion.py` lines 515-565

```python
func_motion_correct = fsl.MCFLIRT(
    save_mats=True,
    save_plots=True,
    ref_file=reference_volume
)
```

**Native Implementation**:
```bash
mcflirt -in func.nii.gz -reffile ref.nii.gz -out func_mc \
    -mats -plots -report
```

**Outputs**:
- `desc-motion_bold` (motion-corrected timeseries)
- `desc-movementParameters_motion` (6 motion parameters: 3 rotation, 3 translation)
- Motion matrices (for later transform composition)

---

## Step 9: Despiking

**Purpose**: Remove intensity spikes

**Tool**: AFNI `3dDespike`

**C-PAC Reference**: `CPAC/func_preproc/func_preproc.py` lines 621-695

```python
despike = afni.Despike(outputtype='NIFTI_GZ')
```

**Native Implementation**:
```bash
3dDespike -prefix func_despike.nii.gz func.nii.gz
```

**Note**: RBC runs despiking TWICE - once early (native space) and once late (template space). You may want to evaluate if both are necessary.

**Outputs**:
- `desc-despike_bold`

---

## Step 10: Slice Timing Correction

**Purpose**: Correct for slice acquisition timing

**Tool**: AFNI `3dTshift`

**C-PAC Reference**: `CPAC/func_preproc/func_preproc.py` lines 400-450

```python
func_slice_timing = afni.TShift(
    tpattern='@filename' or 'alt+z' etc,  # from BIDS metadata
    tzero=0.0,
    outputtype='NIFTI_GZ'
)
```

**Native Implementation**:
```bash
# tpattern from BIDS: SliceTiming field
3dTshift -prefix func_stc.nii.gz -tpattern @slice_timing.1D func.nii.gz
```

**Outputs**:
- `desc-stc_bold` (slice-time corrected)

---

## Step 11: BOLD Masking (FSL_AFNI method)

**Purpose**: Create brain mask for BOLD

**Tool**: FSL + AFNI hybrid (fmriprep-style)

**C-PAC Reference**: `CPAC/func_preproc/func_preproc.py` lines 980-1090

**Method**:
1. Binarize template brain mask
2. Dilate
3. Run FSL BET on mean BOLD
4. Dilate BET result
5. Multiply masks together
6. Apply final mask

```python
# Key steps
binarize_mask = fsl.MathsCommand(args='-bin')
dilate = fsl.DilateImage(operation='max', kernel_shape='sphere', kernel_size=3)
bet = fsl.BET(frac=0.2, mask=True, functional=False)
combine = fsl.BinaryMaths(operation='mul')
```

**Outputs**:
- `space-bold_desc-brain_mask`

---

## Step 12: Coregistration (BOLD → T1w)

**Purpose**: Register functional to anatomical

**Tool**: FSL `flirt` with BBR

**C-PAC Reference**: `CPAC/registration/registration.py` lines 710-1000

**Parameters**:
```yaml
coregistration:
  using: FSL
  dof: 6
  cost: corratio
  boundary_based_registration:
    run: True
    reference: brain
    bbr_wm_map: partial_volume_map
    bbr_schedule: $FSLDIR/etc/flirtsch/bbr.sch
```

**Steps**:
1. Initial linear registration (6 DOF, corratio)
2. BBR refinement using WM boundary

```python
# Initial registration
linear_reg = fsl.FLIRT(cost='corratio', dof=6)

# BBR
bbreg = fsl.FLIRT(
    cost='bbr',
    dof=6,
    schedule='$FSLDIR/etc/flirtsch/bbr.sch',
    wm_seg=wm_mask
)
```

**Native Implementation**:
```bash
# Initial
flirt -in func_mean.nii.gz -ref t1_brain.nii.gz -dof 6 -cost corratio \
    -omat func2anat_init.mat -out func2anat_init.nii.gz

# BBR
flirt -in func_mean.nii.gz -ref t1_brain.nii.gz -dof 6 -cost bbr \
    -wmseg wm_mask.nii.gz -schedule $FSLDIR/etc/flirtsch/bbr.sch \
    -init func2anat_init.mat -omat func2anat.mat -out func2anat.nii.gz
```

**Outputs**:
- `from-bold_to-T1w_mode-image_desc-linear_xfm` (affine matrix)

---

## Step 13: Transform Composition

**Purpose**: Create composite transform BOLD → Template

**C-PAC Reference**: `CPAC/registration/registration.py` lines 3780-3950

**Compose**:
1. Motion correction transforms (per volume)
2. Distortion correction warp (if applicable)
3. BOLD → T1w affine
4. T1w → Template warp

For RBC with `single_step_resampling_from_stc`:
- All transforms composed and applied in ONE interpolation step
- Input: slice-timing-corrected data (before motion correction)

---

## Step 14: Single-Step Resampling to Template

**Purpose**: Apply all transforms in one interpolation

**Tool**: FSL `applywarp` or ANTs `antsApplyTransforms`

**C-PAC Reference**: `CPAC/registration/registration.py` lines 3825-3950

**For each volume**:
```bash
# Compose warps
convertwarp --ref=template_2mm.nii.gz \
    --premat=motion_vol_N.mat \
    --warp1=distortion_warp.nii.gz \
    --postmat=bold2anat.mat \
    --warp2=anat2template_warp.nii.gz \
    --out=composite_warp_vol_N.nii.gz

# Apply
applywarp --in=stc_vol_N.nii.gz --ref=template_2mm.nii.gz \
    --warp=composite_warp_vol_N.nii.gz \
    --out=template_vol_N.nii.gz --interp=spline
```

**Output resolution**: 2mm isotropic

**Outputs**:
- `space-template_desc-preproc_bold` (resampled timeseries in template space)

---

## Step 15: Nuisance Regression

**Purpose**: Remove confound signals

**Tool**: AFNI `3dTproject`

**C-PAC Reference**: `CPAC/nuisance/nuisance.py` lines 1600-1750

### Regressor Generation

**C-PAC Reference**: `CPAC/nuisance/nuisance.py` lines 700-1500

**RBC uses two regressor sets**:

#### Set 1: "36_parameter"
| Component | How Generated |
|-----------|---------------|
| Motion (6) | From mcflirt |
| Motion derivatives (6) | Temporal diff of motion |
| Motion squared (6) | Square of motion |
| Motion deriv squared (6) | Square of derivatives |
| CSF mean (1) | Mean signal in eroded CSF mask |
| CSF deriv + squared + deriv² (3) | Temporal derivatives |
| WM mean + derivs (4) | Same as CSF |
| GlobalSignal + derivs (4) | Whole-brain mean + derivatives |
| **Total: 36** | |

#### Set 2: "aCompCor"
Same as above but replace GlobalSignal with:
- aCompCor (5 PCs from WM+CSF combined mask, using DetrendPC method)

**Mask Erosion** (before regressor extraction):
```yaml
CSF: erode to 90% of original volume (proportion-based)
WM: erode to 60% of original volume  
Brain: erode by 30mm
```

**C-PAC Reference for erosion**: `CPAC/nuisance/nuisance.py` lines 2200-2600

**Regression**:
```python
nuisance_regression = afni.TProject(
    polort=0,      # Don't add polynomials (bandpass handles this)
    ort=regressor_file,
    bandpass=(0.01, 0.1),  # Hz
    mask=mask_file,
    outputtype='NIFTI_GZ'
)
```

**Native Implementation**:
```bash
# Generate regressors to file (one column per regressor)
# ... (custom Python code to extract signals and compute derivatives)

# Apply regression with bandpass
3dTproject -input func.nii.gz -prefix func_cleaned.nii.gz \
    -ort regressors.1D -bandpass 0.01 0.1 -mask mask.nii.gz
```

**Outputs**:
- `desc-cleaned_bold` (denoised timeseries)

---

## Step 16: Template-Space Despiking (Second Pass)

**Same as Step 9**, but applied in template space after nuisance regression.

---

## Step 17: ALFF/fALFF

**Purpose**: Compute amplitude of low frequency fluctuations

**Tool**: AFNI `3dRSFC` or custom implementation

**C-PAC Reference**: `CPAC/alff/alff.py` lines 170-250

```python
# Bandpass filter
bandpass = afni.Bandpass(
    highpass=0.01,
    lowpass=0.1,
    outputtype='NIFTI_GZ'
)

# Compute std of filtered (ALFF)
stddev_filtered = afni.TStat(options='-stdev')

# Compute std of unfiltered (for fALFF denominator)
stddev_unfiltered = afni.TStat(options='-stdev')

# fALFF = ALFF / unfiltered_std
falff = afni.Calc(expr='a/b')
```

**Native Implementation**:
```bash
# ALFF
3dBandpass -prefix func_bp.nii.gz 0.01 0.1 func.nii.gz
3dTstat -stdev -prefix alff.nii.gz func_bp.nii.gz

# fALFF
3dTstat -stdev -prefix std_unfiltered.nii.gz func.nii.gz
3dcalc -a alff.nii.gz -b std_unfiltered.nii.gz -expr 'a/b' -prefix falff.nii.gz
```

**Outputs**:
- `alff` (ALFF map)
- `falff` (fALFF map)

---

## Step 18: ReHo

**Purpose**: Compute regional homogeneity (Kendall's W)

**Tool**: Custom Python implementation

**C-PAC Reference**: `CPAC/reho/utils.py` lines 70-200

**Algorithm**:
1. For each voxel, get neighborhood (7, 19, or 27 voxels)
2. Rank timepoints for each voxel in neighborhood
3. Compute Kendall's coefficient of concordance (KCC)

```python
def compute_reho(in_file, mask_file, cluster_size=27):
    # Load data
    data = nib.load(in_file).get_fdata()  # (x, y, z, t)
    mask = nib.load(mask_file).get_fdata()
    
    # For each voxel in mask:
    #   1. Get neighbors
    #   2. Rank each voxel's timeseries
    #   3. Compute KCC across neighbors
    
    # KCC formula:
    # W = 12 * S / (K^2 * (N^3 - N))
    # where S = sum of squared deviations of rank sums
    #       K = number of voxels (neighbors)
    #       N = number of timepoints
```

**Outputs**:
- `reho` (ReHo map)

---

## Step 19: Network Centrality

**Purpose**: Compute voxelwise connectivity metrics

**C-PAC Reference**: `CPAC/network_centrality/network_centrality.py`

### Degree Centrality
For each voxel, count connections above threshold.

### Local Functional Connectivity Density (lFCD)
Count connected voxels in local cluster.

**Parameters**:
```yaml
template_mask: /cpac_templates/Mask_ABIDE_85Percent_GM.nii.gz
degree_centrality:
  weight_options: [Binarized]
  threshold: 0.001  # significance threshold
local_functional_connectivity_density:
  weight_options: [Binarized, Weighted]
  correlation_threshold_option: Significance threshold
  correlation_threshold: 0.001
```

**Outputs**:
- `centrality` maps (Degree, lFCD)

---

## Step 20: Timeseries Extraction

**Purpose**: Extract mean timeseries from atlas ROIs

**C-PAC Reference**: `CPAC/timeseries/timeseries_analysis.py`

**Method**: For each atlas, compute mean signal within each parcel.

**Atlases used** (partial list):
- Schaefer 200/300/400/1000 parcels
- AAL
- Harvard-Oxford (cortical + subcortical)
- Glasser
- Yeo 7/17 networks
- CC200, CC400

**Native Implementation**:
```python
import nibabel as nib
import numpy as np

def extract_timeseries(func_file, atlas_file):
    func = nib.load(func_file).get_fdata()  # (x, y, z, t)
    atlas = nib.load(atlas_file).get_fdata()  # (x, y, z)
    
    labels = np.unique(atlas[atlas > 0])
    timeseries = []
    
    for label in labels:
        mask = atlas == label
        ts = func[mask].mean(axis=0)  # mean across voxels
        timeseries.append(ts)
    
    return np.array(timeseries)  # (n_rois, n_timepoints)
```

**Outputs**:
- `timeseries` (n_rois × n_timepoints arrays per atlas)
- `correlation_matrix` (using Nilearn's correlation methods)

---

## Step 21: Spatial Smoothing

**Purpose**: Smooth derivative maps

**Tool**: AFNI `3dBlurToFWHM`

**C-PAC Reference**: `CPAC/image_utils/spatial_smoothing.py`

**Parameters**:
```yaml
fwhm: 6  # mm
method: AFNI
```

```bash
3dBlurToFWHM -input map.nii.gz -prefix map_smooth.nii.gz \
    -FWHM 6 -mask mask.nii.gz
```

**Outputs**:
- Smoothed versions of ALFF, fALFF, ReHo, etc.

---

## Step 22: Z-Scoring

**Purpose**: Standardize maps

**C-PAC Reference**: `CPAC/image_utils/statistical_transforms.py`

```python
def z_score(data, mask):
    masked = data[mask > 0]
    mean = masked.mean()
    std = masked.std()
    return (data - mean) / std
```

**Outputs**:
- Z-scored versions of all derivative maps

---

## Summary: Minimal Tool Dependencies

Your reimplementation needs:

| Category | Tools |
|----------|-------|
| **Required** | AFNI (3drefit, 3dresample, 3dcalc, 3dTshift, 3dDespike, 3dTproject, 3dBandpass, 3dTstat, 3dBlurToFWHM) |
| **Required** | FSL (mcflirt, flirt, fast, bet, applywarp, convertwarp, fslmaths) |
| **Required** | ANTs (antsRegistration, antsApplyTransforms, N4BiasFieldCorrection, antsBrainExtraction.sh) |
| **Python** | nibabel, numpy, scipy (for ReHo, timeseries extraction) |

---

## Key Differences from C-PAC to Fix

1. **Use bias-corrected T1w**: After brain extraction, use the N4-corrected output, not the original.

2. **Simplify transform chain**: C-PAC's single-step resampling is complex. You can apply transforms sequentially if numerical precision is acceptable, or carefully compose warps.

3. **Verify mask erosion**: The proportion-based erosion (90% for CSF, 60% for WM) may need tuning.

4. **Consider removing double despiking**: RBC runs despike twice; evaluate if both passes are necessary.

5. **Validate against papers**: The RBC paper should be the ground truth, not C-PAC's implementation.
