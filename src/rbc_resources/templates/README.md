# Templates

Standard-space template images used by the RBC pipeline for brain extraction
and anatomical/functional registration.

## Source

All templates are sourced from the C-PAC Docker image
`fcpindi/c-pac:release-v1.8.5.dev1`.

## OASIS templates

Used by ANTs `antsBrainExtraction.sh` to skull-strip T1w images.

From `/opt/dcan-tools/pipeline/global/templates/OASIS-30_Atropos_template/` in the container:

| Current filename | Original filename |
|---|---|
| `oasis_template.nii.gz` | `T_template0.nii.gz` |
| `oasis_probability_mask.nii.gz` | `T_template0_BrainCerebellumProbabilityMask.nii.gz` |
| `oasis_registration_mask.nii.gz` | `T_template0_BrainCerebellumRegistrationMask.nii.gz` |

## MNI152 templates

Standard-space registration targets and reference grids.

From FSL 6.0 standard templates via the container:

| Current filename | Original filename |
|---|---|
| `mni152_T1w_1mm_brain.nii.gz` | `MNI152_T1_1mm_brain.nii.gz` |
| `mni152_T1w_2mm_brain_mask.nii.gz` | `MNI152_T1_2mm_brain_mask.nii.gz` |
| `mni152_bold_ref_2mm.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref.nii.gz` |
