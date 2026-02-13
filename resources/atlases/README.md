# Atlases

Brain parcellation atlas NIfTI files used for timeseries extraction.

## Source

All atlases are sourced from the C-PAC Docker image
`fcpindi/c-pac:release-v1.8.5.dev1`.

## Files

From `/cpac_templates/` in the container:

| File | Atlas |
|------|-------|
| `Schaefer2018_space-FSLMNI152_res-2mm_desc-200Parcels17NetworksOrder.nii.gz` | Schaefer 200 |
| `Schaefer2018_space-FSLMNI152_res-2mm_desc-300Parcels17NetworksOrder.nii.gz` | Schaefer 300 |
| `Schaefer2018_space-FSLMNI152_res-2mm_desc-400Parcels17NetworksOrder.nii.gz` | Schaefer 400 |
| `Schaefer2018_space-FSLMNI152_res-2mm_desc-1000Parcels17NetworksOrder.nii.gz` | Schaefer 1000 |
| `CC200.nii.gz` | Craddock 200 |
| `CC400.nii.gz` | Craddock 400 |
| `aal_mask_pad.nii.gz` | AAL |

From `/ndmg_atlases/label/Human/` in the container:

| File | Atlas |
|------|-------|
| `HarvardOxfordcort-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz` | Harvard-Oxford Cortical |
| `HarvardOxfordsub-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz` | Harvard-Oxford Subcortical |
| `Glasser_space-MNI152NLin6_res-1x1x1.nii.gz` | Glasser |
| `Yeo-7_space-MNI152NLin6_res-1x1x1.nii.gz` | Yeo 7-network |
| `Yeo-17_space-MNI152NLin6_res-1x1x1.nii.gz` | Yeo 17-network |
