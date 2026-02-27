# Atlases

Brain parcellation atlas NIfTI files used for timeseries extraction.

## Source

All atlases are sourced from the C-PAC Docker image
`fcpindi/c-pac:release-v1.8.5.dev1`.

## Files

From `/cpac_templates/` in the container:

| Current filename | Original filename | Atlas |
|---|---|---|
| `schaefer_200.nii.gz` | `Schaefer2018_space-FSLMNI152_res-2mm_desc-200Parcels17NetworksOrder.nii.gz` | Schaefer 200 |
| `schaefer_300.nii.gz` | `Schaefer2018_space-FSLMNI152_res-2mm_desc-300Parcels17NetworksOrder.nii.gz` | Schaefer 300 |
| `schaefer_400.nii.gz` | `Schaefer2018_space-FSLMNI152_res-2mm_desc-400Parcels17NetworksOrder.nii.gz` | Schaefer 400 |
| `schaefer_1000.nii.gz` | `Schaefer2018_space-FSLMNI152_res-2mm_desc-1000Parcels17NetworksOrder.nii.gz` | Schaefer 1000 |
| `craddock_200.nii.gz` | `CC200.nii.gz` | Craddock 200 |
| `craddock_400.nii.gz` | `CC400.nii.gz` | Craddock 400 |
| `aal.nii.gz` | `aal_mask_pad.nii.gz` | AAL |

From `/ndmg_atlases/label/Human/` in the container:

| Current filename | Original filename | Atlas |
|---|---|---|
| `harvard_oxford_cortical.nii.gz` | `HarvardOxfordcort-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz` | Harvard-Oxford Cortical |
| `harvard_oxford_subcortical.nii.gz` | `HarvardOxfordsub-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz` | Harvard-Oxford Subcortical |
| `glasser.nii.gz` | `Glasser_space-MNI152NLin6_res-1x1x1.nii.gz` | Glasser |
| `yeo_7.nii.gz` | `Yeo-7_space-MNI152NLin6_res-1x1x1.nii.gz` | Yeo 7-network |
| `yeo_17.nii.gz` | `Yeo-17_space-MNI152NLin6_res-1x1x1.nii.gz` | Yeo 17-network |
