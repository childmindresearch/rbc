# Atlases

Brain parcellation atlas NIfTI files used for timeseries extraction.

## Source

NIfTI files, tabular files listing ROI labels, and metadata are sourced from:
[ReproBrainChart/sourcedata-atlases @ rbc-labels](https://github.com/ReproBrainChart/sourcedata-atlases/tree/rbc-labels).

## Files

Each atlas has up to three associated files:
- `.nii.gz` — NIfTI parcellation image
- `.json` — atlas metadata (atlas description, coordinate space, generation method, etc.)
- `.tsv` — tabular file listing ROI labels and metadata

| Filename | Atlas |
|---|---|
| `atlas-CC200_space-MNI152NLin6_res-2_dseg.nii.gz` | Craddock 200 |
| `atlas-CC400_space-MNI152NLin6_res-2_dseg.nii.gz` | Craddock 400 |
| `atlas-Schaefer2018_space-MNI152NLin6_res-2_desc-200Parcels17NetworksOrder_dseg.nii.gz` | Schaefer 200 |
| `atlas-Schaefer2018_space-MNI152NLin6_res-2_desc-300Parcels17NetworksOrder_dseg.nii.gz` | Schaefer 300 |
| `atlas-Schaefer2018_space-MNI152NLin6_res-2_desc-400Parcels17NetworksOrder_dseg.nii.gz` | Schaefer 400 |
| `atlas-Schaefer2018_space-MNI152NLin6_res-2_desc-1000Parcels17NetworksOrder_dseg.nii.gz` | Schaefer 1000 |
| `atlas-AAL_space-MNI152NLin6_res-2_dseg.nii.gz` | AAL |
| `atlas-Brodmann_space-MNI152NLin6_res-2_dseg.nii.gz` | Brodmann |
| `atlas-Glasser_space-MNI152NLin6_res-2_dseg.nii.gz` | Glasser |
| `atlas-Slab907_space-MNI152NLin6_res-2_dseg.nii.gz` | Slab 907 |
| `atlas-HarvardOxfordcortMaxprobThr25_space-MNI152NLin6_res-2_dseg.nii.gz` | Harvard-Oxford Cortical |
| `atlas-HarvardOxfordsubMaxprobThr25_space-MNI152NLin6_res-2_dseg.nii.gz` | Harvard-Oxford Subcortical |
| `atlas-Juelich_space-MNI152NLin6_res-2_dseg.nii.gz` | Juelich |
| `atlas-Yeo7_space-MNI152NLin6_res-2_dseg.nii.gz` | Yeo 7-network |
| `atlas-Yeo7liberal_space-MNI152NLin6_res-2_dseg.nii.gz` | Yeo 7-network (liberal) |
| `atlas-Yeo17_space-MNI152NLin6_res-2_dseg.nii.gz` | Yeo 17-network |
| `atlas-Yeo17liberal_space-MNI152NLin6_res-2_dseg.nii.gz` | Yeo 17-network (liberal) |

## JSON metadata format

Each `.json` file contains two required top-level keys:

| Key | Description |
|---|---|
| `MetaData` | Atlas-level metadata (see below) |
| `rois` | Per-region metadata (see below) |

### `MetaData`

| Key | Required | Description |
|---|---|---|
| `AtlasName` | required | Name of the atlas, matching the filename. |
| `Description` | recommended | Brief description of the atlas and its use case. |
| `Native Coordinate Space` | recommended | Coordinate space the atlas is defined in (e.g. MNI, Talairach). |
| `Hierarchical` | optional | `true` if the atlas has hierarchical subregions, `false` otherwise. |
| `Symmetrical` | optional | `true` if the atlas is designed to be symmetrical, `false` otherwise. |
| `Number of Regions` | optional | Number of regions, not including background. |
| `Average Volume Per Region` | optional | Average region volume, not including background. |
| `Year Generated` | optional | Year the atlas was first created. |
