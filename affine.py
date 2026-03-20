import nibabel as nib
import numpy as np

sbref = nib.load("/Users/janhavi.pillai/Desktop/data/derivatives/longitudinal-test-output/sub-19861/ses-2/func/sub-19861_ses-2_task-rest_run-1_sbref.nii.gz")
skull_stripped = nib.load("/Users/janhavi.pillai/Desktop/masked_ref_bold.nii.gz")  # skull_stripped_bold output

print("sbref affine:")
print(sbref.affine)
print("\nskull_stripped affine:")
print(skull_stripped.affine)
print("\nMatch:", np.allclose(sbref.affine, skull_stripped.affine))