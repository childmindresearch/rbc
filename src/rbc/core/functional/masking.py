"""BOLD masking and intensity uniformization.

The pipeline follows these primary stages:
    1. Registration: Aligns BOLD reference to template via antsAI + Affine.
    2. Warping: Transforms template mask into BOLD native space.
    3. Correction: N4 bias field correction of the BOLD reference.
    4. Skull-stripping: First-pass BET and second-pass 3dAutomask.
    5. Intersection: Combines masks for a conservative final result.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from niwrap import afni, ants, fsl

from rbc.core import CPAC_ANTS_SEED


class MaskingOutputs(NamedTuple):
    """Outputs from the BOLD masking and uniformization pipeline.

    Attributes:
        final_mask: The final, binarized intersection mask.
        skull_stripped_bold: Intensity-uniformized BOLD reference with mask applied.
    """

    final_mask: Path
    skull_stripped_bold: Path


def bold_masking(
    bold_ref: Path, template_mask: Path, template_ref: Path, seed: int = CPAC_ANTS_SEED
) -> MaskingOutputs:
    """Perform multi-pass brain masking on a BOLD reference image.

    Args:
        bold_ref: BOLD reference volume in native space.
        template_mask: 2mm brain mask in template space.
        template_ref: 2mm BOLD reference in template space.
        seed: Random seed for reproducibility (default: CPAC_ANTS_SEED).

    Returns:
        The final binary mask and the skull-stripped, uniformized BOLD reference.
    """
    # --- Phase 1: Template to BOLD Registration ---

    # Initialize affine transformation
    affine_transformation = ants.ants_ai(
        metric=ants.ants_ai_metric_mattes(
            fixed_image=template_ref,
            moving_image=bold_ref,
            number_of_bins=ants.ants_ai_number_of_bins(
                number_of_bins_value=32,
                sampling_strategy=ants.ants_ai_sampling_strategy(
                    sampling_strategy_value="Regular",
                    sampling_percentage=ants.ants_ai_sampling_percentage(
                        sampling_percentage_value=0.2
                    ),
                ),
            ),
        ),
        transform=ants.ants_ai_transform_affine(gradient_step=0.1),
        output="initialization.mat",
        random_seed=seed,
        verbose=True,
        convergence=ants.ants_ai_convergence(
            number_of_iterations=10,
            convergence_threshold=ants.ants_ai_convergence_threshold(
                convergence_threshold_value=0.000001,
                convergence_window_size=ants.ants_ai_convergence_window_size(
                    convergence_window_size_value=10
                ),
            ),
        ),
        translation_search_grid=ants.ants_ai_translation_search_grid(
            step_size=40, grid=[0, 40, 40]
        ),
        search_factor=ants.ants_ai_search_factor(
            search_factor_value=20,
            arc_fraction=ants.ants_ai_arc_fraction(arc_fraction_value=0.12),
        ),
        align_principal_axes=False,
        dimensionality=3,
        masks=ants.ants_ai_masks(fixed_image_mask=template_mask),
    )

    # Refine registration with ANTs
    registration = ants.ants_registration(
        stages=[
            ants.ants_registration_stage(
                transform=ants.ants_registration_transform_affine(gradient_step=0.1),
                metric=ants.ants_registration_metric_mattes(
                    fixed_image=template_ref,
                    moving_image=bold_ref,
                    metric_weight=1,
                    number_of_bins=ants.ants_registration_number_of_bins_1(
                        number_of_bins_value=64,
                        sampling_strategy=ants.ants_registration_sampling_strategy_2(
                            sampling_strategy_value="Random",
                            sampling_percentage=ants.ants_registration_sampling_percentage_2(
                                sampling_percentage_value=0.2
                            ),
                        ),
                    ),
                ),
                convergence=ants.ants_registration_convergence(
                    convergence="200",
                    convergence_threshold=1e-9,
                    convergence_window_size=10,
                ),
                smoothing_sigmas="2.0mm",
                shrink_factors="2",
                use_histogram_matching=True,
            )
        ],
        dimensionality=3,
        winsorize_image_intensities=ants.ants_registration_winsorize_image_intensities(
            lower_quantile=0.05, upper_quantile=0.98
        ),
        random_seed=seed,
        float_=True,
        initial_moving_transform=ants.ants_registration_initial_moving_transform_use_inverse(
            initial_moving_transform=affine_transformation.output_transform,
            use_inverse=ants.ants_registration_use_inverse(use_inverse_value=False),
        ),
        interpolation="Linear",
        initialize_transforms_per_stage=False,
        collapse_output_transforms=True,
        write_composite_transform=False,
        output="transform",
    )

    # --- Phase 2: Warp Template Mask to BOLD Space ---

    # Apply transforms to probability mask
    warped_probseg = ants.ants_apply_transforms(
        reference_image=bold_ref,
        output=ants.ants_apply_transforms_warped_output(
            warped_output_file_name="probseg_transform.nii.gz"
        ),
        default_value=0,
        float_=True,
        input_image=template_mask,
        interpolation=ants.ants_apply_transforms_bspline(order=3),
        transform=[ants.ants_apply_transforms_use_inverse(registration.generic_affine)],
    )

    # Binarize warped mask
    binarized_mask = fsl.fslmaths(
        input_files=[warped_probseg.output.output_image_outfile],
        operations=[
            fsl.fslmaths_operation_thr(thr=0.85),
            fsl.fslmaths_operation_bin(bin_=True),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="binary_mask.nii.gz",
    )

    # Dilate binary mask (3mm sphere)
    dilated_binary_mask = fsl.fslmaths(
        input_files=[binarized_mask.output_file],
        operations=[
            fsl.fslmaths_operation_kernel_sphere(kernel_sphere=3),
            fsl.fslmaths_operation_dil_f(dil_f=True),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="dilated_binary_mask.nii.gz",
        datatype_internal="char",
    )

    # --- Phase 3: Fix Headers and N4 Correction ---

    direction_header = ants.print_header(
        image=dilated_binary_mask.output_file, what_information=4
    )

    direction_string = direction_header.output[0]
    direction_matrix = [float(x) for x in direction_string.replace("x", " ").split()]

    # Set direction on BOLD reference
    bold_ref_dir_corrected = ants.set_direction_by_matrix(
        infile=bold_ref,
        outfile="bold_ref_dir_corrected.nii.gz",
        direction_matrix=direction_matrix,
    )

    # N4 bias field correction
    n4_bias_correction = ants.n4_bias_field_correction(
        input_image=bold_ref_dir_corrected.outfile,
        output=ants.n4_bias_field_correction_corrected_output(
            corrected_output_file_name="ref_bold_corrected.nii.gz"
        ),
        image_dimensionality=3,
        bspline_fitting=ants.n4_bias_field_correction_bspline_fitting(
            spline_distance=[200], spline_order=3
        ),
        mask_image=dilated_binary_mask.output_file,
    )

    # --- Phase 4: First-Pass Skull Stripping ---

    # BET skull strip
    skull_strip = fsl.bet(
        infile=n4_bias_correction.output.output_image_outfile,
        fractional_intensity=0.2,
        binary_mask=True,
        maskfile="ref_bold_corrected_brain",
    )

    # Dilate BET mask (6mm sphere)
    dilated_bet_mask = fsl.fslmaths(
        input_files=[skull_strip.binary_mask],
        operations=[
            fsl.fslmaths_operation_kernel_sphere(kernel_sphere=6),
            fsl.fslmaths_operation_dil_f(dil_f=True),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="bet_mask_dil.nii.gz",
        datatype_internal="char",
    )

    # Apply dilated BET mask
    masked_bold = fsl.fslmaths(
        input_files=[skull_strip.outfile],
        operations=[
            fsl.fslmaths_operation_mas(mas=dilated_bet_mask.output_file),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="ref_bold_corrected_brain_masked.nii.gz",
    )

    # --- Phase 5: Intensity Uniformization & Second-Pass ---

    # AFNI Unifize
    unifized = afni.v_3d_unifize(
        in_file=masked_bold.output_file,
        cl_frac=0.2,
        rbt=[18.3, 65, 90],
        prefix="uni.nii.gz",
        t2=True,
    )

    # AFNI Automask
    automask = afni.v_3d_automask(
        in_file=unifized.out_file,
        apply_prefix="uni_masked.nii.gz",
        dilate=1,
        prefix="uni_mask.nii.gz",
    )

    # --- Phase 6: Combine Masks ---

    # Intersect both masks
    final_mask = fsl.fslmaths(
        input_files=[skull_strip.binary_mask],
        operations=[
            fsl.fslmaths_operation_mul(
                mul=fsl.fslmaths_mul_image(image=automask.mask_file)
            ),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="final_mask.nii.gz",
    )

    # Apply final mask to unifized image
    skull_stripped_bold = fsl.fslmaths(
        input_files=[unifized.out_file],
        operations=[
            fsl.fslmaths_operation_mas(mas=final_mask.output_file),
            fsl.fslmaths_operation_seed(seed=seed),
        ],
        output="masked_ref_bold.nii.gz",
    )

    return MaskingOutputs(
        final_mask=Path(final_mask.output_file),
        skull_stripped_bold=Path(skull_stripped_bold.output_file),
    )
