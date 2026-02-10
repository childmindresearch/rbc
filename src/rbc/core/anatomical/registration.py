"""RBC registration method."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from niwrap import ants

from rbc.core import CPAC_ANTS_SEED
from rbc.core.resources import MNI_TEMPLATES

_PREFIX = "ants_reg"


class CompositeTransforms(NamedTuple):
    """Forward and inverse composite transformation paths."""

    forward: Path
    inverse: Path


def ants_registration(in_file: Path, seed: int = CPAC_ANTS_SEED) -> CompositeTransforms:
    """ANTs registration to MNI152 template.

    Args:
        in_file: Path to file to be compute transformation with template.
        seed: Seed to use for reproducibility.

    Returns:
        A namespace mapping forward and inverse transformation paths.
    """
    registration = ants.ants_registration(
        stages=[
            ants.ants_registration_stage(
                transform=ants.ants_registration_transform_rigid(gradient_step=0.05),
                metric=ants.ants_registration_metric_mutual_information(
                    fixed_image=MNI_TEMPLATES.brain_1mm,
                    moving_image=in_file,
                    metric_weight=1,
                    number_of_bins=ants.ants_registration_number_of_bins(
                        number_of_bins_value=32,
                        sampling_strategy=ants.ants_registration_sampling_strategy_1(
                            sampling_strategy_value="Regular",
                            sampling_percentage=ants.ants_registration_sampling_percentage_1(
                                sampling_percentage_value=0.25
                            ),
                        ),
                    ),
                ),
                convergence=ants.ants_registration_convergence(
                    convergence="100x100",
                    convergence_threshold=0.000001,
                    convergence_window_size=20,
                ),
                smoothing_sigmas="2.0x1.0vox",
                shrink_factors="2x1",
                use_histogram_matching=True,
            ),
            ants.ants_registration_stage(
                transform=ants.ants_registration_transform_affine(gradient_step=0.08),
                metric=ants.ants_registration_metric_mutual_information(
                    fixed_image=MNI_TEMPLATES.brain_1mm,
                    moving_image=in_file,
                    metric_weight=1,
                    number_of_bins=ants.ants_registration_number_of_bins(
                        number_of_bins_value=32,
                        sampling_strategy=ants.ants_registration_sampling_strategy_1(
                            sampling_strategy_value="Regular",
                            sampling_percentage=ants.ants_registration_sampling_percentage_1(
                                sampling_percentage_value=0.25
                            ),
                        ),
                    ),
                ),
                convergence=ants.ants_registration_convergence(
                    convergence="100x100",
                    convergence_threshold=0.000001,
                    convergence_window_size=20,
                ),
                smoothing_sigmas="1.0x0.0vox",
                shrink_factors="2x1",
                use_histogram_matching=True,
            ),
            ants.ants_registration_stage(
                transform=ants.ants_registration_transform_syn(
                    gradient_step=0.1,
                    update_field_variance_in_voxel_space=ants.ants_registration_update_field_variance_in_voxel_space(
                        update_field_variance_in_voxel_space_value=3,
                        total_field_variance_in_voxel_space=ants.ants_registration_total_field_variance_in_voxel_space(
                            total_field_variance_in_voxel_space_value=0
                        ),
                    ),
                ),
                metric=ants.ants_registration_metric_ants_neighbourhood_cross_correlation(
                    fixed_image=MNI_TEMPLATES.brain_1mm,
                    moving_image=in_file,
                    metric_weight=1,
                    radius=ants.ants_registration_radius(radius_value=4),
                ),
                convergence=ants.ants_registration_convergence(
                    convergence="100x70x50x20",
                    convergence_threshold=0.000001,
                    convergence_window_size=10,
                ),
                smoothing_sigmas="3.0x2.0x1.0x0.0vox",
                shrink_factors="8x4x2x1",
                use_histogram_matching=True,
            ),
        ],
        random_seed=seed,
        collapse_output_transforms=True,
        dimensionality=3,
        initial_moving_transform=ants.ants_registration_initial_moving_transform_initialization_feature(
            fixed_image=MNI_TEMPLATES.brain_1mm,
            moving_image=in_file,
            initialization_feature=0,
        ),
        winsorize_image_intensities=ants.ants_registration_winsorize_image_intensities(
            lower_quantile=0.005, upper_quantile=0.995
        ),
        interpolation="LanczosWindowedSinc",
        output=f"[{_PREFIX}_,{_PREFIX}_Warped.nii.gz]",
    )
    fwd = ants.ants_apply_transforms(
        reference_image=MNI_TEMPLATES.brain_1mm,
        transform=[
            ants.ants_apply_transforms_transform_file_name(
                registration.root / f"{_PREFIX}_0GenericAffine.mat"
            ),
            ants.ants_apply_transforms_transform_file_name(
                registration.root / f"{_PREFIX}_1Warp.nii.gz"
            ),
        ],
        output=ants.ants_apply_transforms_composite_displacement_field_output(
            composite_displacement_field="forward_xfm.nii.gz",
            print_out_composite_warp_file=True,
        ),
    )
    rev = ants.ants_apply_transforms(
        reference_image=in_file,
        transform=[
            ants.ants_apply_transforms_transform_file_name(
                registration.root / f"{_PREFIX}_1InverseWarp.nii.gz"
            ),
            ants.ants_apply_transforms_use_inverse(
                registration.root / f"{_PREFIX}_0GenericAffine.mat"
            ),
        ],
        output=ants.ants_apply_transforms_composite_displacement_field_output(
            composite_displacement_field="inverse_xfm.nii.gz",
            print_out_composite_warp_file=True,
        ),
    )
    return CompositeTransforms(
        forward=fwd.output.output_image_outfile, inverse=rev.output.output_image_outfile
    )
