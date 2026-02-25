# /// script
# dependencies = ["niwrap"]
# requires-python = ">=3.11"
#
# ///
"""Generate a template using Freesurfer's mri_robust_template.

Run with:
    uv run scripts/build_robust_template.py <[input_file, ...]> <output_file>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

from niwrap import freesurfer


def create_parser() -> argparse.ArgumentParser:
    """Create parser for template creation."""
    parser = argparse.ArgumentParser(
        prog="create_template",
        description="Create a robust template using Freesurfer's mri_robust_template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s in_files [in_files...] output_file [options]",
    )
    parser.add_argument(
        "in_files",
        nargs="+",
        type=Path,
        help="Space separate list of input file(s) to create a template from",
    )
    parser.add_argument(
        "output_file", help="Output template file (including directory)"
    )
    parser.add_argument(
        "--fs-license",
        required=False,
        type=Path,
        help="Path to Freesurfer license",
    )
    return parser


class RobustTemplateOutputs(NamedTuple):
    """Outputs from template generation - template + transforms."""

    template: Path
    transforms: list[Path]


def generate_robust_template(
    in_files: Sequence[Path], output_file: str
) -> RobustTemplateOutputs:
    """Construct unbiased, robust template for longitudinal volumes with FreeSurfer.

    Uses an iterative method construct a mean volume and robust rigid registration
    of all input images to the current mean/median.

    Within-Subject Template Estimation for Unbiased Longitudinal Image Analysis
        M. Reuter, N.J. Schmansky, H.D. Rosas, B. Fischl.
        NeuroImage 61(4):1402-1418, 2012.
    """
    lta_files = []
    for idx, in_file in enumerate(in_files):
        if not Path(in_file).exists():
            raise FileNotFoundError(f"{in_file} not found.")
        lta_files.append(f"in_file{idx + 1}_to_template.lta")

    # Initialize with same defaults as fmriprep
    robust_template = freesurfer.mri_robust_template(
        mov=in_files,
        template=output_file,
        lta=lta_files,
        inittp=1,  # map everything to first time point
        fixtp=True,
        iscale=True,  # intensity scale (7-DOF - rigid + intensity)
        noit=True,  # no iteration; fmriprep turns this on -> why?
        satit=True,  # autodetect sensitivity
        subsample=200,  # subsample if any dimension has over this many volumes
    )

    return RobustTemplateOutputs(
        template=robust_template.template, transforms=lta_files
    )


def fs_to_ants_xfm(in_xfms: Sequence[Path]) -> list[Path]:
    """Convert Freesurfer transformations to ANTs compatible format."""
    return [
        freesurfer.lta_convert(
            in_lta=in_xfm, out_itk=str(in_xfm.with_suffix(".mat"))
        ).output_transform_file
        for in_xfm in in_xfms
    ]


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Check FS license
    fs_license = args.fs_license or os.getenv("FS_LICENSE")
    if fs_license is None or not Path(fs_license).exists():
        raise FileNotFoundError(f"FreeSurfer license not found: {fs_license}")

    # 1. Construct template
    in_files = args.in_files
    # if len(in_files) == 1:
    #     raise ValueError("Only a single volume found")
    robust_template = generate_robust_template(
        in_files=in_files, output_file="template.nii.gz"
    )

    # 2. Convert transformations to ANTs compatible format
    subj_to_temp = fs_to_ants_xfm(robust_template.transforms)
