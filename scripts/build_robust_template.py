# /// script
# dependencies = [
#     "niwrap",
#     "styxpodman",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# styxpodman = { git = "https://github.com/styx-api/styxpodman", rev = "1382977" }
#
# ///
"""Generate a template using Freesurfer's mri_robust_template.

Run with:
    uv run scripts/build_robust_template.py <[input_file, ...]> <output_file>

Example:
    uv run scripts/build_robust_teplate \
        data/sub-01/ses-*/anat/sub-01_ses-*_T1w.nii.gz \
        sub-01_ses-longitudinal.nii.gz
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from niwrap import (
    Runner,
    freesurfer,
    get_global_runner,
    set_global_runner,
    use_docker,
    use_local,
    use_singularity,
)
from styxpodman import PodmanRunner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

_LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]
CONTAINER_LICENSE_PATH = "/opt/freesurfer/license.txt"


def create_parser() -> argparse.ArgumentParser:
    """Create parser for template creation."""
    parser = argparse.ArgumentParser(
        prog="create_template",
        description="Create a robust template using Freesurfer's mri_robust_template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s in_files [in_files...] output_file [options]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v, -vv, -vvv)",
    )
    parser.add_argument(
        "in_files",
        nargs="+",
        type=Path,
        help="Space separate list of input file(s) to create a template from",
    )
    parser.add_argument(
        "output_file", type=Path, help="Output template file (including directory)"
    )
    parser.add_argument(
        "--fs-license",
        required=False,
        type=Path,
        help="Path to Freesurfer license",
    )
    parser.add_argument(
        "--runner",
        choices=["local", "docker", "podman", "singularity"],
        default="local",
        type=lambda x: x.lower(),
        help="NiWrap runner to use for executing workflow",
    )

    return parser


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: Runner
    verbose: bool


def setup_runner(
    runner: Literal["local", "docker", "podman", "singularity"] = "local",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    verbose: int = 0,
    **kwargs,  # noqa: ANN003 (ignore annotation for kwargs)
) -> StyxContext:
    """Setup Styx with appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use - choices include
            ['local', 'docker', 'podman', 'singularity']
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        Configured logger instance and initialized runner
    """
    match runner_exec := runner.lower():
        case "local":
            use_local()
        case "docker":
            use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                docker_user_id=0,
                **kwargs,
            )
        case "podman":
            set_global_runner(
                runner=PodmanRunner(
                    podman_executable=runner_exec,
                    image_overrides=image_overrides,
                    podman_user_id=0,
                    **kwargs,
                )
            )
        case "singularity":
            use_singularity(
                singularity_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Unknown runner selection '{runner}' - please select one of "
                "'local', 'docker', or 'singularity'"
            )

    styx_runner = get_global_runner()
    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir()) / f"robust_template_{os.urandom(8).hex()}"
        tmp_dir.mkdir(exist_ok=False, parents=True)
    styx_runner.data_dir = tmp_dir
    logger = logging.getLogger(styx_runner.logger_name)
    log_level = min(verbose, len(_LOG_LEVELS) - 1)
    logger.setLevel(_LOG_LEVELS[log_level])
    return StyxContext(logger=logger, runner=styx_runner, verbose=verbose > 0)


def _get_mount_arg(runner: str, host_path: Path) -> list[str]:
    """Return runner-specific mount CLI args."""
    src, dst = str(host_path), CONTAINER_LICENSE_PATH
    if runner in ("podman", "docker"):
        return ["--mount", f"type=bind,source={src},target={dst},readonly"]
    return ["--bind", f"{src}:{dst}"]  # singularity


def mount_fs_license(runner: Runner, fs_license: str) -> None:
    """Mount FreeSurfer license file into an existing runner."""
    runner_name = type(runner).__name__.lower().replace("runner", "")
    license_path = Path(fs_license).resolve()

    if runner_name == "local":
        os.environ["FS_LICENSE"] = str(license_path)
        return

    extra_args_attr = f"{runner_name}_extra_args"
    getattr(runner, extra_args_attr).extend(_get_mount_arg(runner_name, license_path))
    runner.environ["FS_LICENSE"] = CONTAINER_LICENSE_PATH


class RobustTemplateOutputs(NamedTuple):
    """Outputs from template generation - template + transforms."""

    template: Path
    transforms: list[Path]


def generate_robust_template(in_files: Sequence[Path]) -> RobustTemplateOutputs:
    """Construct unbiased, robust template for longitudinal volumes with FreeSurfer.

    Uses an iterative method construct a mean volume and robust rigid registration
    of all input images to the current mean/median.

    Within-Subject Template Estimation for Unbiased Longitudinal Image Analysis
        M. Reuter, N.J. Schmansky, H.D. Rosas, B. Fischl.
        NeuroImage 61(4):1402-1418, 2012.
    """
    lta_files = []
    for in_file in in_files:
        if not Path(in_file).exists():
            raise FileNotFoundError(f"{in_file} not found.")
        fname = in_file.name.split(".")[0]
        lta_files.append(f"{fname}_to-template.lta")

    # Initialize with same defaults as fmriprep
    robust_template = freesurfer.mri_robust_template(
        mov=in_files,
        template="template.nii.gz",
        lta=lta_files,
        inittp=1,  # map everything to first time point
        fixtp=True,
        iscale=True,  # intensity scale (7-DOF - rigid + intensity)
        noit=True,  # no iteration; fmriprep turns this on -> why?
        satit=True,  # autodetect sensitivity
        subsample=200,  # subsample if any dimension has over this many volumes
    )

    return RobustTemplateOutputs(
        template=robust_template.template_output,
        transforms=[robust_template.root / lta for lta in lta_files],
    )


class ITKTransforms(NamedTuple):
    """Output of converted transformations from FS to ITK format."""

    transforms: list[Path]


def fs_to_ants_xfm(in_xfms: Sequence[Path]) -> list[Path]:
    """Convert Freesurfer transformations to ANTs compatible format."""
    result = ITKTransforms(transforms=[])
    for in_xfm in in_xfms:
        fname = in_xfm.with_suffix(".mat").name
        lta = freesurfer.lta_convert(in_lta=in_xfm, out_itk=fname)
        result.transforms.append(lta.root / fname)

    return result.transforms


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # 0. Setup
    fs_license = args.fs_license or os.getenv("FS_LICENSE")
    if fs_license is None or not Path(fs_license).exists():
        raise FileNotFoundError(f"Freesurfer license not found: {fs_license}")
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    mount_fs_license(ctx.runner, fs_license)

    # 1. Construct template
    ctx.logger.info("Starting processing")
    in_files = args.in_files
    if len(in_files) == 1:
        raise ValueError("Only a single volume found")
    ctx.logger.info("Building robust template")
    robust_template = generate_robust_template(in_files=in_files)

    # 2. Convert transformations to ANTs compatible format
    ctx.logger.info("Converting Freesurfer transformations to ANTs compatible format")
    subj_to_temp = fs_to_ants_xfm(robust_template.transforms)

    # 3. Save outputs
    ctx.logger.info("Saving files")
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy2(robust_template.template, args.output_file)
    for xfm in subj_to_temp:
        shutil.copy2(xfm, output_dir)
    ctx.logger.info("Robust template creation complete")
