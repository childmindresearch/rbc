# /// script
# dependencies = [
#     "bids2table>=2.1.2",
#     "niwrap>=0.9.1",
#     "polars>=1.38.1",
#     "styxpodman",
#     "tqdm>=4.67.3",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# styxpodman = { git = "https://github.com/styx-api/styxpodman", rev = "1382977" }
#
# ///
"""Generate a robust, longitudinal T1w template using Freesurfer's mri_robust_template.

Run with:
    uv run scripts/build_robust_template.py <data_dir> <output_dir>
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import bids2table as b2t
import polars as pl
from niwrap import (
    c3d,    # uv script dependency with private repo; using c3d to convert transform
    Runner,
    freesurfer,
    get_global_runner,
    set_global_runner,
    use_docker,
    use_local,
    use_singularity,
)
from styxpodman import PodmanRunner
from tqdm import tqdm

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
        "input_dir",
        type=Path,
        help="BIDS-organized input dataset directory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where output data should be stored",
    )
    parser.add_argument(
        "--fs-license",
        required=False,
        type=Path,
        help="Path to Freesurfer license",
    )
    parser.add_argument(
        "--participant-label",
        nargs="+",
        default=[],
        type=lambda x: x.removeprefix("sub-"),
        help="Space-delimited participant identifier ('sub-' prefix can be removed)",
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
    entities = None
    for in_file in in_files:
        if not Path(in_file).exists():
            raise FileNotFoundError(f"{in_file} not found.")
        entities = b2t.parse_bids_entities(in_file)
        lta_fname = b2t.format_bids_path(
            {
                "sub": entities["sub"],
                "ses": "longitudinal",
                "from": entities["ses"],
                "suffix": "xfm",
                "ext": ".lta",
            }
        ).name
        lta_files.append(lta_fname)

    # Initialize with same defaults as fmriprep
    assert entities is not None, "No entities found"
    robust_template = freesurfer.mri_robust_template(
        mov=list(in_files),
        template=f"sub-{entities['sub']}_ses-longitudinal_T1w.nii.gz",
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


def fs_to_ants_xfm(
    ref_file: Path, src_files: Sequence[Path], in_xfms: Sequence[Path]
) -> list[Path]:
    """Convert Freesurfer transformations to ANTs compatible format.

    freesurfer -> fsl -> itk
    (see https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg55547.html)
    """
    result = ITKTransforms(transforms=[])
    for src_file, in_xfm in zip(src_files, in_xfms):
        fsl_fname = in_xfm.with_suffix(".mat").name
        lta = freesurfer.lta_convert(in_lta=in_xfm, out_fsl=fsl_fname)
        fsl2itk = c3d.c3d_affine_tool(
            transform_file=lta.root / fsl_fname,
            source_file=src_file,
            reference_file=ref_file,
            fsl2ras=True,
            out_itk_transform=in_xfm.with_suffix(".txt").name
        )
        assert fsl2itk.itk_transform_outfile
        result.transforms.append(fsl2itk.itk_transform_outfile)

    return result.transforms


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # 1. Setup
    fs_license = args.fs_license or os.getenv("FS_LICENSE")
    if fs_license is None or not Path(fs_license).exists():
        raise FileNotFoundError(f"Freesurfer license not found: {fs_license}")
    ctx = setup_runner(runner=args.runner, verbose=args.verbose)
    # Taken from rbc's _DEFAULT_ENVS (uses CPAC ANTs seed)
    ctx.runner.environ = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
        "ANTS_RANDOM_SEED": 77742777,
        "FSLOUTPUTTYPE": "NIFTI_GZ"  # Needed for FreeSurfer xfm conversion
    }
    ctx.logger.warning(
        "This script is experimental and may be sensitive to input file naming "
        "conventions."
    )
    mount_fs_license(ctx.runner, fs_license)

    ctx.logger.info("Preparing to generate longitudinal templates")
    tables = b2t.batch_index_dataset(
        b2t.find_bids_datasets(args.input_dir),
        max_workers=0,
        show_progress=ctx.verbose,
    )
    dfs: list[pl.DataFrame] = []
    for table in tables:
        result = pl.from_arrow(table)
        if not isinstance(result, pl.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(result)}")
        dfs.append(result)
    df = pl.concat(dfs)
    # Filters for preprocessed T1w to create longitudinal template
    filters = [
        pl.col("ses") != "longitudinal",
        pl.col("datatype") == "anat",
        pl.col("desc") == "brain",
        pl.col("suffix") == "T1w",
    ]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    df = df.filter(pl.all_horizontal(filters))
    del dfs

    ctx.logger.info("Starting processing")
    if len(df) == 1:
        raise ValueError("Only a single volume found")
    for _, sub_group in tqdm(df.group_by("sub"), disable=not ctx.verbose):
        # 2. Construct template
        sub = sub_group["sub"][0]
        ctx.logger.info(f"Building robust template for sub-{sub}")
        in_files = [
            Path(row["root"]) / row["path"] for row in sub_group.iter_rows(named=True)
        ]
        robust_template = generate_robust_template(in_files=in_files)

        # 3. Convert transformations to ANTs compatible format
        ctx.logger.info("Converting Freesurfer transformations to ANTs format")
        subj_to_temp = fs_to_ants_xfm(
            ref_file=robust_template.template,
            src_files=in_files,
            in_xfms=robust_template.transforms
        )

        # 4. Save outputs
        ctx.logger.info("Saving files")
        output_dir = (
            Path(args.output_dir)
            / b2t.format_bids_path(
                {"sub": sub, "ses": "longitudinal", "datatype": "anat"}
            ).parent
        )
        output_dir.mkdir(exist_ok=True, parents=True)
        for fpath in [robust_template.template, *subj_to_temp]:
            shutil.copy2(fpath, output_dir)
        ctx.logger.info("Robust template creation complete")
    ctx.logger.info("Completed creating all templates.")
