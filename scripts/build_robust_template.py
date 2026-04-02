# /// script
# dependencies = [
#     "niwrap>=0.9.1",
#     "polars>=1.38.1",
#     "rbc",
#     "tqdm>=4.67.3",
# ]
# requires-python = ">=3.12"
#
# [tool.uv.sources]
# rbc = { git = "https://github.com/childmindresearch/rbc-mirror" }
#
# ///
"""Generate a robust, longitudinal T1w template using Freesurfer's mri_robust_template.

Run with:
    uv run scripts/build_robust_template.py <data_dir> <output_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import polars as pl
from niwrap import c3d, Runner, freesurfer
from rbc.cli import _DEFAULT_ENV_VARS
from rbc.cli.base import BaseArgs
from rbc.cli.main import _global_opts
from rbc.context import PipelineContext
from rbc.core.bids import Datatype, Suffix
from rbc.core.bids2table import load_table
from rbc.core.niwrap import setup_runner
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

CONTAINER_LICENSE_PATH = "/opt/freesurfer/license.txt"


def create_parser() -> argparse.ArgumentParser:
    """Create parser for template creation."""
    parser = argparse.ArgumentParser(
        prog="create_template",
        description="Create a robust template using Freesurfer's mri_robust_template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s input_dir output_dir [options]",
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
    _global_opts()
    parser.add_argument(
        "--fs-license",
        required=False,
        type=Path,
        help="Path to Freesurfer license",
    )
    return parser

def _get_mount_arg(runner: str, src: str, dst: str) -> list[str]:
    """Return runner-specific mount CLI args."""
    if runner in ("podman", "docker"):
        return ["--mount", f"type=bind,source={src},target={dst},readonly"]
    return ["--bind", f"{src}:{dst}"]  # singularity


def mount_fs_license(runner: Runner, fs_license: str) -> None:
    """Mount FreeSurfer license file into an existing runner."""
    runner_name = type(runner).__name__.lower().replace("runner", "")

    if runner_name == "local":
        os.environ["FS_LICENSE"] = fs_license
        return

    extra_args_attr = f"{runner_name}_extra_args"
    getattr(runner, extra_args_attr).extend(
        _get_mount_arg(runner=runner_name, src=fs_license, dst=CONTAINER_LICENSE_PATH)
    )
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
    for idx, in_file in enumerate(in_files):
        if not in_file.exists():
            raise FileNotFoundError(f"Input file not found: {in_file}.")
        lta_files.append(f"xfm_{idx:04d}.lta")

    # Initialize with same defaults as fmriprep
    assert entities is not None, "No entities found"
    robust_template = freesurfer.mri_robust_template(
        mov=list(in_files),
        template=f"long_template_T1w.nii.gz",
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
    for src_file, in_xfm in zip(src_files, in_xfms, strict=True):
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


@dataclass(frozen=True)
class TemplateArgs(BaseArgs):
    """Arguments for template-building CLI."""

    fs_license: str

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> TemplateArgs:
        """Validation of template-building script specific arguments to NamedTuple."""
        fs_license = ns.fs_license or os.getenv("FS_LICENSE")
        if fs_license is None or not Path(fs_license).exists():
            raise ValueError(f"FreeSurfer license file not found: {fs_license}")
        return cls(
            **BaseArgs.validate_namespace(ns).__dict__,
            fs_license=str(fs_license)
        )

def process(args: TemplateArgs) -> int:
    """Main processing layer for script."""
    # 1. Setup
    ctx = setup_runner(runner=args.runner, verbose=args.verbose, tmp_dir=args.tmp_dir)
    ctx.runner.environ = {**_DEFAULT_ENV_VARS, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    ctx.logger.warning(
        "This script is experimental and may be sensitive to file naming conventions."
    )
    mount_fs_license(ctx.runner, args.fs_license)

    ctx.logger.info("Preparing to generate longitudinal templates")
    df = load_table(
        dataset_dir=args.input_dir, index_fpath=None, max_workers=0, verbose=ctx.verbose
    )

    filters = [
        pl.col("ses") != "longitudinal",
        pl.col("datatype") == "anat",
        pl.col("space").is_null(),
        pl.col("desc") == "brain",
        pl.col("suffix") == "T1w",
    ]
    if len(args.participant_label) > 0:
        filters.append(pl.col("sub").is_in(args.participant_label))
    if len(args.session_label) > 0:
        filters.append(pl.col("ses").is_in(args.session_label))
    df = df.filter(pl.all_horizontal(filters))

    for _, sub_group in tqdm(
        df.group_by(("sub"), maintain_order=True), disable=not ctx.verbose
    ):
        sessions = sub_group["ses"].to_list()
        pipe_ctx = PipelineContext(
            sub=sub_group["sub"][0], ses=None, output_dir=args.output_dir
        )

        # 2. Construct template
        ctx.logger.info(f"Building robust template for subject: {pipe_ctx.sub}")
        if len(sub_group) <= 1:
            raise ValueError("At least 2 volumes needed to generate a template.")
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
        long = pipe_ctx.bids(datatype=Datatype.ANAT).derive(ses="longitudinal")
        long.save(robust_template.template, suffix=Suffix.T1W, desc="brain")
        for idx, fpath in enumerate(subj_to_temp):
            long.save(fpath, session=sessions[idx], suffix="xfm", extension=".lta")

    ctx.logger.info("Robust template creation complete")
    return 0


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    process(TemplateArgs.validate_namespace(args))
