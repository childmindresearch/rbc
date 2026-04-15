"""FreeSurfer-based robust template construction for longitudinal data.

Builds an unbiased within-subject template via ``mri_robust_template`` and
converts the resulting LTA transforms to ITK format consumable by ANTs.

Reference:
    Within-Subject Template Estimation for Unbiased Longitudinal Image Analysis
    M. Reuter, N.J. Schmansky, H.D. Rosas, B. Fischl.
    NeuroImage 61(4):1402-1418, 2012.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from niwrap import freesurfer

from rbc.core.fsl2itk import mat_to_itk
from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class RobustTemplateOutputs(NamedTuple):
    """Outputs from FreeSurfer robust template construction.

    Attributes:
        template: Robust template volume.
        transforms: Per-input LTA transforms mapping each input to the template.
    """

    template: Path
    transforms: list[Path]


def lta_filename(sub: str, ses: str) -> str:
    """Build the LTA filename for a single session-to-template transform.

    The naming follows BIDS xfm convention with the subject/session entities
    plus an extra ``from-<ses>_to-longitudinal`` pair.
    """
    return f"sub-{sub}_ses-{ses}_from-{ses}_to-longitudinal_mode-image_xfm.lta"


def itk_filename(sub: str, ses: str) -> str:
    """ITK-format counterpart of :func:`lta_filename`."""
    return f"sub-{sub}_ses-{ses}_from-{ses}_to-longitudinal_mode-image_xfm.txt"


def template_filename(sub: str) -> str:
    """Filename for the robust template volume."""
    return f"sub-{sub}_ses-longitudinal_T1w.nii.gz"


def generate_robust_template(
    sub: str,
    sessions: Sequence[str],
    in_files: Sequence[Path],
) -> RobustTemplateOutputs:
    """Construct an unbiased robust template for a single subject.

    Uses an iterative method to construct a mean volume and robust rigid
    registration of all input images to the current mean/median.

    Args:
        sub: Subject label (without the ``sub-`` prefix).
        sessions: Session label per input volume (parallel to ``in_files``).
        in_files: Per-session preprocessed T1w volumes.

    Returns:
        :class:`RobustTemplateOutputs` with the template and per-session LTA
        transforms.

    Raises:
        FileNotFoundError: If any input volume does not exist.
        ValueError: If fewer than two volumes are provided or if the number
            of sessions and input files differ.
    """
    if len(in_files) != len(sessions):
        raise ValueError(
            f"sessions ({len(sessions)}) and in_files ({len(in_files)}) "
            "must have the same length."
        )
    if len(in_files) < 2:
        raise ValueError(f"At least 2 input volumes required, got {len(in_files)}.")
    for in_file in in_files:
        if not in_file.exists():
            raise FileNotFoundError(f"{in_file} not found.")

    lta_files = [lta_filename(sub, ses) for ses in sessions]

    # Initialize with same defaults as fmriprep.
    # TODO(#302): noit=True inherited from fmriprep; quality impact unknown.
    robust_template = freesurfer.mri_robust_template(
        mov=list(in_files),
        template=template_filename(sub),
        lta=lta_files,
        inittp=1,  # map everything to first time point
        fixtp=True,
        iscale=True,  # intensity scale (7-DOF: rigid + intensity)
        noit=True,
        satit=True,  # autodetect sensitivity
        subsample=200,  # subsample if any dimension exceeds this many voxels
    )

    return RobustTemplateOutputs(
        template=robust_template.template_output,
        transforms=[robust_template.root / lta for lta in lta_files],
    )


def fs_to_itk_xfm(
    sub: str,
    sessions: Sequence[str],
    reference: Path,
    sources: Sequence[Path],
    in_xfms: Sequence[Path],
) -> list[Path]:
    """Convert FreeSurfer LTA transforms to ITK ``.txt`` format.

    The conversion goes ``freesurfer -> fsl -> itk`` (see
    https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg55547.html).

    Args:
        sub: Subject label.
        sessions: Per-input session labels (parallel to ``sources`` / ``in_xfms``).
        reference: Reference (fixed) image, typically the robust template.
        sources: Per-session source (moving) images.
        in_xfms: Per-session FreeSurfer LTA transforms.

    Returns:
        Per-session ITK ``.txt`` transform files, written next to the LTAs.
    """
    if not (len(sessions) == len(sources) == len(in_xfms)):
        raise ValueError("sessions, sources, and in_xfms must have the same length.")

    itk_xfms: list[Path] = []
    for ses, source, in_xfm in zip(sessions, sources, in_xfms, strict=True):
        fsl_fname = in_xfm.with_suffix(".mat").name
        lta = freesurfer.lta_convert(in_lta=in_xfm, out_fsl=fsl_fname)
        itk_path = generate_exec_folder("itk_xfm") / itk_filename(sub, ses)
        mat_to_itk(
            mat=lta.root / fsl_fname,
            reference=reference,
            source=source,
            output=itk_path,
        )
        itk_xfms.append(itk_path)

    return itk_xfms
