"""Nuisance regression for fMRI data.

Orchestrates mask erosion, regressor assembly, and AFNI ``3dTproject``
to remove confound signals from BOLD timeseries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from niwrap import afni

from rbc.core.functional.mask_utils import (
    create_union_mask as create_union_mask,
)
from rbc.core.functional.mask_utils import (
    erode_brain_mask,
    erode_csf_mask,
    erode_wm_mask,
)
from rbc.core.functional.regressors import (
    assemble_36param_regressors,
    assemble_acompcor_regressors,
    compute_acompcor,
    extract_mean_signal,
    write_regressor_file,
)
from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from typing import Literal


class ErodedMaskArrays(NamedTuple):
    """In-memory eroded masks produced during nuisance regression."""

    csf: np.ndarray
    wm: np.ndarray
    brain: np.ndarray


class NuisanceRegressionOutputs(NamedTuple):
    """Outputs from :func:`nuisance_regression`."""

    regressed_bold: Path
    regressor_file: Path
    column_names: list[str]
    eroded_masks: ErodedMaskArrays


def bandpass_filter(
    bold: str | Path,
    brain_mask_file: str | Path,
    f_low: float = 0.01,
    f_high: float = 0.1,
) -> Path:
    """Apply bandpass filtering to a BOLD timeseries via AFNI 3dBandpass.

    Retains low-frequency fluctuations (default 0.01--0.1 Hz) while removing
    physiological noise and scanner drift. This is split out from nuisance
    regression so that ALFF/fALFF can be computed from the pre-bandpass
    residuals (where fALFF is meaningful).

    Args:
        bold: 4-D BOLD timeseries to filter.
        brain_mask_file: 3-D brain mask.
        f_low: Low frequency cutoff (Hz).
        f_high: High frequency cutoff (Hz).

    Returns:
        Path to bandpass-filtered BOLD timeseries.
    """
    result = afni.v_3d_bandpass(
        in_file=bold,
        mask=Path(brain_mask_file),
        prefix="bandpassed_bold.nii.gz",
        highpass=f_low,
        lowpass=f_high,
    )
    assert result.out_file is not None  # noqa: S101
    return result.out_file


def nuisance_regression(
    bold_file: str | Path,
    brain_mask_file: str | Path,
    csf_mask_file: str | Path,
    wm_mask_file: str | Path,
    motion_params: str | Path,
    regressor_set: Literal["36-parameter", "aCompCor"] = "36-parameter",
) -> NuisanceRegressionOutputs:
    """Run nuisance regression via AFNI 3dTproject.

    Steps:
        1. Load BOLD and tissue masks
        2. Erode masks (CSF 90%, WM 60%, brain 30mm)
        3. Load motion parameters from ``.1D`` file
        4. Extract tissue mean signals from eroded masks
        5. Assemble regressor matrix (36-param or aCompCor)
        6. Write ``.1D`` regressor file
        7. Call ``3dTproject``

    Args:
        bold_file: 4-D BOLD timeseries.
        brain_mask_file: 3-D brain mask.
        csf_mask_file: 3-D CSF tissue mask.
        wm_mask_file: 3-D WM tissue mask.
        motion_params: AFNI-format ``.1D`` file (T rows x 6 columns).
        regressor_set: ``"36-parameter"`` or ``"aCompCor"``.

    Returns:
        :class:`NuisanceRegressionOutputs` with regressed BOLD path,
        regressor file path, column names, and eroded masks.
    """
    import nibabel as nib

    from rbc.core.functional.regressors import check_regressor_rank

    out_dir = generate_exec_folder("nuisance_regression")

    # 1. Load data
    bold_img = nib.nifti1.load(bold_file)
    bold_data = bold_img.get_fdata()

    brain_mask = nib.nifti1.load(brain_mask_file).get_fdata()
    csf_mask = nib.nifti1.load(csf_mask_file).get_fdata()
    wm_mask = nib.nifti1.load(wm_mask_file).get_fdata()

    # 2. Erode masks
    csf_eroded = erode_csf_mask(csf_mask)
    wm_eroded = erode_wm_mask(wm_mask)

    voxel_sizes = tuple(float(v) for v in bold_img.header.get_zooms()[:3])
    brain_eroded = erode_brain_mask(brain_mask, voxel_sizes)  # type: ignore[arg-type]

    eroded = ErodedMaskArrays(csf=csf_eroded, wm=wm_eroded, brain=brain_eroded)

    # 3. Load motion parameters
    motion_params_data = np.loadtxt(motion_params)

    # 4. Extract tissue mean signals
    csf_signal = extract_mean_signal(bold_data, csf_eroded)
    wm_signal = extract_mean_signal(bold_data, wm_eroded)

    # 5. Assemble regressors
    if regressor_set == "36-parameter":
        global_signal = extract_mean_signal(bold_data, brain_eroded)
        matrix, column_names = assemble_36param_regressors(
            motion_params_data, csf_signal, wm_signal, global_signal
        )
    elif regressor_set == "aCompCor":
        union_mask = (csf_eroded | wm_eroded).astype(np.uint8)
        acompcor_components = compute_acompcor(bold_data, union_mask)
        matrix, column_names = assemble_acompcor_regressors(
            motion_params_data, csf_signal, wm_signal, acompcor_components
        )
    else:
        raise ValueError(
            f"Unknown regressor_set {regressor_set!r}, "
            "expected '36-parameter' or 'aCompCor'"
        )

    # 6. Check conditioning and write regressor file
    check_regressor_rank(matrix, column_names)
    regressor_file = out_dir / "regressors.1D"
    write_regressor_file(matrix, column_names, regressor_file)

    # 7. Call 3dTproject
    result = afni.v_3d_tproject(
        in_file=Path(bold_file),
        prefix="regressed_bold.nii.gz",
        polort=0,
        ort=regressor_file,
        mask=Path(brain_mask_file),
        norm=False,
    )

    return NuisanceRegressionOutputs(
        regressed_bold=Path(result.out_file),
        regressor_file=regressor_file,
        column_names=column_names,
        eroded_masks=eroded,
    )
