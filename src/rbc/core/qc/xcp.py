"""XCP-style quality control output.

Aggregates motion, DVARS, and registration QC metrics into a single
TSV file following the XCP-D output format, with RBC QC threshold
checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from rbc.core.qc.dvars import DVARSQCMetrics
    from rbc.core.qc.motion import MotionQCMetrics
    from rbc.core.qc.registration import RegistrationQCMetrics


class XCPQCMetrics(NamedTuple):
    """All XCP-style QC columns for a single functional run.

    Attributes:
        sub: Subject ID.
        ses: Session label.
        task: Task label.
        run: Run number.
        desc: Description label (e.g. pipeline variant).
        regressors: Nuisance regressor strategy name.
        space: Target space name.
        meanFD: Mean framewise displacement (Jenkinson), mm.
        relMeansRMSMotion: Mean RMS of translation parameters.
        relMaxRMSMotion: Max RMS of translation parameters.
        nVolCensored: Number of volumes exceeding the FD threshold.
        meanDVInit: Mean DVARS before denoising.
        meanDVFinal: Mean DVARS after denoising.
        nVolsRemoved: Number of volumes removed (e.g. non-steady-state).
        motionDVCorrInit: Motion-DVARS correlation before denoising.
        motionDVCorrFinal: Motion-DVARS correlation after denoising.
        coregDice: Coregistration Dice coefficient.
        coregJaccard: Coregistration Jaccard index.
        coregCrossCorr: Coregistration cross-correlation.
        coregCoverage: Coregistration coverage.
        normDice: Normalization Dice coefficient.
        normJaccard: Normalization Jaccard index.
        normCrossCorr: Normalization cross-correlation.
        normCoverage: Normalization coverage.
    """

    sub: str
    ses: str
    task: str
    run: int
    desc: str
    regressors: str
    space: str
    meanFD: float
    relMeansRMSMotion: float
    relMaxRMSMotion: float
    nVolCensored: int
    meanDVInit: float
    meanDVFinal: float
    nVolsRemoved: int
    motionDVCorrInit: float
    motionDVCorrFinal: float
    coregDice: float
    coregJaccard: float
    coregCrossCorr: float
    coregCoverage: float
    normDice: float
    normJaccard: float
    normCrossCorr: float
    normCoverage: float


def generate_xcp_qc(
    *,
    sub: str,
    ses: str,
    task: str,
    run: int,
    desc: str,
    regressors: str,
    space: str,
    motion: MotionQCMetrics,
    dvars_init: DVARSQCMetrics,
    dvars_final: DVARSQCMetrics,
    n_vols_removed: int,
    coreg: RegistrationQCMetrics,
    norm: RegistrationQCMetrics,
) -> XCPQCMetrics:
    """Assemble sub-metrics into a single XCP QC row.

    This is a pure assembly function that maps fields from the
    individual QC NamedTuples into the unified :class:`XCPQCMetrics`.

    Args:
        sub: Subject ID.
        ses: Session label.
        task: Task label.
        run: Run number.
        desc: Description label.
        regressors: Nuisance regressor strategy name.
        space: Target space name.
        motion: Pre-computed motion QC metrics.
        dvars_init: DVARS metrics before denoising.
        dvars_final: DVARS metrics after denoising.
        n_vols_removed: Number of volumes removed.
        coreg: Coregistration overlap metrics.
        norm: Normalization overlap metrics.

    Returns:
        A single :class:`XCPQCMetrics` row ready for TSV output.
    """
    return XCPQCMetrics(
        sub=sub,
        ses=ses,
        task=task,
        run=run,
        desc=desc,
        regressors=regressors,
        space=space,
        meanFD=motion.mean_fd,
        relMeansRMSMotion=motion.rel_means_rms_motion,
        relMaxRMSMotion=motion.rel_max_rms_motion,
        nVolCensored=motion.n_vol_censored,
        meanDVInit=dvars_init.mean_dvars,
        meanDVFinal=dvars_final.mean_dvars,
        nVolsRemoved=n_vols_removed,
        motionDVCorrInit=dvars_init.motion_dvars_corr,
        motionDVCorrFinal=dvars_final.motion_dvars_corr,
        coregDice=coreg.dice,
        coregJaccard=coreg.jaccard,
        coregCrossCorr=coreg.cross_corr,
        coregCoverage=coreg.coverage,
        normDice=norm.dice,
        normJaccard=norm.jaccard,
        normCrossCorr=norm.cross_corr,
        normCoverage=norm.coverage,
    )


def write_xcp_qc(metrics: XCPQCMetrics, out_path: Path) -> Path:
    """Write XCP QC metrics as a single-row TSV file.

    Args:
        metrics: A populated :class:`XCPQCMetrics` row.
        out_path: Destination file path (parent dirs created if needed).

    Returns:
        The output path (same as *out_path*).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame([metrics._asdict()])
    df.write_csv(out_path, separator="\t")

    return out_path


# QC pass/fail thresholds (single source of truth; see :func:`passes_rbc_qc`).
FD_THRESHOLD_MM = 0.2
NORM_CROSS_CORR_THRESHOLD = 0.8


def passes_rbc_qc(fd: np.ndarray, norm_cross_corr: float) -> bool:
    """Check whether a run passes RBC quality thresholds.

    A run passes if **both** conditions are met:

    - ``median(fd) <= FD_THRESHOLD_MM``
    - ``norm_cross_corr >= NORM_CROSS_CORR_THRESHOLD``

    Args:
        fd: Full framewise displacement timeseries (for median).
        norm_cross_corr: Normalization cross-correlation value.

    Returns:
        ``True`` if the run passes both thresholds.
    """
    fd = np.asarray(fd, dtype=np.float64).ravel()
    return bool(
        float(np.median(fd)) <= FD_THRESHOLD_MM
        and norm_cross_corr >= NORM_CROSS_CORR_THRESHOLD
    )
