"""ALFF and fALFF computation for fMRI data.

Computes voxelwise Amplitude of Low-Frequency Fluctuations (ALFF) and
fractional ALFF (fALFF) from resting-state fMRI.  Two variants are provided:

- **amALFF** (arithmetic-mean): Original Zang et al. 2007 definition using
  mean of FFT amplitude coefficients within a frequency band.
- **qmALFF** (quadratic-mean / std-deviation): C-PAC-style approximation
  using temporal standard deviation of a bandpass-filtered signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rbc.core.nifti import Volume

if TYPE_CHECKING:
    from typing import Literal

    AlffMethod = Literal["am", "qm"]


def compute_frequency_bins(
    n_timepoints: int, tr: float, f_low: float, f_high: float
) -> np.ndarray:
    """Map a frequency band to FFT bin indices.

    Args:
        n_timepoints: Number of timepoints (T).
        tr: Repetition time in seconds.
        f_low: Lower frequency bound (Hz), inclusive.
        f_high: Upper frequency bound (Hz), inclusive.

    Returns:
        1-D integer array of bin indices whose frequencies fall within
        ``[f_low, f_high]``.

    Raises:
        ValueError: If *tr* is not positive, *f_low >= f_high*, or no
            bins fall within the requested band.
    """
    if tr <= 0:
        raise ValueError(f"TR must be positive, got {tr}")
    if f_low >= f_high:
        raise ValueError(f"f_low must be < f_high, got {f_low} >= {f_high}")

    freqs = np.fft.rfftfreq(n_timepoints, d=tr)
    idx_low = int(np.searchsorted(freqs, f_low, side="left"))
    idx_high = int(np.searchsorted(freqs, f_high, side="right"))
    bins = np.arange(idx_low, idx_high)

    if bins.size == 0:
        raise ValueError(
            f"No FFT bins in [{f_low}, {f_high}] Hz "
            f"(freq resolution {1 / (n_timepoints * tr):.4f} Hz, "
            f"Nyquist {1 / (2 * tr):.4f} Hz)"
        )
    return bins


def am_alff(
    data: np.ndarray,
    mask: np.ndarray,
    tr: float,
    f_low: float = 0.01,
    f_high: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-mean ALFF and fALFF (Zang et al. 2007).

    ALFF is the mean of the FFT amplitude spectrum within ``[f_low, f_high]``.
    fALFF is the ratio of the summed in-band amplitudes to the total amplitude
    across all frequencies.

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible brain mask.
        tr: Repetition time in seconds.
        f_low: Lower frequency bound (Hz).
        f_high: Upper frequency bound (Hz).

    Returns:
        ``(alff_map, falff_map)`` - both ``(X, Y, Z)`` float64 arrays.

    Raises:
        ValueError: If *data* is not 4-D.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    mask = mask > 0
    nx, ny, nz, nt = data.shape
    bins = compute_frequency_bins(nt, tr, f_low, f_high)

    alff_map = np.zeros((nx, ny, nz), dtype=np.float64)
    falff_map = np.zeros((nx, ny, nz), dtype=np.float64)

    if not np.any(mask):
        return alff_map, falff_map

    flat = data.reshape(-1, nt)
    mask_flat = mask.ravel()
    indices = np.where(mask_flat)[0]

    spectra = np.abs(np.fft.rfft(flat[indices], axis=1))

    band_amplitudes = spectra[:, bins]
    alff_vals = band_amplitudes.mean(axis=1)
    alff_map.ravel()[indices] = alff_vals

    total = spectra.sum(axis=1)
    band_sum = band_amplitudes.sum(axis=1)
    safe = total > 0
    falff_vals = np.zeros(len(indices), dtype=np.float64)
    falff_vals[safe] = band_sum[safe] / total[safe]
    falff_map.ravel()[indices] = falff_vals

    return alff_map, falff_map


def qm_alff(
    data: np.ndarray,
    mask: np.ndarray,
    tr: float,
    f_low: float = 0.01,
    f_high: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic-mean (std-deviation) ALFF and fALFF (C-PAC style).

    ALFF is the temporal standard deviation of the bandpass-filtered signal.
    fALFF is the ratio of filtered-signal std to unfiltered-signal std.

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible brain mask.
        tr: Repetition time in seconds.
        f_low: Lower frequency bound (Hz).
        f_high: Upper frequency bound (Hz).

    Returns:
        ``(alff_map, falff_map)`` - both ``(X, Y, Z)`` float64 arrays.

    Raises:
        ValueError: If *data* is not 4-D.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    mask = mask > 0
    nx, ny, nz, nt = data.shape
    bins = compute_frequency_bins(nt, tr, f_low, f_high)

    alff_map = np.zeros((nx, ny, nz), dtype=np.float64)
    falff_map = np.zeros((nx, ny, nz), dtype=np.float64)

    if not np.any(mask):
        return alff_map, falff_map

    flat = data.reshape(-1, nt)
    mask_flat = mask.ravel()
    indices = np.where(mask_flat)[0]

    spectra = np.fft.rfft(flat[indices], axis=1)

    bandpass_mask = np.zeros(spectra.shape[1], dtype=bool)
    bandpass_mask[bins] = True

    filtered_spectra = np.zeros_like(spectra)
    filtered_spectra[:, bandpass_mask] = spectra[:, bandpass_mask]
    filtered = np.fft.irfft(filtered_spectra, n=nt, axis=1)

    std_unfiltered = flat[indices].std(axis=1, ddof=0)
    std_filtered = filtered.std(axis=1, ddof=0)

    alff_map.ravel()[indices] = std_filtered

    safe = std_unfiltered > 0
    falff_vals = np.zeros(len(indices), dtype=np.float64)
    falff_vals[safe] = std_filtered[safe] / std_unfiltered[safe]
    falff_map.ravel()[indices] = falff_vals

    return alff_map, falff_map


def alff(
    data: np.ndarray,
    mask: np.ndarray,
    tr: float,
    f_low: float = 0.01,
    f_high: float = 0.1,
    method: AlffMethod = "am",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ALFF and fALFF maps from 4-D fMRI data.

    Dispatcher that delegates to :func:`am_alff` or :func:`qm_alff`
    depending on *method*.

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible brain mask.
        tr: Repetition time in seconds.
        f_low: Lower frequency bound (Hz).
        f_high: Upper frequency bound (Hz).
        method: ``"am"`` for amplitude-mean (Zang 2007) or ``"qm"`` for
            quadratic-mean / std-deviation (C-PAC style).

    Returns:
        ``(alff_map, falff_map)`` - both ``(X, Y, Z)`` float64 arrays.

    Raises:
        ValueError: If *method* is not ``"am"`` or ``"qm"``.
    """
    if method == "am":
        return am_alff(data, mask, tr, f_low, f_high)
    if method == "qm":
        return qm_alff(data, mask, tr, f_low, f_high)
    raise ValueError(f"Unknown method {method!r}, expected 'am' or 'qm'")


def compute_alff(
    in_file: str | Path,
    mask_file: str | Path,
    tr: float | None = None,
    f_low: float = 0.01,
    f_high: float = 0.1,
    method: AlffMethod = "am",
    out_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Load NIfTI files, compute ALFF/fALFF, and write results to disk.

    Thin wrapper around :func:`alff` that handles file I/O.

    Args:
        in_file: Path to 4-D NIfTI ``(X, Y, Z, T)``.
        mask_file: Path to 3-D binary brain mask in the same space.
        tr: Repetition time in seconds.  If *None*, read from the NIfTI
            header (``pixdim[4]``).
        f_low: Lower frequency bound (Hz).
        f_high: Upper frequency bound (Hz).
        method: ``"am"`` or ``"qm"``.
        out_file: Output path for the ALFF map.  Defaults to
            ``<in_file stem>_alff.nii.gz``.  The fALFF map is written
            alongside with ``_falff`` suffix.

    Returns:
        ``(alff_path, falff_path)``.
    """
    in_file = Path(in_file)
    bold = Volume.load(in_file, dtype=np.float64, expected_ndim=4)
    mask = Volume.load(mask_file, dtype=np.uint8)
    bold.check_compatible(mask)

    effective_tr = tr if tr is not None else bold.tr
    assert effective_tr is not None  # noqa: S101 - guaranteed by expected_ndim=4
    alff_map, falff_map = alff(
        bold.data, mask.data, effective_tr, f_low, f_high, method
    )

    stem = in_file.name.split(".nii")[0]
    if out_file is None:
        alff_path = in_file.parent / f"{stem}_alff.nii.gz"
    else:
        alff_path = Path(out_file)

    falff_path = alff_path.parent / alff_path.name.replace("_alff", "_falff")
    if falff_path == alff_path:
        falff_path = alff_path.parent / f"{stem}_falff.nii.gz"

    bold.derive(alff_map).save(alff_path)
    bold.derive(falff_map).save(falff_path)
    return alff_path, falff_path
