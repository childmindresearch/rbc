"""Regional Homogeneity (ReHo) computation for fMRI data.

Computes voxelwise Kendall's W across local neighborhoods to measure
temporal synchronization in resting-state fMRI. Supports 7, 19, and 27
voxel neighborhoods corresponding to face, face+edge, and full cube
connectivity respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from rbc.core.nifti import Volume

if TYPE_CHECKING:
    from typing import Literal

    ClusterSize = Literal[7, 19, 27]


def kendall_w(ranks: np.ndarray) -> float:
    """Kendall's coefficient of concordance (W).

    W = 12S / (K^2 * (N^3 - N)) where S = sum of squared deviations of
    rank sums from their mean, K = number of judges (neighbor voxels),
    N = number of subjects (timepoints).

    Args:
        ranks: (N, K) array of ranks.

    Returns:
        Kendall's W statistic.
    """
    n, k = ranks.shape
    rank_sums = ranks.sum(axis=1)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    return 12.0 * s / (k**2 * (n**3 - n))


def rank_timeseries(data: np.ndarray) -> np.ndarray:
    """Rank each voxel's timeseries independently along the time axis.

    Uses segmented processing to limit peak memory to ~2x one volume
    instead of holding the full ranked 4D array at once.

    Args:
        data: (X, Y, Z, T) array.

    Returns:
        (X, Y, Z, T) float32 array of ranks.
    """
    nx, ny, nz, nt = data.shape
    flat = data.reshape(-1, nt)
    ranked = np.empty_like(flat, dtype=np.float32)

    chunk_size = max(1, nx * ny)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        ranked[start:end] = np.apply_along_axis(rankdata, 1, flat[start:end])

    return ranked.reshape(nx, ny, nz, nt)


def get_neighbor_offsets(cluster_size: ClusterSize) -> np.ndarray:
    """3D neighbor offsets for a given connectivity.

    Args:
        cluster_size: 7 (face/6-connectivity + center), 19 (face+edge/
            18-connectivity + center), or 27 (full cube/26-connectivity
            + center).

    Returns:
        (N, 3) int array of (di, dj, dk) displacements.
    """
    offsets = np.array(
        [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)]
    )
    l1 = np.sum(np.abs(offsets), axis=1)

    if cluster_size == 7:
        keep = l1 <= 1
    elif cluster_size == 19:
        keep = l1 <= 2
        keep &= ~np.all(np.abs(offsets) == 1, axis=1)
    else:
        keep = np.ones(27, dtype=bool)

    return offsets[keep]


def reho(
    data: np.ndarray, mask: np.ndarray, cluster_size: ClusterSize = 27
) -> np.ndarray:
    """Compute Regional Homogeneity (ReHo) map from 4D fMRI data.

    ReHo measures local synchronization by computing Kendall's W across
    each voxel's timeseries and its spatial neighbors' timeseries.

    The voxelwise neighborhood loop is the dominant bottleneck. On typical
    whole-brain data (~200k in-mask voxels, ~200 timepoints) this pure-Python
    loop takes on the order of minutes. For production use, the inner loop
    (neighbor gathering + KCC computation) is a good candidate for a PyO3/Rust
    or Cython extension -- the algorithm is embarrassingly parallel over voxels
    and the per-voxel work (rank-sum and a single division) is tiny, so the
    speedup from eliminating Python overhead is ~50-100x.

    References:
        Zang et al. (2004). Regional homogeneity approach to fMRI data
        analysis. NeuroImage, 22(1), 394-400.

    Args:
        data: 4D array (X, Y, Z, T).
        mask: 3D boolean-compatible brain mask in the same space.
        cluster_size: Neighborhood size -- 7 (face), 19 (face+edge),
            or 27 (full cube).

    Returns:
        (X, Y, Z) float64 array of Kendall's W values.

    Raises:
        ValueError: If data is not 4D.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    mask = mask > 0
    nx, ny, nz, _nt = data.shape
    offsets = get_neighbor_offsets(cluster_size)
    ranks = rank_timeseries(data)

    reho_map = np.zeros((nx, ny, nz), dtype=np.float64)

    pad_ranks = np.pad(ranks, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="constant")
    pad_mask = np.pad(mask, 1, mode="constant", constant_values=False)

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                if not pad_mask[i, j, k]:
                    continue

                neighbor_ranks = []
                for di, dj, dk in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if pad_mask[ni, nj, nk]:
                        neighbor_ranks.append(pad_ranks[ni, nj, nk, :])

                if len(neighbor_ranks) < 2:
                    continue

                reho_map[i - 1, j - 1, k - 1] = kendall_w(
                    np.column_stack(neighbor_ranks)
                )

    return reho_map


def compute_reho(
    in_file: str | Path,
    mask_file: str | Path,
    cluster_size: ClusterSize = 27,
    out_file: str | Path | None = None,
) -> Path:
    """Load NIfTI files, compute ReHo, and write the result to disk.

    Thin wrapper around ``reho()`` that handles file I/O.

    Args:
        in_file: Path to 4D NIfTI (X, Y, Z, T).
        mask_file: Path to 3D binary brain mask in the same space.
        cluster_size: Neighborhood size -- 7 (face), 19 (face+edge),
            or 27 (full cube).
        out_file: Output path. Defaults to ``<in_file stem>_reho.nii.gz``.

    Returns:
        Path to the output NIfTI file.
    """
    in_file = Path(in_file)
    bold = Volume.load(in_file, dtype=np.float64, expected_ndim=4)
    mask = Volume.load(mask_file, dtype=np.uint8)
    bold.check_compatible(mask)

    reho_map = reho(bold.data, mask.data, cluster_size)

    if out_file is None:
        stem = in_file.name.split(".nii")[0]
        out_file = in_file.parent / f"{stem}_reho.nii.gz"
    out_file = Path(out_file)

    bold.derive(reho_map).save(out_file)
    return out_file
