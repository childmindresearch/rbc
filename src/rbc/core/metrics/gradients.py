"""Connectivity gradient computation via diffusion map embedding.

Computes continuous axes of brain organization from connectivity matrices
using manifold learning.  The algorithm constructs an affinity matrix from
pairwise correlations, then applies diffusion map embedding (Coifman &
Lafon 2006) to recover low-dimensional gradients.

Vendored from BrainSpace (Vos de Wael et al. 2020) to avoid pulling in
VTK/matplotlib/nilearn.  Core math requires only numpy and scipy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.linalg import eigh

if TYPE_CHECKING:
    from typing import Literal

    AffinityKernel = Literal["cosine", "normalized_angle", "gaussian"]


class GradientResult(NamedTuple):
    """Result of :func:`compute_gradients`."""

    gradients: np.ndarray
    """(n_rois, n_components) gradient scores."""
    lambdas: np.ndarray
    """(n_components,) eigenvalues (descending)."""
    affinity: np.ndarray
    """(n_rois, n_rois) affinity matrix used for embedding."""


class GradientOutputs(NamedTuple):
    """File outputs from :func:`compute_gradients_from_files`."""

    gradients: Path
    """Path to the gradients TSV file."""
    lambdas: Path
    """Path to the eigenvalues TSV file."""


def _dominant_set(matrix: np.ndarray, k: int) -> np.ndarray:
    """Keep the top *k* entries per row, zeroing the rest.

    Matches BrainSpace's ``dominant_set`` (dense path).
    """
    n = matrix.shape[0]
    idx = np.argpartition(matrix, n - k, axis=1)
    row = np.arange(n)[:, None]
    col = idx[:, -k:]
    out = np.zeros_like(matrix)
    out[row, col] = matrix[row, col]
    return out


def compute_affinity(
    matrix: np.ndarray,
    kernel: AffinityKernel = "normalized_angle",
    sparsity: float = 0.9,
) -> np.ndarray:
    """Build a non-negative symmetric affinity matrix from a connectivity matrix.

    Follows BrainSpace's default pipeline: sparsify the input matrix first,
    then apply the kernel, then zero out negatives.

    Args:
        matrix: (n_rois, n_rois) correlation / connectivity matrix.
        kernel: Similarity kernel -- ``"cosine"``, ``"normalized_angle"``
            (default, matches BrainSpace), or ``"gaussian"``.
        sparsity: Fraction of weakest connections to zero out per row.
            Must be in [0, 1).  Default ``0.9`` retains the top 10%
            of connections (BrainSpace default).

    Returns:
        (n_rois, n_rois) non-negative symmetric affinity matrix.

    Raises:
        ValueError: If *matrix* is not 2-D square, or *sparsity* is
            outside [0, 1).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    n = matrix.shape[0]
    matrix = np.array(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Sparsify input before kernel (BrainSpace default: pre_sparsify=True)
    if sparsity > 0 and n > 1:
        k = max(1, int(n * (1.0 - sparsity)))
        matrix = _dominant_set(matrix, k)

    # Compute pairwise cosine similarity between rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / norms
    cosine_sim = normed @ normed.T
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    if kernel == "cosine":
        affinity = cosine_sim
    elif kernel == "normalized_angle":
        affinity = 1.0 - np.arccos(cosine_sim) / np.pi
    elif kernel == "gaussian":
        # RBF kernel: exp(-gamma * ||x_i - x_j||^2), gamma = 1/n_features
        norms_sq = np.sum(matrix**2, axis=1)
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (matrix @ matrix.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        gamma = 1.0 / matrix.shape[1]
        affinity = np.exp(-gamma * dist_sq)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")

    # Zero out negatives (BrainSpace: non_negative=True)
    affinity[affinity < 0] = 0.0

    return affinity


def diffusion_map_embedding(
    affinity: np.ndarray,
    n_components: int = 10,
    alpha: float = 0.5,
    diffusion_time: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diffusion map embedding from an affinity matrix.

    Follows BrainSpace's ``diffusion_mapping``.  Uses ``scipy.linalg.eigh``
    on the symmetrized operator (rather than ``eigsh`` on the asymmetric
    transition matrix) for dense numerical stability, then maps back to
    the diffusion coordinate space with the same post-processing.

    Args:
        affinity: (n_rois, n_rois) non-negative symmetric affinity matrix.
        n_components: Number of gradient components to return.
        alpha: Anisotropic diffusion parameter (0 = classical, 1 = Laplace-
            Beltrami normalization, 0.5 = default).
        diffusion_time: Diffusion time.  0 uses multi-scale diffusion maps
            (BrainSpace default).

    Returns:
        ``(gradients, lambdas)`` -- (n_rois, n_components) float64 gradient
        scores and (n_components,) eigenvalues in descending order.

    Raises:
        ValueError: If *affinity* is not 2-D square or *n_components* exceeds
            the number of ROIs minus one.
    """
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"Expected square 2D affinity, got shape {affinity.shape}")
    n = affinity.shape[0]
    if n_components >= n:
        raise ValueError(f"n_components ({n_components}) must be < n_rois ({n})")

    affinity = np.array(affinity, dtype=np.float64)

    # Anisotropic diffusion: W_alpha = D^{-alpha} W D^{-alpha}
    if alpha > 0:
        d = affinity.sum(axis=1)
        d = np.where(d == 0, 1.0, d)
        d_alpha = d ** (-alpha)
        affinity = affinity * d_alpha[:, None] * d_alpha[None, :]

    # Form symmetric operator Ms = D^{-1/2} W_alpha D^{-1/2}
    # (equivalent eigenvalues to transition matrix P = D^{-1} W_alpha,
    # but eigh on the symmetric form is more stable for dense matrices)
    d2 = affinity.sum(axis=1)
    d2 = np.where(d2 == 0, 1.0, d2)
    d_inv_sqrt = d2 ** (-0.5)
    ms = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    ms = (ms + ms.T) / 2.0

    eigenvalues, eigenvectors = eigh(ms)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Map symmetric eigenvectors back to transition-matrix space
    v = eigenvectors * d_inv_sqrt[:, None]

    # Normalize by first eigenvector (converts to diffusion coordinates)
    v /= v[:, [0]]
    eigenvalues /= eigenvalues[0]

    # Drop first (trivial) component
    eigenvalues = eigenvalues[1 : n_components + 1]
    v = v[:, 1 : n_components + 1]

    # Diffusion time scaling
    if diffusion_time <= 0:
        # Multi-scale diffusion map (BrainSpace default)
        eigenvalues = eigenvalues / (1.0 - eigenvalues)
    else:
        eigenvalues = eigenvalues**diffusion_time

    # Rescale eigenvectors by eigenvalues
    v *= eigenvalues[None, :]

    # Sign convention: largest-absolute-value element is positive
    signs = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(v.shape[1])])
    v *= signs[None, :]

    return v, eigenvalues


def compute_gradients(
    matrix: np.ndarray,
    n_components: int = 10,
    kernel: AffinityKernel = "normalized_angle",
    sparsity: float = 0.9,
    alpha: float = 0.5,
    diffusion_time: int = 0,
) -> GradientResult:
    """Compute connectivity gradients from a correlation matrix.

    Combines :func:`compute_affinity` and :func:`diffusion_map_embedding`
    into a single call.

    Args:
        matrix: (n_rois, n_rois) symmetric connectivity / correlation matrix.
        n_components: Number of gradient components.
        kernel: Affinity kernel (see :func:`compute_affinity`).
        sparsity: Sparsity fraction (see :func:`compute_affinity`).
        alpha: Anisotropic diffusion parameter.
        diffusion_time: Diffusion time parameter.

    Returns:
        :class:`GradientResult` with ``gradients``, ``lambdas``, and
        ``affinity``.

    Raises:
        ValueError: If *matrix* is not 2-D square, not symmetric, or has
            fewer than 2 ROIs.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 ROIs, got {n}")

    mat = np.array(matrix, dtype=np.float64)
    mat_no_nan = np.nan_to_num(mat, nan=0.0)
    if not np.allclose(mat_no_nan, mat_no_nan.T, atol=1e-8):
        raise ValueError("Matrix must be symmetric")

    n_components = min(n_components, n - 1)

    aff = compute_affinity(mat, kernel=kernel, sparsity=sparsity)
    gradients, lambdas = diffusion_map_embedding(
        aff, n_components=n_components, alpha=alpha, diffusion_time=diffusion_time
    )

    return GradientResult(gradients=gradients, lambdas=lambdas, affinity=aff)


def compute_gradients_from_files(
    in_file: str | Path,
    out_dir: str | Path | None = None,
    n_components: int = 10,
    kernel: AffinityKernel = "normalized_angle",
    sparsity: float = 0.9,
    alpha: float = 0.5,
    diffusion_time: int = 0,
) -> GradientOutputs:
    """Load a correlation matrix from TSV and write gradient results to disk.

    Thin wrapper around :func:`compute_gradients` that handles file I/O.

    Args:
        in_file: Path to TSV file containing a correlation matrix (as
            produced by :func:`~rbc.core.metrics.timeseries.compute_timeseries`).
        out_dir: Output directory.  Defaults to the parent directory of
            *in_file*.
        n_components: Number of gradient components.
        kernel: Affinity kernel.
        sparsity: Sparsity fraction.
        alpha: Anisotropic diffusion parameter.
        diffusion_time: Diffusion time parameter.

    Returns:
        :class:`GradientOutputs` with paths to the gradient and eigenvalue
        TSV files.
    """
    in_file = Path(in_file)
    matrix = np.loadtxt(in_file, delimiter="\t")

    result = compute_gradients(
        matrix,
        n_components=n_components,
        kernel=kernel,
        sparsity=sparsity,
        alpha=alpha,
        diffusion_time=diffusion_time,
    )

    if out_dir is None:
        out_dir = in_file.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_file.stem
    grad_path = out_dir / f"{stem}_gradients.tsv"
    lambda_path = out_dir / f"{stem}_lambdas.tsv"

    np.savetxt(grad_path, result.gradients, delimiter="\t")
    np.savetxt(lambda_path, result.lambdas.reshape(1, -1), delimiter="\t")

    return GradientOutputs(gradients=grad_path, lambdas=lambda_path)
