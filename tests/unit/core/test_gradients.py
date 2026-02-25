"""Unit tests for rbc.core.metrics.gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rbc.core.metrics.gradients import (
    compute_affinity,
    compute_gradients,
    compute_gradients_from_files,
    diffusion_map_embedding,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_two_block_corr(n_per_block: int = 20, rng_seed: int = 42) -> np.ndarray:
    """Create a correlation matrix with two clearly separated blocks.

    Within-block correlations are high (~0.8), between-block correlations
    are low (~0.1).
    """
    n = 2 * n_per_block
    corr = np.eye(n)
    rng = np.random.default_rng(rng_seed)

    for i in range(n):
        for j in range(i + 1, n):
            same_block = (i < n_per_block) == (j < n_per_block)
            val = rng.uniform(0.7, 0.9) if same_block else rng.uniform(0.0, 0.2)
            corr[i, j] = val
            corr[j, i] = val
    return corr


class TestComputeAffinity:
    """Tests for compute_affinity."""

    def test_output_shape_and_symmetry(self) -> None:
        """Output should be (n, n) and symmetric."""
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((10, 10))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, sparsity=0.0)
        assert aff.shape == (10, 10)
        np.testing.assert_allclose(aff, aff.T)

    def test_non_negative(self) -> None:
        """All affinity values should be >= 0."""
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((8, 8))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat)
        assert np.all(aff >= -1e-15)

    def test_sparsity_reduces_input(self) -> None:
        """With sparsity > 0, some entries in the input are zeroed before kernel."""
        n = 20
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((n, n))
        mat = (mat + mat.T) / 2

        aff_full = compute_affinity(mat, sparsity=0.0)
        aff_sparse = compute_affinity(mat, sparsity=0.9)

        # Sparse version should differ from full
        assert not np.allclose(aff_full, aff_sparse)

    def test_sparsity_zero_keeps_all(self) -> None:
        """sparsity=0 should not zero out any off-diagonal connections."""
        n = 10
        rng = np.random.default_rng(3)
        mat = rng.standard_normal((n, n))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, sparsity=0.0)
        off_diag = aff[~np.eye(n, dtype=bool)]
        assert np.all(off_diag > 0)

    def test_kernel_cosine(self) -> None:
        """Cosine kernel should produce non-negative values (after clamping)."""
        rng = np.random.default_rng(4)
        mat = rng.standard_normal((8, 8))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, kernel="cosine", sparsity=0.0)
        assert np.all(aff >= -1e-15)
        assert np.all(aff <= 1.0 + 1e-15)

    def test_kernel_normalized_angle(self) -> None:
        """Normalized angle kernel should produce values in [0, 1]."""
        rng = np.random.default_rng(5)
        mat = rng.standard_normal((8, 8))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, kernel="normalized_angle", sparsity=0.0)
        assert np.all(aff >= -1e-15)
        assert np.all(aff <= 1.0 + 1e-15)

    def test_kernel_gaussian(self) -> None:
        """Gaussian kernel should produce values in (0, 1]."""
        rng = np.random.default_rng(6)
        mat = rng.standard_normal((8, 8))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, kernel="gaussian", sparsity=0.0)
        assert np.all(aff >= -1e-15)
        assert np.all(aff <= 1.0 + 1e-15)

    def test_diagonal_is_one(self) -> None:
        """Diagonal should be 1.0 (self-similarity) for normalized_angle."""
        rng = np.random.default_rng(7)
        mat = rng.standard_normal((6, 6))
        mat = (mat + mat.T) / 2
        aff = compute_affinity(mat, sparsity=0.0)
        np.testing.assert_allclose(np.diag(aff), 1.0)

    def test_identity_input(self) -> None:
        """Identity matrix: one-hot rows have cosine sim = 0 off-diagonal."""
        mat = np.eye(5)
        aff = compute_affinity(mat, sparsity=0.0)
        off_diag = aff[0, 1]
        assert 0.4 < off_diag < 0.6

    def test_rejects_non_square(self) -> None:
        """Should reject non-square input."""
        with pytest.raises(ValueError, match="square"):
            compute_affinity(np.zeros((3, 4)))

    def test_rejects_invalid_sparsity(self) -> None:
        """Should reject sparsity outside [0, 1)."""
        mat = np.eye(3)
        with pytest.raises(ValueError, match="sparsity"):
            compute_affinity(mat, sparsity=1.0)
        with pytest.raises(ValueError, match="sparsity"):
            compute_affinity(mat, sparsity=-0.1)

    def test_nan_handling(self) -> None:
        """NaN values in the matrix should be treated as 0."""
        mat = np.ones((4, 4))
        mat[0, 1] = np.nan
        mat[1, 0] = np.nan
        aff = compute_affinity(mat, sparsity=0.0)
        assert not np.any(np.isnan(aff))


class TestDiffusionMapEmbedding:
    """Tests for diffusion_map_embedding."""

    def _make_affinity(self, n: int = 20, seed: int = 0) -> np.ndarray:
        """Build a simple positive affinity matrix."""
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((n, n))
        mat = (mat + mat.T) / 2
        return compute_affinity(mat, sparsity=0.5)

    def test_output_shapes(self) -> None:
        """Gradients should be (n, n_components), lambdas (n_components,)."""
        aff = self._make_affinity(20)
        grads, lams = diffusion_map_embedding(aff, n_components=5)
        assert grads.shape == (20, 5)
        assert lams.shape == (5,)

    def test_eigenvalues_descending(self) -> None:
        """Eigenvalues should be sorted in descending order."""
        aff = self._make_affinity(20)
        _, lams = diffusion_map_embedding(aff, n_components=5)
        assert np.all(np.diff(lams) <= 1e-10)

    def test_deterministic(self) -> None:
        """Same input should produce identical output."""
        aff = self._make_affinity(15)
        g1, l1 = diffusion_map_embedding(aff, n_components=3)
        g2, l2 = diffusion_map_embedding(aff, n_components=3)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(l1, l2)

    def test_block_separable(self) -> None:
        """Near-block-diagonal affinity with weak cross-links should separate."""
        n = 10
        aff = np.full((2 * n, 2 * n), 0.01)
        aff[:n, :n] = 0.5
        aff[n:, n:] = 0.5
        np.fill_diagonal(aff, 1.0)

        grads, _ = diffusion_map_embedding(aff, n_components=2)
        g1 = grads[:, 0]
        block_a = g1[:n]
        block_b = g1[n:]
        assert np.sign(block_a.mean()) != np.sign(block_b.mean())

    def test_n_components_respected(self) -> None:
        """Should return exactly n_components columns."""
        aff = self._make_affinity(20)
        for nc in [2, 5, 10]:
            grads, lams = diffusion_map_embedding(aff, n_components=nc)
            assert grads.shape[1] == nc
            assert lams.shape[0] == nc

    def test_alpha_zero(self) -> None:
        """alpha=0 should skip the anisotropic normalization step."""
        aff = self._make_affinity(15)
        g0, _ = diffusion_map_embedding(aff, n_components=3, alpha=0.0)
        g1, _ = diffusion_map_embedding(aff, n_components=3, alpha=1.0)
        assert not np.allclose(g0, g1)

    def test_diffusion_time_positive(self) -> None:
        """Positive diffusion_time should scale eigenvalues as w^t."""
        aff = self._make_affinity(15)
        g0, _l0 = diffusion_map_embedding(aff, n_components=3, diffusion_time=0)
        g1, _l1 = diffusion_map_embedding(aff, n_components=3, diffusion_time=1)
        # Different scaling → different gradient magnitudes
        assert not np.allclose(g0, g1)

    def test_rejects_non_square(self) -> None:
        """Should reject non-square affinity."""
        with pytest.raises(ValueError, match="square"):
            diffusion_map_embedding(np.zeros((3, 4)), n_components=1)

    def test_rejects_too_many_components(self) -> None:
        """Should reject n_components >= n_rois."""
        aff = self._make_affinity(5)
        with pytest.raises(ValueError, match="n_components"):
            diffusion_map_embedding(aff, n_components=5)


class TestComputeGradients:
    """Tests for the high-level compute_gradients function."""

    def test_output_types_and_shapes(self) -> None:
        """Result should be a GradientResult with correct shapes."""
        rng = np.random.default_rng(10)
        mat = rng.standard_normal((15, 15))
        mat = (mat + mat.T) / 2
        result = compute_gradients(mat, n_components=3)

        assert result.gradients.shape == (15, 3)
        assert result.lambdas.shape == (3,)
        assert result.affinity.shape == (15, 15)

    def test_two_block_first_gradient_separates(self) -> None:
        """Two-block correlation matrix: first gradient separates blocks."""
        corr = _make_two_block_corr(n_per_block=20)
        result = compute_gradients(corr, n_components=5, sparsity=0.8)

        g1 = result.gradients[:, 0]
        block_a = g1[:20]
        block_b = g1[20:]

        assert np.sign(block_a.mean()) != np.sign(block_b.mean())
        gap = abs(block_a.mean() - block_b.mean())
        within = max(block_a.std(), block_b.std())
        assert gap > 2 * within

    def test_rejects_non_symmetric(self) -> None:
        """Should reject non-symmetric matrix."""
        mat = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            compute_gradients(mat)

    def test_rejects_too_few_rois(self) -> None:
        """Should reject matrix with < 2 ROIs."""
        with pytest.raises(ValueError, match="2 ROIs"):
            compute_gradients(np.array([[1.0]]))

    def test_clamps_n_components(self) -> None:
        """n_components > n-1 should be silently clamped."""
        mat = np.eye(5)
        result = compute_gradients(mat, n_components=100)
        assert result.gradients.shape[1] == 4  # n - 1

    def test_correlation_matrix_input(self) -> None:
        """Standard correlation matrix should work end-to-end."""
        rng = np.random.default_rng(20)
        ts = rng.standard_normal((10, 50))
        corr = np.corrcoef(ts)
        result = compute_gradients(corr, n_components=5)
        assert result.gradients.shape == (10, 5)
        assert not np.any(np.isnan(result.gradients))


class TestComputeGradientsFromFiles:
    """Tests for the file I/O wrapper."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Write TSV, compute, load results."""
        corr = _make_two_block_corr(n_per_block=10)
        in_file = tmp_path / "sub-01_correlation_matrix.tsv"
        np.savetxt(in_file, corr, delimiter="\t")

        result = compute_gradients_from_files(in_file, n_components=3)

        assert result.gradients.exists()
        assert result.lambdas.exists()

        grads = np.loadtxt(result.gradients, delimiter="\t")
        assert grads.shape == (20, 3)

        lams = np.loadtxt(result.lambdas, delimiter="\t")
        assert lams.size == 3

    def test_output_naming(self, tmp_path: Path) -> None:
        """Output files should follow <stem>_gradients.tsv convention."""
        corr = _make_two_block_corr(n_per_block=5)
        in_file = tmp_path / "sub-01_correlation_matrix.tsv"
        np.savetxt(in_file, corr, delimiter="\t")

        result = compute_gradients_from_files(in_file, n_components=2)

        assert result.gradients.name == "sub-01_correlation_matrix_gradients.tsv"
        assert result.lambdas.name == "sub-01_correlation_matrix_lambdas.tsv"

    def test_custom_out_dir(self, tmp_path: Path) -> None:
        """Should write to a custom output directory."""
        corr = _make_two_block_corr(n_per_block=5)
        in_file = tmp_path / "corr.tsv"
        out_dir = tmp_path / "output"
        np.savetxt(in_file, corr, delimiter="\t")

        result = compute_gradients_from_files(in_file, out_dir=out_dir, n_components=2)

        assert result.gradients.parent == out_dir
        assert result.lambdas.parent == out_dir
