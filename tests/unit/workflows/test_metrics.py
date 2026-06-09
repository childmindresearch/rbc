"""Unit tests for rbc.workflows.metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from rbc.workflows import metrics as metrics_mod
from rbc.workflows.metrics import single_session_metrics

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _save_nifti(path: Path, data: np.ndarray) -> None:
    nib.nifti1.Nifti1Image(data, affine=np.eye(4)).to_filename(str(path))


def test_atlas_outputs_are_per_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each atlas's timeseries must land in its own file with its own ROI count.

    Regression for a prior single_session_metrics bug where every atlas's
    ``compute_timeseries`` call shared the same ``out_dir`` and overwrote
    the previous atlas's file; ``MetricsOutputs.timeseries[label]`` ended
    up pointing at the last-iterated atlas's data for every key.
    """
    rng = np.random.default_rng(0)
    bold = rng.standard_normal((6, 6, 6, 8))
    mask = np.ones((6, 6, 6), dtype=np.int16)

    atlas_3 = np.zeros((6, 6, 6), dtype=np.int16)
    atlas_3[0:2] = 1
    atlas_3[2:4] = 2
    atlas_3[4:6] = 3

    atlas_5 = np.zeros((6, 6, 6), dtype=np.int16)
    for i in range(5):
        atlas_5[:, i, :] = i + 1
    # last column unlabeled (kept as 0) so label set is exactly {1..5}
    atlas_5[:, 5, :] = 0

    bold_path = tmp_path / "bold.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    atlas_3_path = tmp_path / "atlas3.nii.gz"
    atlas_5_path = tmp_path / "atlas5.nii.gz"
    _save_nifti(bold_path, bold)
    _save_nifti(mask_path, mask.astype(np.float64))
    _save_nifti(atlas_3_path, atlas_3.astype(np.float64))
    _save_nifti(atlas_5_path, atlas_5.astype(np.float64))

    # Skip the scalar maps -- this test only cares about the atlas loop.
    from pathlib import Path as _Path

    counter = {"n": 0}

    def _next_scratch(name: str) -> _Path:
        counter["n"] += 1
        p = tmp_path / f"{name}_{counter['n']}.nii.gz"
        _save_nifti(p, np.zeros((6, 6, 6)))
        return p

    def _scalar_pair(*_args: object, **kwargs: object) -> tuple[_Path, _Path]:
        out_file = kwargs.get("out_file")
        alff = (
            _Path(out_file)  # type: ignore[arg-type]
            if out_file is not None
            else _next_scratch("alff")
        )
        if not alff.exists():
            _save_nifti(alff, np.zeros((6, 6, 6)))
        return alff, _next_scratch("falff")

    def _scalar_single(*_args: object, **_kwargs: object) -> _Path:
        return _next_scratch("scalar")

    def _smooth(in_path: _Path, _mask: _Path, **_kwargs: object) -> _Path:
        return in_path

    monkeypatch.setattr(metrics_mod, "compute_alff", _scalar_pair)
    monkeypatch.setattr(metrics_mod, "compute_reho", _scalar_single)
    monkeypatch.setattr(metrics_mod, "smooth", _smooth)
    monkeypatch.setattr(metrics_mod, "compute_zscore", _scalar_single)

    outputs = single_session_metrics(
        regressed_bold=bold_path,
        cleaned_bold=bold_path,
        template_brain_mask=mask_path,
        tr=2.0,
        atlas_files={"atl3": atlas_3_path, "atl5": atlas_5_path},
        smooth=6.0,
    )

    # Distinct files per atlas, never overwriting each other.
    assert outputs.timeseries["atl3"] != outputs.timeseries["atl5"]
    assert outputs.correlation_matrix["atl3"] != outputs.correlation_matrix["atl5"]

    ts3 = np.loadtxt(outputs.timeseries["atl3"], delimiter="\t")
    ts5 = np.loadtxt(outputs.timeseries["atl5"], delimiter="\t")
    assert ts3.shape == (3, 8)
    assert ts5.shape == (5, 8)

    corr3 = np.loadtxt(outputs.correlation_matrix["atl3"], delimiter="\t")
    corr5 = np.loadtxt(outputs.correlation_matrix["atl5"], delimiter="\t")
    assert corr3.shape == (3, 3)
    assert corr5.shape == (5, 5)
