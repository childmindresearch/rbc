"""Unit tests for scripts/fix_rbc_release_tr.py.

The AFNI 3dTproject call (``apply_regression_bandpass``) is exercised by RBC's
integration tests against a real runner; here we cover the parts that don't
need a container: discovery, TR detection, dry-run, and verify mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "fix_rbc_release_tr.py"
FIXTURE_RELEASE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "data"
    / "cpac_outputs"
    / "ds000001"
    / "output"
    / "pipeline_RBCv0"
)


@pytest.fixture(scope="module")
def fix_module() -> ModuleType:
    """Import the script module under a synthetic name."""
    spec = importlib.util.spec_from_file_location(
        "rbc_fix_release_tr_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=False)
def require_fixture() -> None:
    """Skip tests that need the CPAC fixture if it isn't downloaded."""
    if not FIXTURE_RELEASE.exists():
        pytest.skip(f"CPAC fixture not present at {FIXTURE_RELEASE}")


@pytest.mark.usefixtures("require_fixture")
def test_discover_runs_ds000001(fix_module: ModuleType) -> None:
    """Discovery finds the single ds000001 run and resolves all sibling inputs."""
    runs = list(fix_module.discover_runs(FIXTURE_RELEASE))
    assert len(runs) == 1
    (run,) = runs
    assert run.sub == "01"
    assert run.ses == "1"
    assert run.task == "balloonanalogrisktask"
    assert run.run == "01"
    assert run.space == "MNI152NLin6ASym"
    assert run.head_bold.name.endswith("desc-head_bold.nii.gz")
    assert run.bold_mask.name.endswith("desc-bold_mask.nii.gz")
    assert run.native_bold.name.endswith("desc-preproc_bold.nii.gz")
    assert set(run.regressors) == {"36Parameter", "aCompCor"}
    for reg_path in run.regressors.values():
        assert reg_path.exists()


@pytest.mark.usefixtures("require_fixture")
def test_detect_tr_reads_native_header(fix_module: ModuleType) -> None:
    """TR is read from the native preproc_bold header (or honors override)."""
    (run,) = fix_module.discover_runs(FIXTURE_RELEASE)
    assert fix_module._detect_tr(run.native_bold, None) == pytest.approx(2.0)
    assert fix_module._detect_tr(run.native_bold, 3.5) == 3.5


def test_detect_tr_rejects_implausible(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """A NIfTI with TR=0 raises rather than silently returning a bogus value."""
    import nibabel as nib
    import numpy as np

    bad = tmp_path / "bad.nii.gz"
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 5)), affine=np.eye(4))
    img.header.set_zooms((2.0, 2.0, 2.0, 0.0))
    nib.save(img, bad)
    with pytest.raises(ValueError, match="Implausible TR"):
        fix_module._detect_tr(bad, None)


@pytest.mark.usefixtures("require_fixture")
def test_patch_head_bold_restores_tr(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """Patching copies the file and restores TR; the original is untouched."""
    import nibabel as nib

    (run,) = fix_module.discover_runs(FIXTURE_RELEASE)
    patched = fix_module._patch_head_bold(run.head_bold, run.native_bold, tmp_path)
    src_tr = float(nib.nifti1.load(run.native_bold).header.get_zooms()[3])
    patched_tr = float(nib.nifti1.load(patched).header.get_zooms()[3])
    assert patched_tr == pytest.approx(src_tr)
    orig_tr = float(nib.nifti1.load(run.head_bold).header.get_zooms()[3])
    assert orig_tr == 0.0


@pytest.mark.usefixtures("require_fixture")
def test_verify_mode_reports_bug(
    fix_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """--verify exits 0 and reports the head_bold/native TR mismatch."""
    rc = fix_module._verify_release(FIXTURE_RELEASE)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Bug present" in out
    assert "native preproc_bold TR : 2.0000s" in out
    assert "template head_bold  TR : 0.0000s" in out


@pytest.mark.usefixtures("require_fixture")
def test_dry_run_lists_outputs(
    fix_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints the would-be output filenames and exits 0."""
    rc = fix_module.main(["--input-dir", str(FIXTURE_RELEASE), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reg-36Parameter_desc-preproc_bold.nii.gz" in out
    assert "reg-aCompCor_desc-preproc_bold.nii.gz" in out


@pytest.mark.usefixtures("require_fixture")
def test_participant_label_filters(fix_module: ModuleType) -> None:
    """--participant-label accepts both ``sub-XX`` and bare ``XX`` forms."""
    runs = list(fix_module.discover_runs(FIXTURE_RELEASE))
    assert fix_module._filter_runs(runs, ["sub-01"]) == runs
    assert fix_module._filter_runs(runs, ["01"]) == runs
    assert fix_module._filter_runs(runs, ["sub-99"]) == []


def test_discover_skips_runs_with_missing_inputs(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """A head_bold with no native preproc_bold sibling is skipped, not crashed on."""
    func_dir = tmp_path / "sub-01" / "ses-1" / "func"
    func_dir.mkdir(parents=True)
    head = (
        func_dir / "sub-01_ses-1_task-rest_space-MNI152NLin6ASym_desc-head_bold.nii.gz"
    )
    head.write_bytes(b"")
    runs = list(fix_module.discover_runs(tmp_path))
    assert runs == []
