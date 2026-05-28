"""Unit tests for scripts/fix_rbc_release_tr.py.

The AFNI ``3dTproject`` and metrics workflow calls are exercised by RBC's
integration tests against a real runner; here we cover the parts that don't
need a container: discovery, TR detection, dry-run, verify mode, atlas
resolution, and the release-style output layout.
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


def test_detect_tr_rejects_implausible(fix_module: ModuleType, tmp_path: Path) -> None:
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
def test_patch_head_bold_restores_tr(fix_module: ModuleType, tmp_path: Path) -> None:
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
    assert "Bug present in 1 run(s)" in out
    assert "Scanned 1 run(s): 1 buggy" in out
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


def test_discover_preserves_acq_entity(fix_module: ModuleType, tmp_path: Path) -> None:
    """A run with an ``acq-`` token resolves its siblings under the same token.

    Guards against the regression where ``discover_runs`` rebuilt the stem
    from sub/ses/task/run only and silently skipped every run carrying
    ``acq-VARIANT*`` (HBN/BHRC/CCNP) or ``acq-singleband`` (PNC), and 100%
    of NKI (whose runs all carry ``acq-1400VARIANT*``).
    """
    func_dir = tmp_path / "sub-NDARAA075AMK" / "ses-HBNsiteSI" / "func"
    func_dir.mkdir(parents=True)
    stem = "sub-NDARAA075AMK_ses-HBNsiteSI_task-rest_acq-VARIANTObliquity"
    space = "MNI152NLin6ASym"
    for name in (
        f"{stem}_space-{space}_desc-head_bold.nii.gz",
        f"{stem}_desc-preproc_bold.nii.gz",
        f"{stem}_space-{space}_desc-bold_mask.nii.gz",
        f"{stem}_reg-36Parameter_regressors.1D",
        f"{stem}_reg-aCompCor_regressors.1D",
    ):
        (func_dir / name).write_bytes(b"")

    runs = list(fix_module.discover_runs(tmp_path))
    assert len(runs) == 1
    (run,) = runs
    assert run.sub == "NDARAA075AMK"
    assert run.task == "rest"
    assert run.space == space
    # The reconstructed sibling paths must include the ``acq`` token.
    assert "acq-VARIANTObliquity" in run.native_bold.name
    assert "acq-VARIANTObliquity" in run.bold_mask.name
    for reg_path in run.regressors.values():
        assert "acq-VARIANTObliquity" in reg_path.name
    # ``_run_stem`` round-trips the full entity set.
    assert fix_module._run_stem(run) == stem
    assert fix_module._run_stem(run, with_space=True) == f"{stem}_space-{space}"


def test_stage_mask_normalizes_sform_qform_codes_and_affine(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """Staged copy adopts the BOLD's codes AND its sform/qform matrices.

    Reproduces the release shape: mask ships with sform_code=0 and an
    affine that differs from head_bold's by floating-point noise. The
    staged mask must end up with reference's codes, reference's affine,
    and the mask's data preserved -- otherwise AFNI 3dTproject rejects
    the pair as "NOT on the same 3D grid" once the sform_code agrees but
    the matrices don't.
    """
    import nibabel as nib
    import numpy as np

    ref_affine = np.eye(4)
    ref_affine[:3, :3] *= 2.0  # 2mm voxels
    # Mask affine has the same voxel size but a sub-ulp origin nudge,
    # mimicking what release headers actually carry.
    mask_affine = ref_affine.copy()
    mask_affine[0, 3] += 1e-6

    ref_path = tmp_path / "ref_bold.nii.gz"
    ref_img = nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), ref_affine)
    ref_img.header.set_sform(ref_affine, code=1)  # SCANNER
    ref_img.header.set_qform(ref_affine, code=1)
    ref_img.header["xyzt_units"] = 10  # mm + sec (release shape)
    nib.save(ref_img, ref_path)

    mask_path = tmp_path / "mask.nii.gz"
    mask_img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), mask_affine)
    mask_img.header.set_sform(mask_affine, code=0)  # UNKNOWN (release default)
    mask_img.header.set_qform(mask_affine, code=0)
    mask_img.header["xyzt_units"] = 0  # release ships some masks unitless
    nib.save(mask_img, mask_path)

    work_dir = tmp_path / "work"
    staged = fix_module._stage_mask(mask_path, ref_path, work_dir)
    assert staged.parent == work_dir
    assert staged.name == mask_path.name

    staged_img = nib.nifti1.load(staged)
    staged_hdr = staged_img.header
    assert int(staged_hdr["sform_code"]) == 1
    assert int(staged_hdr["qform_code"]) == 1
    # ``xyzt_units`` must survive verbatim -- AFNI's grid check rejects
    # (mm, sec) vs (mm, unknown-time) as different grids.
    assert int(staged_hdr["xyzt_units"]) == 10
    np.testing.assert_array_equal(staged_img.affine, ref_affine)
    np.testing.assert_array_equal(staged_hdr.get_sform(), ref_affine)
    np.testing.assert_array_equal(staged_hdr.get_qform(), ref_affine)
    # Data preserved (still a binary 1-mask).
    np.testing.assert_array_equal(
        np.asarray(staged_img.dataobj),
        np.ones((4, 4, 4), dtype=np.uint8),
    )


def test_stage_mask_rejects_shape_mismatch(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """Shape mismatches require resampling -- header alone can't fix them."""
    import nibabel as nib
    import numpy as np

    affine = np.eye(4)
    ref_path = tmp_path / "ref_bold.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), affine), ref_path
    )

    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 5), dtype=np.uint8), affine), mask_path)

    with pytest.raises(ValueError, match="shape"):
        fix_module._stage_mask(mask_path, ref_path, tmp_path / "work")


def test_resolve_atlases_finds_release_atlases(fix_module: ModuleType) -> None:
    """``_resolve_atlases`` returns release-name keys for all bundled atlases."""
    atlases = fix_module._resolve_atlases()
    # Spot-check that the renaming between release and rbc_resources lands the
    # right files (HOCPATh25 -> HarvardOxfordcort..., Slab -> Slab907).
    assert atlases["AAL"].name == "atlas-AAL_space-MNI152NLin6_res-2_dseg.nii.gz"
    assert (
        atlases["HOCPATh25"].name
        == "atlas-HarvardOxfordcortMaxprobThr25_space-MNI152NLin6_res-2_dseg.nii.gz"
    )
    assert atlases["Slab"].name == "atlas-Slab907_space-MNI152NLin6_res-2_dseg.nii.gz"
    assert (
        atlases["Schaefer2018p1000n17"].name
        == "atlas-Schaefer2018_space-MNI152NLin6_res-2_"
        "desc-1000Parcels17NetworksOrder_dseg.nii.gz"
    )
    for path in atlases.values():
        assert path.exists()


def test_metric_output_paths_matches_release_layout(
    fix_module: ModuleType, tmp_path: Path
) -> None:
    """``_metric_output_paths`` enumerates exactly the release-shaped names."""
    run = fix_module.Run(
        sub="01",
        ses="1",
        task="balloonanalogrisktask",
        run="01",
        space="MNI152NLin6ASym",
        head_bold=(
            tmp_path / "sub-01_ses-1_task-balloonanalogrisktask_run-01_"
            "space-MNI152NLin6ASym_desc-head_bold.nii.gz"
        ),
        bold_mask=tmp_path / "mask.nii.gz",
        native_bold=tmp_path / "native.nii.gz",
        regressors={},
    )
    atlases = {"AAL": tmp_path / "aal.nii.gz", "Yeo7": tmp_path / "yeo.nii.gz"}
    paths = fix_module._metric_output_paths(tmp_path, run, "36Parameter", atlases)
    names = {p.name for p in paths}

    stem_sp = (
        "sub-01_ses-1_task-balloonanalogrisktask_run-01_"
        "space-MNI152NLin6ASym_reg-36Parameter"
    )
    expected_scalars = {
        f"{stem_sp}_desc-{v}_{m}.nii.gz"
        for m in ("alff", "falff", "reho")
        for v in ("sm6", "smZstd", "zstd")
    }
    assert expected_scalars.issubset(names)

    stem_no_sp = "sub-01_ses-1_task-balloonanalogrisktask_run-01"
    for atl in ("AAL", "Yeo7"):
        base = f"{stem_no_sp}_atlas-{atl}_space-MNI152NLin6ASym_reg-36Parameter"
        assert f"{base}_desc-Mean_timeseries.1D" in names
        assert f"{base}_desc-PearsonNilearn_correlations.tsv" in names
        assert f"{base}_desc-PartialNilearn_correlations.tsv" in names

    # 3 scalars * 3 variants + 2 atlases * 3 outputs
    assert len(paths) == 9 + 6


def test_export_metrics_writes_release_layout(
    fix_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Land every release file and transpose timeseries to ``.1D`` orientation.

    Verifies both the file layout and that RBC's (n_rois, n_timepoints) source
    is transposed to AFNI's ``.1D`` (n_timepoints, n_rois) convention on
    write.
    """
    import numpy as np

    from rbc.workflows.metrics import MetricsOutputs

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def make(name: str) -> Path:
        p = src_dir / name
        p.write_bytes(b"FAKE")
        return p

    # RBC's compute_timeseries layout: rows = ROIs, cols = timepoints.
    rng = np.random.default_rng(0)
    aal_ts = rng.random((3, 5))
    yeo_ts = rng.random((2, 7))
    aal_path = src_dir / "aal_ts.tsv"
    yeo_path = src_dir / "yeo_ts.tsv"
    np.savetxt(aal_path, aal_ts, delimiter="\t")
    np.savetxt(yeo_path, yeo_ts, delimiter="\t")

    metrics = MetricsOutputs(
        alff=make("alff_raw.nii.gz"),
        falff=make("falff_raw.nii.gz"),
        alff_smooth=make("alff_sm.nii.gz"),
        falff_smooth=make("falff_sm.nii.gz"),
        alff_zscored=make("alff_smZ.nii.gz"),
        falff_zscored=make("falff_smZ.nii.gz"),
        reho=make("reho_raw.nii.gz"),
        reho_smooth=make("reho_sm.nii.gz"),
        reho_zscored=make("reho_smZ.nii.gz"),
        timeseries={"AAL": aal_path, "Yeo7": yeo_path},
        correlation_matrix={
            "AAL": make("aal_corr.tsv"),
            "Yeo7": make("yeo_corr.tsv"),
        },
    )

    def fake_zscore(raw: Path, _mask: Path) -> Path:
        p = src_dir / f"{Path(raw).stem}_zscored.nii.gz"
        p.write_bytes(b"ZSTD")
        return p

    def fake_partial(_ts: Path, out: Path) -> None:
        out.write_bytes(b"PARTIAL")

    monkeypatch.setattr(fix_module, "compute_zscore", fake_zscore)
    monkeypatch.setattr(fix_module, "_compute_partial_correlation", fake_partial)

    run = fix_module.Run(
        sub="01",
        ses="1",
        task="balloonanalogrisktask",
        run="01",
        space="MNI152NLin6ASym",
        head_bold=(
            tmp_path / "sub-01_ses-1_task-balloonanalogrisktask_run-01_"
            "space-MNI152NLin6ASym_desc-head_bold.nii.gz"
        ),
        bold_mask=tmp_path / "mask.nii.gz",
        native_bold=tmp_path / "native.nii.gz",
        regressors={},
    )
    atlases = {"AAL": tmp_path / "aal.nii.gz", "Yeo7": tmp_path / "yeo.nii.gz"}

    fix_module._export_metrics(
        metrics,
        out_dir,
        run,
        "36Parameter",
        template_brain_mask=tmp_path / "mask.nii.gz",
        atlases=atlases,
    )

    expected = fix_module._metric_output_paths(out_dir, run, "36Parameter", atlases)
    missing = [p.name for p in expected if not p.exists()]
    assert not missing, f"missing release files: {missing}"

    # Orientation check: source is (3, 5)/(2, 7); release ``.1D`` must be the
    # transpose: (5, 3)/(7, 2). Critically, this is what AFNI's ``.1D``
    # consumers and the published release expect.
    stem = "sub-01_ses-1_task-balloonanalogrisktask_run-01"
    aal_dst = (
        out_dir / f"{stem}_atlas-AAL_space-MNI152NLin6ASym_reg-36Parameter_"
        "desc-Mean_timeseries.1D"
    )
    yeo_dst = (
        out_dir / f"{stem}_atlas-Yeo7_space-MNI152NLin6ASym_reg-36Parameter_"
        "desc-Mean_timeseries.1D"
    )
    np.testing.assert_allclose(np.loadtxt(aal_dst), aal_ts.T)
    np.testing.assert_allclose(np.loadtxt(yeo_dst), yeo_ts.T)


@pytest.mark.usefixtures("require_fixture")
def test_process_run_skips_when_outputs_exist(
    fix_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With both BOLD and metric files pre-staged, no AFNI/metrics calls happen."""
    (run,) = fix_module.discover_runs(FIXTURE_RELEASE)
    out_func_dir = tmp_path / run.head_bold.parent.relative_to(FIXTURE_RELEASE)
    out_func_dir.mkdir(parents=True)
    stem = fix_module._run_stem(run, with_space=True)
    base_stem = fix_module._run_stem(run)
    atlases = {"AAL": tmp_path / "aal.nii.gz", "Yeo7": tmp_path / "yeo.nii.gz"}
    for reg_set in run.regressors:
        bold = out_func_dir / f"{stem}_reg-{reg_set}_desc-preproc_bold.nii.gz"
        reg = out_func_dir / f"{base_stem}_reg-{reg_set}_desc-bandpassed_regressors.1D"
        bold.write_bytes(b"x")
        reg.write_bytes(b"x")
        for p in fix_module._metric_output_paths(out_func_dir, run, reg_set, atlases):
            p.write_bytes(b"x")

    def _fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("expensive call should not be made when outputs exist")

    monkeypatch.setattr(fix_module, "apply_regression_bandpass", _fail)
    monkeypatch.setattr(fix_module, "apply_regression", _fail)
    monkeypatch.setattr(fix_module, "bandpass_regressor_file", _fail)
    monkeypatch.setattr(fix_module, "single_session_metrics", _fail)

    fix_module._process_run(
        run,
        FIXTURE_RELEASE,
        tmp_path,
        tmp_path / "work",
        bandpass=(0.01, 0.1),
        tr_override=2.0,
        overwrite=False,
        skip_metrics=False,
        atlases=atlases,
    )


@pytest.mark.usefixtures("require_fixture")
def test_process_run_with_skip_metrics_only_checks_bold(
    fix_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``skip_metrics=True`` skips even when no metric files exist, given BOLD."""
    (run,) = fix_module.discover_runs(FIXTURE_RELEASE)
    out_func_dir = tmp_path / run.head_bold.parent.relative_to(FIXTURE_RELEASE)
    out_func_dir.mkdir(parents=True)
    stem = fix_module._run_stem(run, with_space=True)
    base_stem = fix_module._run_stem(run)
    for reg_set in run.regressors:
        bold = out_func_dir / f"{stem}_reg-{reg_set}_desc-preproc_bold.nii.gz"
        reg = out_func_dir / f"{base_stem}_reg-{reg_set}_desc-bandpassed_regressors.1D"
        bold.write_bytes(b"x")
        reg.write_bytes(b"x")

    def _fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("nothing should be re-run with skip_metrics + bold present")

    monkeypatch.setattr(fix_module, "apply_regression_bandpass", _fail)
    monkeypatch.setattr(fix_module, "apply_regression", _fail)
    monkeypatch.setattr(fix_module, "single_session_metrics", _fail)

    fix_module._process_run(
        run,
        FIXTURE_RELEASE,
        tmp_path,
        tmp_path / "work",
        bandpass=(0.01, 0.1),
        tr_override=2.0,
        overwrite=False,
        skip_metrics=True,
        atlases={},
    )
