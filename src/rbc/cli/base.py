"""Base arguments shared across all workflow CLIs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbc_resources import (
    ATLAS_REGISTRY,
    MNI_TEMPLATES,
    OASIS_TEMPLATES,
    BrainExtractionTemplates,
    RegistrationTemplates,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

_VALID_RUNNERS = frozenset({"local", "docker", "podman", "singularity"})
_logger = logging.getLogger("rbc")


@dataclass(frozen=True)
class BaseArgs:
    """Base (global) arguments shared across all workflow CLIs."""

    input_dir: Path
    output_dir: Path
    runner: Literal["local", "docker", "podman", "singularity"]
    participant_label: list[str]
    session_label: list[str]
    verbose: int
    tmp_dir: Path | None
    brain_extraction_templates: BrainExtractionTemplates
    templates: RegistrationTemplates
    custom_atlases: dict[str, Path]

    @classmethod
    def validate_namespace(cls, ns: argparse.Namespace) -> BaseArgs:
        """Validation of base arguments."""
        if not ns.input_dir.exists():
            raise ValueError(f"Input path does not exist: {ns.input_dir}")
        if ns.runner not in _VALID_RUNNERS:
            raise ValueError(
                f"Expected one of {_VALID_RUNNERS} for runner, got: {ns.runner!r}"
            )

        for labels, prefix in (
            (ns.participant_label, "sub-"),
            (ns.session_label, "ses-"),
        ):
            if bad := next(
                (label for label in labels if label.startswith(prefix)), None
            ):
                raise ValueError(f"Label must not start with {prefix!r}: {bad!r}")

        tmp_dir: Path | None = ns.tmp_dir
        if tmp_dir is not None and not tmp_dir.is_dir():
            raise ValueError(f"Temporary directory does not exist: {tmp_dir}")

        brain_extraction_templates = _resolve_brain_extraction_templates(ns)
        templates = _resolve_registration_templates(ns)
        custom_atlases = _parse_custom_atlases(ns.custom_atlas)
        _warn_voxel_spacing(ns)

        return cls(
            input_dir=ns.input_dir,
            output_dir=ns.output_dir,
            runner=ns.runner,
            participant_label=ns.participant_label,
            session_label=ns.session_label,
            verbose=ns.verbose,
            tmp_dir=tmp_dir,
            brain_extraction_templates=brain_extraction_templates,
            templates=templates,
            custom_atlases=custom_atlases,
        )


def _validate_atlas(atlas: str | None) -> None:
    """Validate atlas is available and exists."""
    if atlas not in ATLAS_REGISTRY:
        raise ValueError(f"Unknown atlas, got: {atlas!r}")


def _validate_task(task: str | None) -> None:
    """Validate BIDS task label contains only alphanumeric characters and '+'."""
    if task is not None and not re.fullmatch(r"[0-9a-zA-Z+]+", task):
        raise ValueError(
            f"Task must contain only alphanumeric characters and '+', got: {task!r}."
        )


def _validate_positive(value: float | int | None, name: str) -> None:
    """Validate that a numeric value is strictly positive."""
    if value is not None and value <= 0:
        raise ValueError(f"{name} should be greater than 0, got: {value!r}.")


def _validate_file(path: Path | None, flag: str) -> None:
    """Raise if a user-provided path does not exist."""
    if path is not None and not path.exists():
        raise FileNotFoundError(f"{flag}: file not found: {path}")


def _resolve_brain_extraction_templates(
    ns: argparse.Namespace,
) -> BrainExtractionTemplates:
    """Build BrainExtractionTemplates from CLI overrides, falling back to bundled."""
    be_template: Path | None = ns.brain_extraction_template
    be_prob: Path | None = ns.brain_extraction_prob_mask
    be_reg: Path | None = ns.brain_extraction_reg_mask
    _validate_file(be_template, "--brain-extraction-template")
    _validate_file(be_prob, "--brain-extraction-prob-mask")
    _validate_file(be_reg, "--brain-extraction-reg-mask")

    provided = [be_template, be_prob, be_reg]
    n_provided = sum(p is not None for p in provided)
    if 0 < n_provided < 3:
        _logger.warning(
            "Only %d of 3 brain extraction templates provided. The three "
            "templates (--brain-extraction-template, --brain-extraction-prob-mask, "
            "--brain-extraction-reg-mask) are designed to work together. "
            "Missing templates will fall back to bundled OASIS.",
            n_provided,
        )

    return BrainExtractionTemplates(
        template=be_template or OASIS_TEMPLATES.template,
        probability_mask=be_prob or OASIS_TEMPLATES.probability_mask,
        registration_mask=be_reg or OASIS_TEMPLATES.registration_mask,
    )


def _resolve_registration_templates(
    ns: argparse.Namespace,
) -> RegistrationTemplates:
    """Build RegistrationTemplates from CLI overrides, falling back to bundled."""
    anat: Path | None = ns.anat_template
    func: Path | None = ns.func_template
    func_mask: Path | None = ns.func_template_mask
    func_ref: Path | None = ns.func_template_ref
    _validate_file(anat, "--anat-template")
    _validate_file(func, "--func-template")
    _validate_file(func_mask, "--func-template-mask")
    _validate_file(func_ref, "--func-template-ref")
    return RegistrationTemplates(
        brain_1mm=anat or MNI_TEMPLATES.brain_1mm,
        brain_2mm=func or MNI_TEMPLATES.brain_2mm,
        brain_mask_2mm=func_mask or MNI_TEMPLATES.brain_mask_2mm,
        bold_ref=func_ref or MNI_TEMPLATES.bold_ref,
    )


def _parse_custom_atlases(raw: list[str] | None) -> dict[str, Path]:
    r"""Parse --custom-atlas entries into {label: path} dict.

    Accepts ``name=path`` or just ``path`` (label derived from filename stem).
    The ``=`` separator is used instead of ``:`` to avoid conflicts with
    Windows drive letters (e.g. ``C:\\...``).
    """
    from pathlib import Path

    from rbc.core.bids import bids_safe_label

    if not raw:
        return {}

    atlases: dict[str, Path] = {}
    for entry in raw:
        if "=" in entry:
            name, _, path_str = entry.partition("=")
            label = bids_safe_label(name)
            path = Path(path_str)
        else:
            path = Path(entry)
            stem = path.name.split(".")[0]
            label = bids_safe_label(stem)
        if not label:
            raise ValueError(
                f"Cannot derive a BIDS-safe label from custom atlas: {entry!r}"
            )
        if not path.exists():
            raise FileNotFoundError(f"--custom-atlas: file not found: {path}")
        if label in ATLAS_REGISTRY:
            raise ValueError(
                f"Custom atlas label {label!r} conflicts with a built-in atlas. "
                f"Use a different name: --custom-atlas othername={path}"
            )
        atlases[label] = path
    return atlases


def _warn_voxel_spacing(ns: argparse.Namespace, *, tol: float = 0.01) -> None:
    """Log a warning if custom templates have unexpected voxel spacing."""
    import nibabel as nib
    import numpy as np

    checks: list[tuple[Path | None, str, float]] = [
        (ns.anat_template, "--anat-template", 1.0),
        (ns.func_template, "--func-template", 2.0),
        (ns.func_template_mask, "--func-template-mask", 2.0),
        (ns.func_template_ref, "--func-template-ref", 2.0),
    ]
    for path, flag, expected in checks:
        if path is None:
            continue
        img = nib.nifti1.load(path)
        affine = img.affine
        sizes = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
        if any(abs(s - expected) > tol for s in sizes):
            actual = "x".join(f"{s:.2f}" for s in sizes)
            _logger.warning(
                "%s: expected ~%.0f mm voxels, got %s mm (%s)",
                flag,
                expected,
                actual,
                path,
            )
