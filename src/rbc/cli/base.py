"""Base arguments shared across all workflow CLIs."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib

from rbc_resources import BRAIN_EXTRACTION_TEMPLATES, BrainExtractionTemplates

_logger = logging.getLogger(__name__)

_VALID_RUNNERS = frozenset({"auto", "local", "docker", "podman", "singularity"})


@dataclass(frozen=True)
class BaseArgs:
    """Base (global) arguments shared across all workflow CLIs."""

    input_dir: Path
    output_dir: Path
    runner: Literal["auto", "local", "docker", "podman", "singularity"]
    participant_label: list[str]
    session_label: list[str]
    verbose: int
    tmp_dir: Path | None

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
        if tmp_dir is not None and tmp_dir.exists() and not tmp_dir.is_dir():
            raise ValueError(
                f"Temporary path exists, but is not a directory: {tmp_dir}"
            )

        return cls(
            input_dir=ns.input_dir,
            output_dir=ns.output_dir,
            runner=ns.runner,
            participant_label=ns.participant_label,
            session_label=ns.session_label,
            verbose=ns.verbose,
            tmp_dir=tmp_dir,
        )


def _validate_nifti_path(value: str) -> Path:
    """Validate and return a NIfTI file path (for use as argparse ``type=``).

    Raises:
        argparse.ArgumentTypeError: If the path does not exist or has an
            unexpected extension.
    """
    path = Path(value).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not (path.name.endswith(".nii.gz") or path.name.endswith(".nii")):
        raise argparse.ArgumentTypeError(
            f"Expected a NIfTI file (.nii or .nii.gz), got: {path.name}"
        )
    return path


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


def _validate_atlas_nifti(path: Path) -> None:
    """Validate a custom atlas NIfTI has integer labels and is 3-D.

    Args:
        path: Path to a NIfTI atlas file.

    Raises:
        ValueError: If the atlas is not a 3-D volume.
    """
    try:
        hdr = nib.nifti1.load(path).header
    except Exception:
        _logger.warning("Could not read NIfTI header for atlas %s", path)
        return
    dtype = hdr.get_data_dtype()
    if dtype.kind not in ("i", "u"):  # integer or unsigned integer
        _logger.warning(
            "Atlas %s has dtype %s (expected integer labels). "
            "Parcellation-based extraction may produce unexpected results.",
            path.name,
            dtype,
        )
    shape = hdr.get_data_shape()
    if len(shape) > 3:
        raise ValueError(
            f"Atlas {path.name} is {len(shape)}-D (expected a 3-D "
            f"parcellation). Multi-volume atlases are ambiguous; extract "
            f"the desired volume first."
        )


def _build_brain_extraction_templates(
    ns: argparse.Namespace,
) -> BrainExtractionTemplates:
    """Construct brain extraction templates from CLI namespace.

    Falls back to bundled OASIS defaults for any field not provided.
    Warns if only some of the three templates are overridden (they work
    as a matched set).
    """
    custom = (
        ns.brain_extraction_template,
        ns.brain_extraction_prob_mask,
        ns.brain_extraction_reg_mask,
    )
    n_custom = sum(v is not None for v in custom)
    if 0 < n_custom < 3:
        _logger.warning(
            "Only %d of 3 brain extraction templates provided. The three "
            "files (template, probability mask, registration mask) are "
            "designed to work as a matched set. Mixing custom and bundled "
            "templates may produce poor brain extraction results.",
            n_custom,
        )
    return BrainExtractionTemplates(
        template=_or_default(
            ns.brain_extraction_template, BRAIN_EXTRACTION_TEMPLATES.template
        ),
        probability_mask=_or_default(
            ns.brain_extraction_prob_mask, BRAIN_EXTRACTION_TEMPLATES.probability_mask
        ),
        registration_mask=_or_default(
            ns.brain_extraction_reg_mask, BRAIN_EXTRACTION_TEMPLATES.registration_mask
        ),
    )


def _or_default(value: Path | None, default: Path) -> Path:
    """Return *value* if not None, otherwise *default*."""
    return value if value is not None else default
