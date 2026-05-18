"""BIDS discovery and export for the longitudinal template workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from rbc.bids import FUNC_GROUP_ENTITIES, Suffix, bids_safe_label

if TYPE_CHECKING:
    from rbc.bids import Bids
    from rbc.workflows.longitudinal.template import LongitudinalTemplateOutputs


class TemplateInputs(NamedTuple):
    """Per-subject inputs for longitudinal template construction.

    Attributes:
        sub: Subject label.
        sessions: Per-input session labels (parallel to ``files``).
        files: Per-session preprocessed T1w brain volumes.
        bold_ref: Per-session preprocessed BOLD volumes (all tasks).
    """

    sub: str
    sessions: list[str]
    files: list[Path]
    bold_files: dict[str, Path]


def discover_template_inputs(
    df: pl.DataFrame,
) -> tuple[list[TemplateInputs], list[str]]:
    """Group preprocessed T1w brain volumes by subject for template generation.

    Filters for ``ses != 'longitudinal'`` brain-extracted T1w files and
    returns one :class:`TemplateInputs` per subject that has at least two
    sessions. Subjects with a single session are reported separately so the
    caller can warn or fail.

    Args:
        df: Full BIDS table (raw + derivatives).

    Returns:
        Tuple of (per-subject template inputs, single-session subject labels
        that were excluded).
    """
    filtered = df.filter(
        pl.col("ses") != "longitudinal",
        pl.col("datatype") == "anat",
        pl.col("desc") == "brain",
        pl.col("suffix") == "T1w",
        # Native-space brains only; the MNI-registered desc-brain T1w that
        # cross-sectional anat also writes would otherwise be picked up as
        # a second input per session, producing duplicate LTA filenames in
        # the mri_robust_template invocation.
        pl.col("space").is_null(),
    )
    bold_rows = df.filter(
        pl.col("ses") != "longitudinal",
        pl.col("datatype") == "func",
        pl.col("desc") == "preproc",
        pl.col("suffix") == "bold",
        pl.col("space").is_null(),
    )

    inputs: list[TemplateInputs] = []
    skipped: list[str] = []
    for _, sub_group in filtered.group_by("sub", maintain_order=True):
        sub = sub_group["sub"][0]
        if len(sub_group) < 2:
            skipped.append(sub)
            continue
        sessions = [row["ses"] for row in sub_group.iter_rows(named=True)]
        files = [
            Path(row["root"]) / row["path"] for row in sub_group.iter_rows(named=True)
        ]
        sub_bold = bold_rows.filter(
            (pl.col("sub") == sub) & (pl.col("ses") == sessions[0])
        ).unique(subset=(*FUNC_GROUP_ENTITIES, "root", "path"))
        # Check each task is unique, otherwise raise assertion error with details
        if sub_bold.height != sub_bold.unique().height:
            conflicts = (
                sub_bold.filter(pl.struct(FUNC_GROUP_ENTITIES).is_duplicated())
                .group_by(FUNC_GROUP_ENTITIES)
                .agg(pl.format("{}/{}", "root", "path").alias("paths"))
            )
            raise AssertionError(
                f"Found multiple non-matching grids for subject {sub}:\n"
                + "\n".join(str(dict(row)) for row in conflicts.iter_rows(named=True))
            )
        bold_files = {
            row["task"]: Path(row["root"]) / row["path"]
            for row in sub_bold.iter_rows(named=True)
        }
        inputs.append(
            TemplateInputs(
                sub=sub, sessions=sessions, files=files, bold_files=bold_files
            )
        )
    return inputs, skipped


def export_template(tpl: Bids, outputs: LongitudinalTemplateOutputs) -> None:
    """Export longitudinal template workflow outputs to BIDS-named derivatives.

    Args:
        tpl: Bids builder with ``ses="longitudinal"`` and
            ``datatype=ANAT`` bound to the subject.
        outputs: Results from the longitudinal template workflow.
    """
    tpl.save(outputs.template, suffix=Suffix.T1W)
    for btask, bold_template in outputs.bold_templates.items():
        tpl.save(bold_template, res=btask, suffix=Suffix.T1W)
    for ses, xfm in zip(outputs.sessions, outputs.transforms, strict=True):
        ses_label = bids_safe_label(ses)
        tpl.save(
            xfm,
            suffix="xfm",
            extension=".txt",
            extra={
                "from": ses_label,
                "to": "longitudinal",
                "mode": "image",
            },
        )
