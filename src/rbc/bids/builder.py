"""BIDS schema constants, utilities, and path builder.

Re-exports auto-generated schema definitions from ``_schema`` and provides
the :class:`Bids` builder for composing BIDS entity specifications.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rbc.bids._schema import (
    _STANDARD_ENTITIES,
    BIDS_VERSION,
    BidsEntities,
    BIDSFile,
    Datatype,
    EntityKwargs,
    Extension,
    Modality,
    Suffix,
    TemplateSpace,
    bids_name,
    bids_name_from_entities,
    bids_path,
    bids_path_from_entities,
    bids_safe_label,
    extract_entities,
    parse_bids_name,
)

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

__all__ = [
    "BIDS_VERSION",
    "_STANDARD_ENTITIES",
    "BIDSFile",
    "Bids",
    "BidsEntities",
    "Datatype",
    "EntityKwargs",
    "Extension",
    "Modality",
    "Suffix",
    "TemplateSpace",
    "bids_name",
    "bids_name_from_entities",
    "bids_path",
    "bids_path_from_entities",
    "bids_safe_label",
    "extract_entities",
    "parse_bids_name",
]

_logger = logging.getLogger(__name__)

_SENTINEL: str | None = object()  # type: ignore[assignment]


@dataclass(frozen=True)
class Bids:
    """Immutable BIDS entity accumulator for export and query.

    Build up a BIDS specification via :meth:`derive`, then resolve it
    with :meth:`save` (export a file), :meth:`save_dir` (export a directory),
    or :meth:`find` (query a bids2table DataFrame).

    Examples::

        func = pipe_ctx.bids(datatype=Datatype.FUNC, entities=ents)
        func.save(outputs.sbref, suffix=Suffix.SBREF)
        func.save(outputs.bold, suffix=Suffix.BOLD, desc="preproc")

        mni = func.derive(space=TemplateSpace.MNI152NLIN6ASYM)
        mni.save(outputs.template_bold, suffix=Suffix.BOLD, desc="preproc")

        anat = pipe_ctx.bids(datatype=Datatype.ANAT)
        t1w = anat.find(anat_df, suffix=Suffix.T1W, desc="brain")
    """

    _sub: str
    _ses: str | None
    _output_dir: Path
    _datatype: str | None = None
    _entities: dict[str, str | int] = field(default_factory=dict)
    _extra: dict[str, str | int] | None = None

    def derive(
        self,
        *,
        sub: str | None = None,
        ses: str | None = _SENTINEL,
        output_dir: Path | None = None,
        datatype: str | None = _SENTINEL,
        extra: dict[str, str | int] | None = None,
        entities: EntityKwargs | None = None,
        **overrides: str | int,
    ) -> Bids:
        """Derive a new builder inheriting all state, merging overrides.

        Args:
            sub: Override the BIDS subject label.
            ses: Override the BIDS session label (pass ``None`` to clear).
            output_dir: Override the output directory.
            datatype: Override the BIDS datatype directory.
            extra: Non-standard entities to merge with existing extra.
            entities: Bulk entity dict to merge (e.g. from
                :func:`extract_entities`).
            **overrides: Individual entity overrides (e.g. ``space="MNI152"``).

        Returns:
            A new :class:`Bids` with merged state.
        """
        merged: dict[str, str | int] = {
            **self._entities,
            **(entities or {}),  # type: ignore[dict-item]
            **overrides,
        }
        merged_extra = (
            {**(self._extra or {}), **(extra or {})} if self._extra or extra else None
        )
        return Bids(
            _sub=sub if sub is not None else self._sub,
            _ses=ses if ses is not _SENTINEL else self._ses,
            _output_dir=output_dir if output_dir is not None else self._output_dir,
            _datatype=datatype if datatype is not _SENTINEL else self._datatype,
            _entities=merged,
            _extra=merged_extra,
        )

    def path(
        self,
        *,
        suffix: str,
        extension: str = ".nii.gz",
        extra: dict[str, str | int] | None = None,
        **overrides: str | int,
    ) -> Path:
        """Return the resolved BIDS output path without copying.

        Useful for checking expected output locations or verifying
        naming in tests without file I/O.

        Args:
            suffix: BIDS suffix (e.g. ``"T1w"``, ``"bold"``).
            extension: File extension including leading dot.
            extra: Non-standard entities (merged with session extra).
            **overrides: Per-call entity overrides (e.g. ``desc="preproc"``).

        Returns:
            Absolute path where the derivative would be written.
        """
        from pathlib import Path as _Path

        rel = self._build_path(
            suffix=suffix, extension=extension, extra=extra, overrides=overrides
        )
        return _Path(self._output_dir / rel)

    def save(
        self,
        src: Path,
        *,
        suffix: str,
        extension: str = ".nii.gz",
        extra: dict[str, str | int] | None = None,
        **overrides: str | int,
    ) -> Path:
        """Copy *src* to a BIDS-named derivative path.

        Args:
            src: Source file to copy.
            suffix: BIDS suffix (e.g. ``"T1w"``, ``"bold"``).
            extension: File extension including leading dot.
            extra: Non-standard entities (merged with session extra).
            **overrides: Per-call entity overrides (e.g. ``desc="preproc"``).

        Returns:
            Path to the copied output file.
        """
        dest = self.path(suffix=suffix, extension=extension, extra=extra, **overrides)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            _logger.warning(f"{str(dest)!r} already exists, file will be overwritten")
        shutil.copy2(src, dest)
        return dest

    def save_dir(
        self,
        src_dir: Path,
        *,
        suffix: str,
        extension: str = "",
        extra: dict[str, str | int] | None = None,
        **overrides: str | int,
    ) -> Path:
        """Copy a directory to a BIDS-named derivative path.

        Args:
            src_dir: Source directory to copy.
            suffix: BIDS suffix.
            extension: File extension (usually empty for directories).
            extra: Non-standard entities (merged with session extra).
            **overrides: Per-call entity overrides.

        Returns:
            Path to the copied output directory.
        """
        dest = self.path(suffix=suffix, extension=extension, extra=extra, **overrides)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            _logger.warning(
                f"{str(dest)!r} already exists, directory contents may be overwritten"
            )
        shutil.copytree(src_dir, dest, dirs_exist_ok=True)
        return dest

    def _merge_query(
        self,
        extra: dict[str, str | int] | None,
        has: list[str] | None,
        without: list[str] | None,
        overrides: dict[str, str | int | bool],
    ) -> tuple[dict[str, str | int | bool], dict[str, str | int] | None]:
        """Merge entity filters and extra dicts for find/find_all."""
        merged: dict[str, str | int | bool] = {**self._entities, **overrides}
        if has:
            for key in has:
                if key not in merged:
                    merged[key] = True
        if without:
            for key in without:
                merged[key] = False
        merged_extra = (
            {**(self._extra or {}), **(extra or {})} if self._extra or extra else None
        )
        return merged, merged_extra

    def find(
        self,
        df: pl.DataFrame,
        *,
        suffix: str | None = None,
        extension: str = "",
        extra: dict[str, str | int] | None = None,
        has: list[str] | None = None,
        without: list[str] | None = None,
        **overrides: str | int | bool,
    ) -> Path | None:
        """Find a single BIDS file matching accumulated entities.

        Args:
            df: bids2table DataFrame to search.
            suffix: BIDS suffix to match.
            extension: File extension to match.
            extra: Non-standard entity filters (merged with session extra).
            has: Entity keys that must be present (any value).
            without: Entity keys that must be absent (null).
            **overrides: Per-call entity filters.

        Returns:
            Path to the matching file, or ``None`` if no match found.

        Raises:
            ValueError: If multiple matches found.
        """
        from rbc.bids.query import find_file

        merged, merged_extra = self._merge_query(extra, has, without, overrides)
        return find_file(
            df,
            sub=self._sub,
            ses=self._ses,
            datatype=self._datatype,
            suffix=suffix,
            extension=extension,
            extra=merged_extra,
            entities=merged,
        )

    def expect(
        self,
        df: pl.DataFrame,
        *,
        suffix: str | None = None,
        extension: str = "",
        extra: dict[str, str | int] | None = None,
        has: list[str] | None = None,
        without: list[str] | None = None,
        **overrides: str | int | bool,
    ) -> Path:
        """Find a single BIDS file, raising if not found.

        Like :meth:`find` but raises :class:`FileNotFoundError` instead
        of returning ``None``.

        Args:
            df: bids2table DataFrame to search.
            suffix: BIDS suffix to match.
            extension: File extension to match.
            extra: Non-standard entity filters (merged with session extra).
            has: Entity keys that must be present (any value).
            without: Entity keys that must be absent (null).
            **overrides: Per-call entity filters.

        Returns:
            Path to the matching BIDS file.

        Raises:
            FileNotFoundError: If no matching file found.
            ValueError: If multiple matches found.
        """
        result = self.find(
            df,
            suffix=suffix,
            extension=extension,
            extra=extra,
            has=has,
            without=without,
            **overrides,
        )
        if result is None:
            merged, merged_extra = self._merge_query(extra, has, without, overrides)
            raise FileNotFoundError(
                f"Expected BIDS file not found: "
                f"sub={self._sub!r}, ses={self._ses!r}, "
                f"datatype={self._datatype!r}, suffix={suffix!r}, "
                f"entities={merged!r}, extra={merged_extra!r}"
            )
        return result

    def find_all(
        self,
        df: pl.DataFrame,
        *,
        suffix: str | None = None,
        extension: str = "",
        extra: dict[str, str | int] | None = None,
        has: list[str] | None = None,
        without: list[str] | None = None,
        **overrides: str | int | bool,
    ) -> list[Path]:
        """Find all BIDS files matching accumulated entities.

        Args:
            df: bids2table DataFrame to search.
            suffix: BIDS suffix to match.
            extension: File extension to match.
            extra: Non-standard entity filters (merged with session extra).
            has: Entity keys that must be present (any value).
            without: Entity keys that must be absent (null).
            **overrides: Per-call entity filters.

        Returns:
            List of matching file paths (may be empty).
        """
        from rbc.bids.query import find_files

        merged, merged_extra = self._merge_query(extra, has, without, overrides)
        return find_files(
            df,
            sub=self._sub,
            ses=self._ses,
            datatype=self._datatype,
            suffix=suffix,
            extension=extension,
            extra=merged_extra,
            entities=merged,
        )

    def _build_path(
        self,
        *,
        suffix: str,
        extension: str,
        extra: dict[str, str | int] | None,
        overrides: dict[str, str | int],
    ) -> str:
        """Build BIDS relative path from accumulated + per-call state."""
        if self._datatype is None:
            raise ValueError(
                "Cannot build a BIDS path without a datatype. "
                "Set datatype when creating or deriving the Bids builder."
            )
        ents: dict[str, str | int | None] = {"sub": self._sub, "ses": self._ses}
        ents.update(self._entities)
        ents.update(overrides)
        merged_extra = (
            {**(self._extra or {}), **(extra or {})} if self._extra or extra else None
        )
        return str(
            bids_path_from_entities(
                ents,
                suffix=suffix,
                extension=extension,
                datatype=self._datatype,
                extra=merged_extra,
            )
        )
