"""Per-workflow metadata loading and validation.

Each workflow that depends on BIDS sidecar or NIfTI header metadata
defines a frozen dataclass here.  A ``load()`` classmethod reads,
validates, and logs the metadata *once*, before any processing begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nibabel as nib
from bids2table import load_bids_metadata

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger(__name__)

_TR_TOLERANCE = 1e-3
_TR_PLAUSIBLE_LOW = 0.1  # fastest multiband sequences ~100 ms
_TR_PLAUSIBLE_HIGH = 20.0  # very slow sparse designs


def _resolve_tr(
    *,
    sidecar_tr: float | None,
    header_tr: float | None,
    override: float | None,
) -> float:
    """Determine a single TR value from multiple sources.

    Precedence: *override* > *sidecar_tr* > *header_tr*.

    Args:
        sidecar_tr: RepetitionTime from the BIDS JSON sidecar, or *None*.
        header_tr: pixdim[4] from the NIfTI header, or *None*.
        override: CLI-provided TR that takes priority over everything.

    Returns:
        Validated TR in seconds (guaranteed > 0).

    Raises:
        ValueError: If TR cannot be determined or sources disagree.
    """
    if override is not None:
        _logger.info("Using CLI-provided TR: %.4f s", override)
        if sidecar_tr and sidecar_tr > 0 and abs(override - sidecar_tr) > _TR_TOLERANCE:
            _logger.warning(
                "CLI TR (%.4f s) differs from sidecar (%.4f s)", override, sidecar_tr
            )
        if header_tr and header_tr > 0 and abs(override - header_tr) > _TR_TOLERANCE:
            _logger.warning(
                "CLI TR (%.4f s) differs from NIfTI header (%.4f s)",
                override,
                header_tr,
            )
        return override

    if sidecar_tr is not None and sidecar_tr > 0:
        if (
            header_tr is not None
            and header_tr > 0
            and abs(sidecar_tr - header_tr) > _TR_TOLERANCE
        ):
            msg = (
                f"TR mismatch: BIDS sidecar={sidecar_tr:.4f} s, "
                f"NIfTI header={header_tr:.4f} s. "
                f"Pass --tr to resolve manually."
            )
            raise ValueError(msg)
        _logger.info("TR: %.4f s (from BIDS sidecar)", sidecar_tr)
        return sidecar_tr

    if header_tr is not None and header_tr > 0:
        _logger.warning(
            "No RepetitionTime in BIDS sidecar; "
            "falling back to NIfTI header TR: %.4f s",
            header_tr,
        )
        return header_tr

    msg = (
        "Cannot determine TR: no RepetitionTime in BIDS sidecar "
        "and NIfTI header pixdim[4] is missing or zero. "
        "Pass --tr to specify manually."
    )
    raise ValueError(msg)


def _warn_implausible_tr(tr: float) -> None:
    """Log a warning if TR falls outside the plausible fMRI range."""
    if tr < _TR_PLAUSIBLE_LOW:
        _logger.warning(
            "TR=%.4f s looks unusually short. Verify units (expected seconds).", tr
        )
    elif tr > _TR_PLAUSIBLE_HIGH:
        _logger.warning("TR=%.4f s looks unusually long. Verify this is correct.", tr)


def _validate_slice_timing(slice_timing: list[float], tr: float) -> None:
    """Raise if any slice time falls outside [0, TR).

    The BIDS spec requires SliceTiming values to be in [0, TR).
    Out-of-range values indicate a unit mismatch or corrupt sidecar
    and would cause AFNI 3dTshift to produce wrong results silently.
    """
    out_of_range = [t for t in slice_timing if t < 0 or t >= tr]
    if out_of_range:
        msg = (
            f"SliceTiming contains values outside [0, TR={tr:.4f}): "
            f"{out_of_range}. "
            f"Check sidecar units (BIDS spec requires seconds in [0, TR))."
        )
        raise ValueError(msg)


def _header_slice_timing(hdr: nib.Nifti1Header) -> list[float] | None:
    """Extract per-slice acquisition times from the NIfTI header, if present.

    Returns *None* when the header lacks the necessary fields
    (slice_code, slice_duration, dim_info).
    """
    try:
        return list(hdr.get_slice_times())
    except nib.spatialimages.HeaderDataError:
        return None


@dataclass(frozen=True)
class FunctionalMetadata:
    """Validated, immutable metadata for a single BOLD run.

    Attributes:
        tr: Repetition time in seconds (always > 0).
        slice_timing: Per-slice acquisition times from the BIDS sidecar,
            or *None* when the sidecar does not include SliceTiming.
    """

    tr: float
    slice_timing: list[float] | None

    @classmethod
    def load(
        cls,
        bold_path: Path,
        *,
        tr_override: float | None = None,
    ) -> FunctionalMetadata:
        """Load and validate BOLD metadata from sidecar and NIfTI header.

        Args:
            bold_path: Path to the raw BOLD NIfTI file.
            tr_override: CLI-provided TR that overrides sidecar/header.

        Returns:
            A frozen :class:`FunctionalMetadata` instance.

        Raises:
            ValueError: If TR cannot be determined, sources conflict,
                or SliceTiming values fall outside [0, TR).
        """
        sidecar = load_bids_metadata(bold_path)
        sidecar_tr: float | None = sidecar.get("RepetitionTime")

        hdr = nib.nifti1.load(bold_path).header
        raw_pixdim = float(hdr["pixdim"][4])  # type: ignore[index]
        header_tr: float | None = raw_pixdim if raw_pixdim > 0 else None

        tr = _resolve_tr(
            sidecar_tr=sidecar_tr,
            header_tr=header_tr,
            override=tr_override,
        )
        _warn_implausible_tr(tr)

        slice_timing: list[float] | None = sidecar.get("SliceTiming")
        if slice_timing is not None:
            _logger.info(
                "SliceTiming: from BIDS sidecar (%d slices)", len(slice_timing)
            )
            _validate_slice_timing(slice_timing, tr)
        else:
            slice_timing = _header_slice_timing(hdr)
            if slice_timing is not None:
                _logger.warning(
                    "No SliceTiming in BIDS sidecar; "
                    "falling back to NIfTI header (%d slices)",
                    len(slice_timing),
                )
                _validate_slice_timing(slice_timing, tr)
            else:
                _logger.warning(
                    "No SliceTiming in BIDS sidecar or NIfTI header; "
                    "slice timing correction will be skipped"
                )

        return cls(tr=tr, slice_timing=slice_timing)
