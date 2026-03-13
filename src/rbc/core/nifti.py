"""NIfTI helpers and Volume abstraction for safe image I/O.

Provides :class:`Volume` for type-safe loading, deriving, and saving of NIfTI
images, plus lightweight metadata queries (:func:`nifti_num_volumes`,
:func:`nifti_num_slices`) that avoid loading full image data.
"""

from __future__ import annotations

import warnings
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    NanPolicy = Literal["warn", "raise", "ignore"]

__all__ = [
    "Space",
    "Units",
    "Volume",
    "nifti_num_slices",
    "nifti_num_volumes",
]


# NIfTI spatial unit codes (nibabel xyzt_units bits 0-2)
_NIB_UNIT_TO_UNITS: dict[int, Units] = {}  # populated after Units definition
_UNITS_TO_NIB_UNIT: dict[Units, int] = {}


class Space(IntEnum):
    """NIfTI coordinate-space codes (sform_code / qform_code).

    Values match the NIfTI-1 standard so they can be passed directly to
    nibabel header setters.
    """

    UNKNOWN = 0
    SCANNER = 1
    ALIGNED = 2
    TALAIRACH = 3
    MNI = 4


class Units(Enum):
    """Spatial units for voxel dimensions."""

    UNKNOWN = "unknown"
    MM = "mm"
    M = "m"
    MICRON = "um"


# Mapping between nibabel integer unit codes and Units enum.
_NIB_UNIT_TO_UNITS = {
    0: Units.UNKNOWN,
    1: Units.M,
    2: Units.MM,
    3: Units.MICRON,
}
_UNITS_TO_NIB_UNIT = {v: k for k, v in _NIB_UNIT_TO_UNITS.items()}


def _spatial_unit_from_xyzt(xyzt_units: int) -> Units:
    """Extract spatial Units from nibabel's combined xyzt_units integer."""
    spatial_code = xyzt_units & 0x07
    return _NIB_UNIT_TO_UNITS.get(spatial_code, Units.UNKNOWN)


class Volume:
    """Immutable wrapper around a NIfTI image's data and essential metadata.

    ``Volume`` enforces safe patterns for neuroimaging I/O:

    * ``dtype`` is required at load time, so callers are explicit about what
      they expect.
    * ``derive()`` builds a new Volume with correct spatial metadata, avoiding
      the common 4D-header-on-3D-data bug.
    * ``save()`` always constructs a fresh ``Nifti1Image`` from data + affine,
      guaranteeing the header shape matches the data.
    """

    __slots__ = (
        "_affine",
        "_data",
        "_qform_code",
        "_sform_code",
        "_source_path",
        "_tr",
        "_xyzt_units",
    )

    def __init__(  # noqa: D107
        self,
        data: np.ndarray,
        affine: np.ndarray,
        *,
        sform_code: int = 0,
        qform_code: int = 0,
        xyzt_units: int = 0,
        tr: float | None = None,
        source_path: Path | None = None,
    ) -> None:
        self._data = data
        self._affine = affine
        self._sform_code = sform_code
        self._qform_code = qform_code
        self._xyzt_units = xyzt_units
        self._tr = tr
        self._source_path = source_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        dtype: type[np.generic] | np.dtype | Literal["any"] = "any",
        *,
        expected_ndim: int | None = None,
    ) -> Volume:
        """Load a NIfTI image from *path*.

        Args:
            path: Filesystem path to a ``.nii`` or ``.nii.gz`` file.
            dtype: Target numpy dtype for the data array, or ``"any"`` to
                keep whatever dtype nibabel returns from ``get_fdata()``.
            expected_ndim: If set, raise ``ValueError`` when the loaded
                image's dimensionality does not match.

        Returns:
            A new :class:`Volume` instance.

        Raises:
            ValueError: If *expected_ndim* is given and does not match.
        """
        path = Path(path)
        img = nib.nifti1.load(path)
        hdr = img.header

        data = img.get_fdata()
        if dtype != "any":
            data = data.astype(dtype, copy=False)

        if expected_ndim is not None and data.ndim != expected_ndim:
            msg = f"Expected {expected_ndim}D image, got {data.ndim}D ({path.name})"
            raise ValueError(msg)

        tr: float | None = None
        if data.ndim >= 4:
            tr = float(hdr["pixdim"][4])

        return cls(
            data=data,
            affine=img.affine.copy(),
            sform_code=int(hdr["sform_code"]),
            qform_code=int(hdr["qform_code"]),
            xyzt_units=int(hdr["xyzt_units"]),
            tr=tr,
            source_path=path,
        )

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        affine: np.ndarray,
        *,
        spatial_units: Units = Units.MM,
        sform: Space = Space.UNKNOWN,
    ) -> Volume:
        """Construct a Volume from a raw numpy array and affine.

        Args:
            data: N-D array (>= 3D).
            affine: 4x4 affine matrix.
            spatial_units: Spatial units for voxel dimensions.
            sform: Coordinate-space code for sform (qform is set equal).

        Returns:
            A new :class:`Volume`.

        Raises:
            ValueError: If *affine* is not 4x4.
        """
        if affine.shape != (4, 4):
            msg = f"Affine must be 4x4, got {affine.shape}"
            raise ValueError(msg)

        nib_unit = _UNITS_TO_NIB_UNIT.get(spatial_units, 0)
        return cls(
            data=data,
            affine=affine.copy(),
            sform_code=int(sform),
            qform_code=int(sform),
            xyzt_units=nib_unit,
            tr=None,
            source_path=None,
        )

    @property
    def data(self) -> np.ndarray:
        """The image data array (read-only reference)."""
        return self._data

    @property
    def affine(self) -> np.ndarray:
        """4x4 affine matrix mapping voxel to world coordinates."""
        return self._affine

    @property
    def tr(self) -> float | None:
        """Repetition time in seconds, or None for 3D images."""
        return self._tr

    @property
    def voxel_sizes(self) -> tuple[float, ...]:
        """Voxel dimensions derived from the affine (column norms)."""
        return tuple(float(np.linalg.norm(self._affine[:3, i])) for i in range(3))

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """First three dimensions of the data array."""
        s = self._data.shape
        return (s[0], s[1], s[2])

    @property
    def sform(self) -> Space:
        """Coordinate-space code for the sform matrix."""
        try:
            return Space(self._sform_code)
        except ValueError:
            return Space.UNKNOWN

    @property
    def qform(self) -> Space:
        """Coordinate-space code for the qform matrix."""
        try:
            return Space(self._qform_code)
        except ValueError:
            return Space.UNKNOWN

    @property
    def spatial_units(self) -> Units:
        """Spatial units decoded from xyzt_units."""
        return _spatial_unit_from_xyzt(self._xyzt_units)

    @property
    def source_path(self) -> Path | None:
        """Original file path, if loaded from disk."""
        return self._source_path

    def check(
        self,
        *,
        ndim: int | None = None,
        dtype: type[np.generic] | np.dtype | None = None,
        spatial_units: Units | None = None,
        sform: Space | None = None,
        qform: Space | None = None,
        min_volumes: int | None = None,
    ) -> Volume:
        """Assert metadata expectations (chainable).

        Args:
            ndim: Expected number of dimensions.
            dtype: Expected numpy dtype.
            spatial_units: Expected spatial units.
            sform: Expected sform coordinate-space code.
            qform: Expected qform coordinate-space code.
            min_volumes: Minimum number of volumes (4th dim).

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If any expectation is violated.
        """
        src = f" ({self._source_path.name})" if self._source_path else ""

        if ndim is not None and self._data.ndim != ndim:
            msg = f"Expected {ndim}D, got {self._data.ndim}D{src}"
            raise ValueError(msg)

        if dtype is not None and not np.issubdtype(self._data.dtype, dtype):
            msg = f"Expected dtype {np.dtype(dtype)}, got {self._data.dtype}{src}"
            raise ValueError(msg)

        if spatial_units is not None and self.spatial_units != spatial_units:
            msg = (
                f"Expected spatial_units={spatial_units.value}, "
                f"got {self.spatial_units.value}{src}"
            )
            raise ValueError(msg)

        if sform is not None and self.sform != sform:
            msg = f"Expected sform={sform.name}, got {self.sform.name}{src}"
            raise ValueError(msg)

        if qform is not None and self.qform != qform:
            msg = f"Expected qform={qform.name}, got {self.qform.name}{src}"
            raise ValueError(msg)

        if min_volumes is not None:
            nvol = self._data.shape[3] if self._data.ndim >= 4 else 1
            if nvol < min_volumes:
                msg = f"Expected >= {min_volumes} volumes, got {nvol}{src}"
                raise ValueError(msg)

        return self

    def check_compatible(
        self,
        other: Volume,
        *,
        atol: float = 1e-4,
        check_sform: bool = True,
        check_units: bool = True,
    ) -> None:
        """Assert that *other* lives on the same spatial grid.

        Args:
            other: Volume to compare against.
            atol: Absolute tolerance for affine comparison.
            check_sform: Whether to require matching sform codes.
            check_units: Whether to require matching spatial units.

        Raises:
            ValueError: If the volumes are incompatible.
        """
        src_a = f" ({self._source_path.name})" if self._source_path else ""
        src_b = f" ({other._source_path.name})" if other._source_path else ""
        tag = f"{src_a} vs{src_b}"

        if self.spatial_shape != other.spatial_shape:
            msg = (
                f"Spatial shape mismatch: {self.spatial_shape} vs "
                f"{other.spatial_shape}{tag}"
            )
            raise ValueError(msg)

        if not np.allclose(self._affine, other._affine, atol=atol):
            msg = f"Affine mismatch{tag}"
            raise ValueError(msg)

        if check_sform and self.sform != other.sform:
            msg = f"sform mismatch: {self.sform.name} vs {other.sform.name}{tag}"
            raise ValueError(msg)

        if check_units and self.spatial_units != other.spatial_units:
            msg = (
                f"Spatial units mismatch: {self.spatial_units.value} vs "
                f"{other.spatial_units.value}{tag}"
            )
            raise ValueError(msg)

    def derive(self, data: np.ndarray) -> Volume:
        """Create a new Volume inheriting spatial metadata with new *data*.

        This is the core safety mechanism: it validates that the spatial
        dimensions match and sets TR to None when producing 3D output from
        a 4D source.

        Args:
            data: Array whose first 3 dims must match ``self.spatial_shape``.

        Returns:
            A new :class:`Volume` sharing the same affine and metadata.

        Raises:
            ValueError: If spatial dims don't match, data is < 3D, or
                attempting 3D -> 4D derivation.
        """
        if data.ndim < 3:
            msg = f"Derived data must be >= 3D, got {data.ndim}D"
            raise ValueError(msg)

        if data.shape[:3] != self.spatial_shape:
            msg = (
                f"Spatial shape mismatch: source {self.spatial_shape}, "
                f"derived {data.shape[:3]}"
            )
            raise ValueError(msg)

        if self._data.ndim == 3 and data.ndim > 3:
            msg = "Cannot derive 4D+ data from a 3D source (use from_array)"
            raise ValueError(msg)

        tr = self._tr if data.ndim >= 4 else None

        return Volume(
            data=data,
            affine=self._affine,
            sform_code=self._sform_code,
            qform_code=self._qform_code,
            xyzt_units=self._xyzt_units,
            tr=tr,
            source_path=self._source_path,
        )

    def replace(
        self,
        *,
        sform: Space | None = None,
        qform: Space | None = None,
        spatial_units: Units | None = None,
        tr: float | None = ...,  # type: ignore[assignment]
        affine: np.ndarray | None = None,
    ) -> Volume:
        """Return a new Volume with selected metadata overridden.

        The data array is shared (no copy). Unspecified fields are preserved.

        Args:
            sform: New sform code.
            qform: New qform code.
            spatial_units: New spatial units.
            tr: New TR value (pass ``None`` explicitly to clear).
            affine: New 4x4 affine.

        Returns:
            A new :class:`Volume`.
        """
        new_xyzt = self._xyzt_units
        if spatial_units is not None:
            nib_unit = _UNITS_TO_NIB_UNIT.get(spatial_units, 0)
            # Replace spatial bits (0-2), keep temporal bits (3-5)
            new_xyzt = (self._xyzt_units & 0x38) | nib_unit

        new_affine = affine if affine is not None else self._affine

        return Volume(
            data=self._data,
            affine=new_affine,
            sform_code=int(sform) if sform is not None else self._sform_code,
            qform_code=int(qform) if qform is not None else self._qform_code,
            xyzt_units=new_xyzt,
            tr=self._tr if tr is ... else tr,
            source_path=self._source_path,
        )

    def save(self, path: str | Path, *, nan: NanPolicy = "warn") -> Path:
        """Write this Volume to a NIfTI file.

        Constructs a fresh ``Nifti1Image`` from data + affine, guaranteeing
        the header shape always matches the data array.

        Args:
            path: Output file path.
            nan: Policy for NaN/Inf values in floating-point data:
                ``"warn"`` logs a warning, ``"raise"`` raises ValueError,
                ``"ignore"`` skips the check.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If *nan* is ``"raise"`` and data contains NaN/Inf.
        """
        path = Path(path)

        if nan != "ignore" and np.issubdtype(self._data.dtype, np.floating):
            has_bad = bool(np.isnan(self._data).any() or np.isinf(self._data).any())
            if has_bad:
                src = f" ({self._source_path.name})" if self._source_path else ""
                msg = f"Data contains NaN or Inf values{src}"
                if nan == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)

        img = nib.Nifti1Image(self._data, self._affine)
        hdr = img.header
        hdr["sform_code"] = self._sform_code
        hdr["qform_code"] = self._qform_code
        hdr.set_sform(self._affine, code=self._sform_code)
        hdr.set_qform(self._affine, code=self._qform_code)
        hdr["xyzt_units"] = self._xyzt_units

        if self._tr is not None and self._data.ndim >= 4:
            pixdim = hdr["pixdim"].copy()
            pixdim[4] = self._tr
            hdr["pixdim"] = pixdim

        img.to_filename(str(path))
        return path

    def __repr__(self) -> str:  # noqa: D105
        src = self._source_path.name if self._source_path else "memory"
        return (
            f"Volume(shape={self._data.shape}, dtype={self._data.dtype}, source={src})"
        )


def nifti_num_volumes(in_file: str | Path) -> int:
    """Return the number of volumes in a NIfTI image (returns 1 for 3-D images)."""
    shape = nib.nifti1.load(in_file).shape
    return shape[3] if len(shape) > 3 else 1


def nifti_num_slices(in_file: str | Path) -> int:
    """Return the number of slices along the slice axis in a NIfTI image."""
    img = nib.nifti1.load(in_file)
    dim_info = img.header.get_dim_info()
    slice_axis = dim_info[2]
    if slice_axis is not None:
        return img.shape[slice_axis]

    return img.shape[2] if len(img.shape) >= 3 else 1
