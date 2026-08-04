"""Reproject and transform coordinates between coordinate reference systems (CRS).

Provides generic helpers to transform ``x``/``y`` coordinates of :class:`xarray.Dataset` and
:class:`xarray.DataArray` objects between CRSs, to derive ``lat``/``lon`` from projected
``x``/``y`` grid coordinates, and to derive projected ``x``/``y`` from geographic ``lat``/``lon``
point data.

The CRS is expected as a string compatible with :meth:`pyproj.crs.CRS.from_string` (a PROJ
string, CRS WKT string, or an authority string such as ``"EPSG:4326"``), carried on
``obj.attrs["crs"]``.
"""

# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import xarray as xr

from veriflow.constants import DEFAULT_CRS, StandardAttribute, StandardCoord, StandardDim

if TYPE_CHECKING:
    from types import ModuleType

    import pyproj

__all__ = [
    "DEFAULT_CRS",
    "add_latlon_from_crs",
    "derive_xy",
    "parse_crs",
    "reproject_to_crs",
    "transform_coordinates",
    "transform_dataset_coordinates",
]


XrObj = TypeVar("XrObj", xr.Dataset, xr.DataArray)


def _import_pyproj() -> ModuleType:
    """Import and return :mod:`pyproj`, raising a helpful error when it is not installed.

    ``pyproj`` is an optional dependency, part of the ``spatial`` extra. It is imported lazily
    so that veriflow can be used without it, and is only required when a coordinate
    transformation is actually performed.
    """
    try:
        import pyproj  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "Coordinate transformations require the optional 'pyproj' dependency, which is not "
            "installed. Install it with the 'spatial' extra, e.g. "
            "`pip install veriflow[spatial]`."
        )
        raise ImportError(msg) from exc
    return pyproj


def parse_crs(crs: str) -> pyproj.CRS:
    """Parse a CRS string into a :class:`pyproj.CRS`.

    Accepts any string compatible with :meth:`pyproj.crs.CRS.from_string`, e.g. a PROJ string,
    a CRS WKT string, or an authority string such as ``"EPSG:4326"``.
    """
    pyproj = _import_pyproj()
    return pyproj.CRS.from_string(crs)


def transform_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: str,
    target_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform ``x``/``y`` coordinate arrays from ``source_crs`` to ``target_crs``.

    Uses ``always_xy=True`` so that inputs and outputs are ordered ``(x, y)`` / ``(lon, lat)``
    regardless of the axis order defined by the CRS.
    """
    pyproj = _import_pyproj()
    transformer = pyproj.Transformer.from_crs(
        parse_crs(source_crs),
        parse_crs(target_crs),
        always_xy=True,
    )
    x_transformed, y_transformed = transformer.transform(np.asarray(x), np.asarray(y))
    return np.asarray(x_transformed), np.asarray(y_transformed)


def _crs_of(obj: XrObj, default: str | None = None) -> str:
    """Return the CRS string carried on ``obj.attrs['crs']``.

    Falls back to ``default`` when the attribute is absent. Raises when no CRS can be
    determined.
    """
    crs = obj.attrs.get(StandardAttribute.crs, default)
    if crs is None:
        msg = (
            "No CRS found: the object has no 'crs' attribute and no default was provided. "
            "Set 'crs' on the dataset attributes (a string compatible with "
            "pyproj.CRS.from_string)."
        )
        raise ValueError(msg)
    return str(crs)


def transform_dataset_coordinates(
    obj: XrObj,
    source_crs: str,
    target_crs: str,
) -> XrObj:
    """Reproject the ``x``/``y`` coordinates of ``obj`` from ``source_crs`` to ``target_crs``.

    Both ``x`` and ``y`` must be present and share the same dimensions (as is the case for
    point data along ``station``, or 2D curvilinear coordinates). Rectilinear grids, where
    ``x`` and ``y`` are 1D coordinates on separate dimensions, cannot be reprojected without
    resampling and are not supported here.

    The reprojected object carries the target CRS on its ``crs`` attribute.
    """
    if StandardDim.x not in obj.coords or StandardDim.y not in obj.coords:
        msg = "Cannot reproject: 'x' and 'y' coordinates are both required."
        raise ValueError(msg)

    x_coord = obj.coords[StandardDim.x]
    y_coord = obj.coords[StandardDim.y]
    if x_coord.dims != y_coord.dims:
        msg = (
            "Cannot reproject rectilinear grid coordinates (x and y on separate dimensions) "
            f"without resampling. Got x.dims={x_coord.dims}, y.dims={y_coord.dims}."
        )
        raise ValueError(msg)

    x_new, y_new = transform_coordinates(
        x_coord.to_numpy(),
        y_coord.to_numpy(),
        source_crs,
        target_crs,
    )
    result = obj.assign_coords(
        {
            StandardDim.x: (x_coord.dims, x_new),
            StandardDim.y: (y_coord.dims, y_new),
        },
    )
    result.attrs[StandardAttribute.crs] = target_crs
    return result


def derive_xy(
    obj: XrObj,
    target_crs: str,
    source_crs: str = DEFAULT_CRS,
) -> XrObj:
    """Derive ``x``/``y`` coordinates in ``target_crs`` from geographic ``lat``/``lon``.

    Intended for point data that only carries ``lat``/``lon`` (assumed to be in
    ``source_crs``, EPSG:4326 by default). The computed ``x``/``y`` are assigned along the
    same dimensions as ``lat``/``lon``, and the target CRS is stored on the ``crs`` attribute.
    """
    if StandardCoord.lat.name not in obj.coords or StandardCoord.lon.name not in obj.coords:
        msg = "Cannot derive x/y: 'lat' and 'lon' coordinates are both required."
        raise ValueError(msg)

    lon_coord = obj.coords[StandardCoord.lon.name]
    lat_coord = obj.coords[StandardCoord.lat.name]
    x_new, y_new = transform_coordinates(
        lon_coord.to_numpy(),
        lat_coord.to_numpy(),
        source_crs,
        target_crs,
    )
    result = obj.assign_coords(
        {
            StandardDim.x: (lon_coord.dims, x_new),
            StandardDim.y: (lat_coord.dims, y_new),
        },
    )
    result.attrs[StandardAttribute.crs] = target_crs
    return result


def add_latlon_from_crs(ds: xr.Dataset) -> xr.Dataset:
    """Derive 2D ``lat``/``lon`` coordinates from a grid's ``x``/``y`` and its CRS.

    Expects ``x`` and ``y`` as 1D dimension coordinates and a ``crs`` attribute on the dataset.
    The resulting ``lat``/``lon`` are 2D auxiliary coordinates with dimensions ``(y, x)``,
    expressed in geographic coordinates (EPSG:4326).
    """
    if StandardDim.x not in ds.coords or StandardDim.y not in ds.coords:
        msg = "Cannot derive lat/lon: 'x' and 'y' coordinates are both required."
        raise ValueError(msg)

    source_crs = _crs_of(ds)
    x = ds.coords[StandardDim.x].to_numpy()
    y = ds.coords[StandardDim.y].to_numpy()
    xx, yy = np.meshgrid(x, y)  # shape (len(y), len(x)) -> dims (y, x)
    lon, lat = transform_coordinates(xx, yy, source_crs, DEFAULT_CRS)
    return ds.assign_coords(
        {
            StandardCoord.lon.name: ((StandardDim.y, StandardDim.x), lon),
            StandardCoord.lat.name: ((StandardDim.y, StandardDim.x), lat),
        },
    )


def reproject_to_crs(obj: XrObj, target_crs: str) -> XrObj:
    """Ensure ``obj`` carries ``x``/``y`` coordinates expressed in ``target_crs``.

    The source CRS is taken from ``obj.attrs['crs']`` when present, otherwise geographic
    coordinates (EPSG:4326) are assumed for ``lat``/``lon`` point data. Dispatches to
    :func:`transform_dataset_coordinates` when ``x``/``y`` are already present, or to
    :func:`derive_xy` when only ``lat``/``lon`` are available. Objects without any spatial
    coordinates are returned unchanged.
    """
    has_xy = StandardDim.x in obj.coords and StandardDim.y in obj.coords
    has_latlon = StandardCoord.lat.name in obj.coords and StandardCoord.lon.name in obj.coords
    if not has_xy and not has_latlon:
        # Nothing spatial to reproject (e.g. results reduced over the spatial dimensions).
        return obj

    source_crs = _crs_of(obj, default=DEFAULT_CRS)

    if has_xy:
        if source_crs == target_crs:
            result = obj.copy()
            result.attrs[StandardAttribute.crs] = target_crs
            return result
        return transform_dataset_coordinates(obj, source_crs, target_crs)

    # No x/y: derive from geographic lat/lon.
    return derive_xy(obj, target_crs, source_crs=source_crs)
