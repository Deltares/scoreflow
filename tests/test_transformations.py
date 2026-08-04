# mypy: ignore-errors
"""Tests for CRS parsing and coordinate transformation helpers."""

import numpy as np
import pytest
import xarray as xr

from veriflow.constants import DEFAULT_CRS, StandardAttribute, StandardCoord, StandardDim
from veriflow.transformations import (
    add_latlon_from_crs,
    derive_xy,
    parse_crs,
    project_to_crs,
    transform_coordinates,
    transform_dataset_coordinates,
)

pyproj = pytest.importorskip("pyproj")


# A projected CRS (Amersfoort / RD New, metre-based) used to exercise reprojection.
PROJECTED_CRS = "EPSG:28992"


pytestmark = pytest.mark.pyproj


def test_parse_crs_valid() -> None:
    """A valid authority string parses into a pyproj.CRS."""
    assert parse_crs(DEFAULT_CRS) == pyproj.CRS.from_epsg(4326)


def test_parse_crs_invalid_raises() -> None:
    """An invalid CRS string raises."""
    with pytest.raises(pyproj.exceptions.CRSError):
        parse_crs("not-a-crs")


def test_transform_coordinates_round_trip() -> None:
    """Transforming to a CRS and back recovers the original coordinates."""
    lon = np.array([4.0, 5.5, 6.9])
    lat = np.array([50.0, 51.0, 52.0])

    x, y = transform_coordinates(lon, lat, DEFAULT_CRS, PROJECTED_CRS)
    lon_back, lat_back = transform_coordinates(x, y, PROJECTED_CRS, DEFAULT_CRS)

    np.testing.assert_allclose(lon_back, lon, atol=1e-6)
    np.testing.assert_allclose(lat_back, lat, atol=1e-6)


def test_add_latlon_from_crs_creates_2d_coords() -> None:
    """Deriving lat/lon from a projected grid yields 2D (y, x) coordinates."""
    grid_x = np.array([100000.0, 150000.0, 200000.0])
    grid_y = np.array([400000.0, 450000.0])
    ds = xr.Dataset(
        coords={
            StandardDim.x: (StandardDim.x, grid_x),
            StandardDim.y: (StandardDim.y, grid_y),
        },
        attrs={StandardAttribute.crs: PROJECTED_CRS},
    )

    result = add_latlon_from_crs(ds)

    assert result[StandardCoord.lat.name].dims == (StandardDim.y, StandardDim.x)
    assert result[StandardCoord.lon.name].dims == (StandardDim.y, StandardDim.x)
    assert result[StandardCoord.lat.name].shape == (len(grid_y), len(grid_x))
    # Netherlands RD grid should map to plausible WGS84 latitudes/longitudes.
    lat_values = result[StandardCoord.lat.name].to_numpy()
    min_plausible_lat, max_plausible_lat = 40.0, 60.0
    assert np.all((lat_values > min_plausible_lat) & (lat_values < max_plausible_lat))


def test_add_latlon_from_crs_requires_crs() -> None:
    """Deriving lat/lon without a CRS attribute raises."""
    ds = xr.Dataset(
        coords={
            StandardDim.x: (StandardDim.x, np.array([1.0, 2.0])),
            StandardDim.y: (StandardDim.y, np.array([3.0, 4.0])),
        },
    )
    with pytest.raises(ValueError, match="No CRS found"):
        add_latlon_from_crs(ds)


def test_derive_xy_from_latlon() -> None:
    """Deriving x/y from lat/lon assigns coordinates along the same dimension."""
    lon = np.array([4.0, 5.0, 6.0])
    lat = np.array([50.0, 51.0, 52.0])
    da = xr.DataArray(
        data=np.arange(3.0),
        dims=["station"],
        coords={
            StandardCoord.lon.name: ("station", lon),
            StandardCoord.lat.name: ("station", lat),
        },
    )

    result = derive_xy(da, PROJECTED_CRS)

    assert result[StandardDim.x].dims == ("station",)
    assert result[StandardDim.y].dims == ("station",)
    assert result.attrs[StandardAttribute.crs] == PROJECTED_CRS
    expected_x, expected_y = transform_coordinates(lon, lat, DEFAULT_CRS, PROJECTED_CRS)
    np.testing.assert_allclose(result[StandardDim.x].to_numpy(), expected_x)
    np.testing.assert_allclose(result[StandardDim.y].to_numpy(), expected_y)


def test_transform_dataset_coordinates_rejects_rectilinear_grid() -> None:
    """Rectilinear grids (x/y on separate dims) cannot be reprojected."""
    ds = xr.Dataset(
        coords={
            StandardDim.x: (StandardDim.x, np.array([1.0, 2.0])),
            StandardDim.y: (StandardDim.y, np.array([3.0, 4.0])),
        },
        attrs={StandardAttribute.crs: DEFAULT_CRS},
    )
    with pytest.raises(ValueError, match="rectilinear"):
        transform_dataset_coordinates(ds, DEFAULT_CRS, PROJECTED_CRS)


def test_reproject_to_crs_without_spatial_coords_is_noop() -> None:
    """Objects without spatial coordinates are returned unchanged."""
    da = xr.DataArray(np.arange(3.0), dims=["time"])
    assert project_to_crs(da, PROJECTED_CRS).identical(da)


def test_reproject_to_crs_derives_xy_from_point_latlon() -> None:
    """Point lat/lon data is reprojected by deriving x/y in the target CRS."""
    da = xr.DataArray(
        data=np.arange(2.0),
        dims=["station"],
        coords={
            StandardCoord.lon.name: ("station", np.array([4.0, 5.0])),
            StandardCoord.lat.name: ("station", np.array([50.0, 51.0])),
        },
    )

    result = project_to_crs(da, PROJECTED_CRS)

    assert StandardDim.x in result.coords
    assert StandardDim.y in result.coords
    assert result.attrs[StandardAttribute.crs] == PROJECTED_CRS


def test_reproject_to_crs_same_crs_preserves_xy() -> None:
    """Reprojecting x/y data to its own CRS leaves the coordinates unchanged."""
    x = np.array([100000.0, 200000.0])
    y = np.array([400000.0, 500000.0])
    da = xr.DataArray(
        data=np.arange(2.0),
        dims=["station"],
        coords={
            StandardDim.x: ("station", x),
            StandardDim.y: ("station", y),
        },
        attrs={StandardAttribute.crs: PROJECTED_CRS},
    )

    result = project_to_crs(da, PROJECTED_CRS)

    np.testing.assert_array_equal(result[StandardDim.x].to_numpy(), x)
    np.testing.assert_array_equal(result[StandardDim.y].to_numpy(), y)
